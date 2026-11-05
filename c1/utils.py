"""Core utilities for the C1 project.

Provides deterministic seeding, atomic file operations, config validation,
resumable JSONL I/O with truncated-line tolerance, SHA-256 hashing, and
run metadata tracking.

Traces to SPEC §2, §4, §15, §16.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Any, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding (SPEC §4, §15)
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (if loaded).

    All randomness in the pipeline flows from explicit config seeds.
    PyTorch is only seeded if it has already been imported, maintaining
    zero top-level torch dependency.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Configuration Schema & Loader (SPEC §2, §4)
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Path configuration."""

    model_config = ConfigDict(extra="forbid")
    data_dir: str = "data"
    out_dir: str = "outputs"


class CorpusConfig(BaseModel):
    """Corpus generation configuration."""

    model_config = ConfigDict(extra="forbid")
    n_notes: int = 300
    generator_seed: int = 2026


class SplitConfig(BaseModel):
    """Dev/test split configuration."""

    model_config = ConfigDict(extra="forbid")
    dev_fraction: float = 0.30
    seed: int = 2026


class ModelConfig(BaseModel):
    """Single model specification."""

    model_config = ConfigDict(extra="forbid")
    hf_id: str
    context: int


class QuantisationConfig(BaseModel):
    """4-bit quantization configuration."""

    model_config = ConfigDict(extra="forbid")
    type: str = "nf4"
    compute_dtype: str = "float16"
    double_quant: bool = True


class GreedyConfig(BaseModel):
    """Greedy decoding parameters."""

    model_config = ConfigDict(extra="forbid")
    do_sample: bool = False
    max_new_tokens: int = 200


class SampleConfig(BaseModel):
    """Sampled decoding parameters (S1)."""

    model_config = ConfigDict(extra="forbid")
    k: int = 10
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 200
    mode: str = "batched"
    seed: int = 2026


class DecodingConfig(BaseModel):
    """Decoding configurations for greedy and sampled runs."""

    model_config = ConfigDict(extra="forbid")
    greedy: GreedyConfig = Field(default_factory=GreedyConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)


class S2Config(BaseModel):
    """S2 verbalized scoring configuration."""

    model_config = ConfigDict(extra="forbid")
    max_new_tokens: int = 200


class S4Config(BaseModel):
    """S4 verification configuration."""

    model_config = ConfigDict(extra="forbid")
    batch_size: int = 1


class RandomControlConfig(BaseModel):
    """Random withholding control parameters."""

    model_config = ConfigDict(extra="forbid")
    repeats: int = 100
    seed: int = 7


class BootstrapConfig(BaseModel):
    """Bootstrap configuration."""

    model_config = ConfigDict(extra="forbid")
    B: int = 1000
    seed: int = 42
    ci: float = 0.95


class Config(BaseModel):
    """Master configuration schema matching configs/c1.yaml (SPEC §2)."""

    model_config = ConfigDict(extra="forbid")
    run_id: str = "c1_run01"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: dict[str, ModelConfig]
    quantisation: QuantisationConfig = Field(default_factory=QuantisationConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    s2: S2Config = Field(default_factory=S2Config)
    s4: S4Config = Field(default_factory=S4Config)
    coverage_targets: list[float] = Field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6]
    )
    primary_target: float = 0.8
    random_control: RandomControlConfig = Field(default_factory=RandomControlConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    ece_bins: int = 10
    k_sensitivity: list[int] = Field(default_factory=lambda: [3, 5, 10])


def load_config(path: str | Path) -> Config:
    """Load and validate pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated Config model instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the configuration fails schema checks.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary")

    return Config.model_validate(data)


# ---------------------------------------------------------------------------
# Atomic File Writers (AGENTS.md §4, SPEC §4)
# ---------------------------------------------------------------------------


def write_json_atomic(path: str | Path, data: Any, indent: int = 2) -> None:
    """Atomically write data as JSON using a tempfile and os.replace.

    Flushes and fsyncs prior to replacement to guarantee durability.

    Args:
        path: Destination file path.
        data: JSON-serializable object or Pydantic model.
        indent: JSON indentation spaces.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        data = data.model_dump()

    tmp_prefix = f".{target.name}.tmp."
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=tmp_prefix,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            json.dump(data, tmp_file, indent=indent, default=str)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            tmp_file.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    os.replace(tmp_path, target)


def write_csv_atomic(
    path: str | Path,
    data: Sequence[dict[str, Any]] | Any,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Atomically write tabular data as CSV using a tempfile and os.replace.

    Supports pandas DataFrame or a sequence of dicts.

    Args:
        path: Destination file path.
        data: pandas DataFrame or sequence of dicts.
        fieldnames: Optional column sequence if data is a list of dicts.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_prefix = f".{target.name}.tmp."
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=tmp_prefix,
        newline="",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            # Check if pandas DataFrame
            if hasattr(data, "to_csv"):
                data.to_csv(tmp_file, index=False)
            else:
                rows = list(data)
                if not rows and not fieldnames:
                    cols = []
                elif fieldnames:
                    cols = list(fieldnames)
                else:
                    cols = list(rows[0].keys())

                writer = csv.DictWriter(tmp_file, fieldnames=cols)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            tmp_file.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    os.replace(tmp_path, target)


# ---------------------------------------------------------------------------
# Resumable JSONL I/O (AGENTS.md §3, SPEC §4, §15)
# ---------------------------------------------------------------------------


def append_jsonl(path: str | Path, record: dict[str, Any] | BaseModel) -> None:
    """Append a single record to a JSONL file with flush and fsync.

    Args:
        path: Target JSONL file path.
        record: Dictionary or Pydantic model to serialize.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(record, BaseModel):
        data = record.model_dump()
    else:
        data = record

    serialized = json.dumps(data, ensure_ascii=False) + "\n"
    with open(target, "a", encoding="utf-8") as f:
        f.write(serialized)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl_resume(
    path: str | Path, clean_corrupted_tail: bool = True
) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating a truncated final line upon Colab disconnect.

    If the final non-empty line cannot be parsed as valid JSON, it is logged and
    ignored. If `clean_corrupted_tail` is True, the file is automatically truncated
    at the end of the last complete valid line to ensure subsequent appends remain clean.
    Corrupted lines anywhere earlier in the file raise ValueError.

    Args:
        path: Path to the JSONL file.
        clean_corrupted_tail: Whether to rewrite the file to strip any trailing
            corrupt bytes.

    Returns:
        List of parsed JSON records.

    Raises:
        ValueError: If an intermediate line (not the last) is corrupt.
    """
    target = Path(path)
    if not target.is_file():
        return []

    with open(target, "rb") as f:
        content = f.read()

    if not content:
        return []

    lines = content.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    valid_bytes_length = 0

    for i, line_bytes in enumerate(lines):
        stripped = line_bytes.strip()
        if not stripped:
            valid_bytes_length += len(line_bytes)
            continue

        try:
            record = json.loads(line_bytes.decode("utf-8"))
            records.append(record)
            valid_bytes_length += len(line_bytes)
        except Exception as err:
            # Check whether there are any subsequent non-whitespace lines
            is_last = (i == len(lines) - 1) or all(
                not subsequent.strip() for subsequent in lines[i + 1 :]
            )
            if is_last:
                logger.warning(
                    "Tolerating truncated final line in %s: %r",
                    target,
                    line_bytes[:100],
                )
                if clean_corrupted_tail:
                    with open(target, "wb") as f_out:
                        f_out.write(content[:valid_bytes_length])
                        f_out.flush()
                        os.fsync(f_out.fileno())
                break
            else:
                raise ValueError(
                    f"Corrupted record at line {i + 1} in {target}: {err}"
                ) from err

    return records


def get_completed_ids(path: str | Path, id_key: str = "id") -> set[str]:
    """Retrieve all completed record IDs from a resumable JSONL file.

    Args:
        path: Path to the JSONL file.
        id_key: Dictionary key containing the unique note or sample identifier.

    Returns:
        Set of string IDs already recorded.
    """
    records = read_jsonl_resume(path)
    return {str(r[id_key]) for r in records if id_key in r}


# ---------------------------------------------------------------------------
# Cryptographic Hashing (SPEC §1.2, §4, §5.1)
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hexdigest of a file in 64KB blocks.

    Args:
        path: Path to the file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_string(s: str) -> str:
    """Compute SHA-256 hexdigest of a string.

    Args:
        s: Input text string.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Reproducibility Metadata Writer (SPEC §4, §15, §16)
# ---------------------------------------------------------------------------


def get_git_commit() -> str | None:
    """Get the current git commit SHA, or None if unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit if commit else None
    except Exception:
        return None


def get_gpu_name() -> str:
    """Detect available GPU device name without forcing torch/CUDA load."""
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if output:
            return output.splitlines()[0]
    except Exception:
        pass

    return "CPU"


def get_library_versions() -> dict[str, str]:
    """Collect installed versions of critical libraries without importing them."""
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pytest",
        "pydantic",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def write_run_meta(
    out_dir: str | Path,
    config: Config | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write or update run_meta.json in out_dir.

    Records config snapshot, seeds, git commit, GPU name, library versions,
    file SHA-256s, and timestamps per SPEC §4.

    Args:
        out_dir: Directory where run_meta.json should reside.
        config: Configuration instance or dictionary.
        extra: Additional stage-specific metadata to merge.

    Returns:
        The full metadata dictionary that was saved.
    """
    out_path = Path(out_dir) / "run_meta.json"

    meta: dict[str, Any] = {}
    if out_path.is_file():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as err:
            logger.warning("Could not read existing run_meta.json at %s: %s", out_path, err)

    # Core environment details
    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    meta["python_version"] = platform.python_version()
    meta["platform"] = platform.platform()
    meta["git_commit"] = get_git_commit()
    meta["gpu_name"] = get_gpu_name()
    meta["library_versions"] = get_library_versions()

    # Configuration and seeds
    if config is not None:
        cfg_dict = config.model_dump() if isinstance(config, BaseModel) else dict(config)
        meta["config"] = cfg_dict
        meta["seeds"] = {
            "corpus_generator_seed": cfg_dict.get("corpus", {}).get("generator_seed"),
            "split_seed": cfg_dict.get("split", {}).get("seed"),
            "sample_seed": cfg_dict.get("decoding", {}).get("sample", {}).get("seed"),
            "random_control_seed": cfg_dict.get("random_control", {}).get("seed"),
            "bootstrap_seed": cfg_dict.get("bootstrap", {}).get("seed"),
        }

    # Merge extras
    if extra:
        for k, v in extra.items():
            if k == "file_hashes" and "file_hashes" in meta and isinstance(meta["file_hashes"], dict):
                meta["file_hashes"].update(v)
            else:
                meta[k] = v

    write_json_atomic(out_path, meta)
    return meta

