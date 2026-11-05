"""Shim module forwarding to c1.codes to resolve vendor imports.

Traces to SPEC §1.3.
"""

from __future__ import annotations

from c1.codes import is_well_formed, normalize

__all__ = ["normalize", "is_well_formed"]
