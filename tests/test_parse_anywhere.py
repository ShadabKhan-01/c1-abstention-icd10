"""Parser reads codes anywhere in a line (SPEC §5.2, ASSUMPTIONS A28/A29)."""

from c1.parse import parse_greedy_output, parse_s2_output


def test_codes_in_brackets_and_after_labels() -> None:
    assert parse_greedy_output("Gout (M10.0)Varicose veins (I83.91)")[0] == ["M10.0", "I83.91"]
    assert parse_greedy_output("- Eczema (L40)")[0] == ["L40"]
    assert parse_greedy_output("- ICD10CM: I10.9 (Essential hypertension)")[0] == ["I10.9"]


def test_header_only_output_fails_to_parse() -> None:
    pred, _, failed = parse_greedy_output("ICD-10-CM Diagnosis Codes:")
    assert pred == [] and failed


def test_three_char_token_after_a_word_is_text() -> None:
    assert parse_greedy_output("Vitamin B12 deficiency anemia (D51.9)")[0] == ["D51.9"]
    assert parse_greedy_output("I10")[0] == ["I10"]


def test_span_points_at_the_code() -> None:
    text = "x\n- Eczema (L40)"
    span = parse_greedy_output(text)[1][0]
    assert text[span["start"]:span["end"]] == "L40"


def test_s2_fallback_line_formats() -> None:
    raw = "Gout (M10.0): 90\nI10, 85%\n- E11.9 (Type 2 diabetes) - 70/100\nF90.0"
    scores, failed = parse_s2_output(raw, ["M10.0", "I10", "E11.9", "F90.0"])
    assert scores == {"M10.0": 0.9, "I10": 0.85, "E11.9": 0.7, "F90.0": None}
    assert failed == ["F90.0"]
