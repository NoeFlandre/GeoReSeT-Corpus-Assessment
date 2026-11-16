from georeset_corpus_assessment.parsing import parse_prediction

VALID = frozenset({"MA211", "MA22", "URBAN"})


def test_reads_identifiers_from_final_line() -> None:
    text = "The saltmarsh is clear.\nFINAL: MA211, MA22"
    prediction = parse_prediction(text, VALID)
    assert prediction.identifiers == ("MA211", "MA22")
    assert prediction.invalid == ()
    assert prediction.error is None


def test_uses_the_last_final_line() -> None:
    text = "FINAL: URBAN\nOn reflection:\nFINAL: MA22"
    assert parse_prediction(text, VALID).identifiers == ("MA22",)


def test_rejects_unknown_tokens_and_keeps_order_without_duplicates() -> None:
    text = "FINAL: MA22, forest, MA22, URBAN, , MA999"
    prediction = parse_prediction(text, VALID)
    assert prediction.identifiers == ("MA22", "URBAN")
    assert prediction.invalid == ("forest", "MA999")


def test_none_gives_an_empty_valid_answer() -> None:
    prediction = parse_prediction("No habitat.\nFINAL: none", VALID)
    assert prediction.identifiers == ()
    assert prediction.error is None


def test_empty_final_body_counts_as_none() -> None:
    assert parse_prediction("FINAL:", VALID).identifiers == ()


def test_markdown_emphasis_around_final_is_tolerated() -> None:
    prediction = parse_prediction("**FINAL:** `MA211`.", VALID)
    assert prediction.identifiers == ("MA211",)


def test_missing_final_line_is_an_error() -> None:
    prediction = parse_prediction("I think it is MA211.", VALID)
    assert prediction.identifiers == ()
    assert prediction.error == "no_final_line"
