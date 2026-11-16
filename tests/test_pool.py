import h3
import polars as pl

from georeset_corpus_assessment.pool import add_cells, attach_gold, select_sentences


def make_sentences() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["keep", "nllb", "not_land", "multi"],
            "text": ["t1", "t2", "t3", "t4"],
            "translation": ["original", "nllb", "original", "original"],
            "landuse": [True, True, False, True],
            "polygon_ids": [["way/1"], ["way/2"], ["way/3"], ["way/4", "way/5"]],
            "region": ["r", "r", "r", "r"],
            "lat": [45.0, 45.0, 45.0, 45.0],
            "lon": [5.0, 5.0, 5.0, 5.0],
        },
    )


def test_select_keeps_only_original_landuse_single_polygon() -> None:
    selected = select_sentences(make_sentences())
    assert selected["id"].to_list() == ["keep"]
    assert selected["polygon_id"].to_list() == ["way/1"]


def test_attach_gold_keeps_codes_at_or_above_minimum_share() -> None:
    pool = select_sentences(make_sentences())
    labels = pl.DataFrame(
        {
            "polygon_id": ["way/1"],
            "eunis_code": [[1101, 1200, 0]],
            "eunis_shares": [[0.5, 0.09, 0.1]],
        },
    )
    gold = attach_gold(pool, labels)
    assert gold["gold_codes"].to_list() == [[1101, 0]]
    assert "eunis_code" not in gold.columns


def test_attach_gold_drops_polygons_without_any_code_above_minimum() -> None:
    pool = select_sentences(make_sentences())
    labels = pl.DataFrame(
        {
            "polygon_id": ["way/1"],
            "eunis_code": [[1101]],
            "eunis_shares": [[0.05]],
        },
    )
    assert attach_gold(pool, labels).height == 0


def test_attach_gold_drops_polygons_without_labels() -> None:
    pool = select_sentences(make_sentences())
    labels = pl.DataFrame(
        {"polygon_id": ["way/9"], "eunis_code": [[1101]], "eunis_shares": [[1.0]]},
    )
    assert attach_gold(pool, labels).height == 0


def test_add_cells_uses_h3_resolution_five() -> None:
    pool = select_sentences(make_sentences())
    cells = add_cells(pool)["cell"].to_list()
    assert cells == [h3.latlng_to_cell(45.0, 5.0, 5)]
