import polars as pl
import pytest

from georeset_corpus_assessment.legend import Legend, load_legend

LEGEND_ROWS = [
    {
        "Code": 0,
        "EUNIS1": "URBAN",
        "EUNIS2": None,
        "EUNIS3": None,
        "EUNIS1_name": "Urban",
        "EUNIS2_name": None,
        "EUNIS3_name": None,
    },
    {
        "Code": 1000,
        "EUNIS1": "MA2",
        "EUNIS2": None,
        "EUNIS3": None,
        "EUNIS1_name": "Littoral",
        "EUNIS2_name": None,
        "EUNIS3_name": None,
    },
    {
        "Code": 1100,
        "EUNIS1": "MA2",
        "EUNIS2": "MA21",
        "EUNIS3": None,
        "EUNIS1_name": "Littoral",
        "EUNIS2_name": "Arctic",
        "EUNIS3_name": None,
    },
    {
        "Code": 1101,
        "EUNIS1": "MA2",
        "EUNIS2": "MA21",
        "EUNIS3": "MA211",
        "EUNIS1_name": "Littoral",
        "EUNIS2_name": "Arctic",
        "EUNIS3_name": "Saltmarsh",
    },
    {
        "Code": 1200,
        "EUNIS1": "MA2",
        "EUNIS2": "MA22",
        "EUNIS3": None,
        "EUNIS1_name": "Littoral",
        "EUNIS2_name": "Atlantic",
        "EUNIS3_name": "  ",
    },
]


@pytest.fixture
def legend_table() -> pl.DataFrame:
    return pl.DataFrame(
        LEGEND_ROWS,
        schema_overrides={
            "EUNIS2": pl.String,
            "EUNIS3": pl.String,
            "EUNIS2_name": pl.String,
            "EUNIS3_name": pl.String,
        },
    )


@pytest.fixture
def legend(legend_table: pl.DataFrame) -> Legend:
    return load_legend(legend_table)
