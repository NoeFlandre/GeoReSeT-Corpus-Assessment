import polars as pl

from georeset_corpus_assessment.sampling import round_robin, stable_key


def make_pool() -> pl.DataFrame:
    cells = {"A": 3, "B": 1, "C": 2}
    rows = [
        {"id": f"{cell}{index}", "cell": cell}
        for cell, count in cells.items()
        for index in range(count)
    ]
    return pl.DataFrame(rows)


def test_stable_key_depends_on_seed_and_value() -> None:
    assert stable_key(1, "x") == stable_key(1, "x")
    assert stable_key(1, "x") != stable_key(2, "x")
    assert stable_key(1, "x") != stable_key(1, "y")


def test_each_round_takes_one_row_per_cell_still_holding_rows() -> None:
    order = round_robin(make_pool(), seed=7)
    per_round = order.group_by("round").agg(pl.col("cell").sort()).sort("round")
    assert per_round["cell"].to_list() == [
        ["A", "B", "C"],
        ["A", "C"],
        ["A"],
    ]


def test_every_cell_round_is_distinct_and_contiguous() -> None:
    order = round_robin(make_pool(), seed=7)
    for cell in ["A", "B", "C"]:
        rounds = order.filter(pl.col("cell") == cell)["round"].sort().to_list()
        assert rounds == list(range(len(rounds)))


def test_rounds_are_nondecreasing_along_positions() -> None:
    order = round_robin(make_pool(), seed=7).sort("position")
    rounds = order["round"].to_list()
    assert rounds == sorted(rounds)


def test_any_prefix_covers_every_cell_before_repeating() -> None:
    order = round_robin(make_pool(), seed=7).sort("position")
    prefix = order.head(3)
    assert sorted(prefix["cell"].to_list()) == ["A", "B", "C"]


def test_order_is_reproducible_for_a_seed() -> None:
    first = round_robin(make_pool(), seed=3)["id"].to_list()
    second = round_robin(make_pool(), seed=3)["id"].to_list()
    assert first == second


def test_round_robin_keeps_the_other_columns() -> None:
    pool = make_pool().with_columns(pl.col("id").str.to_uppercase().alias("text"))
    order = round_robin(pool, seed=1)
    assert "text" in order.columns
    assert order.filter(pl.col("id") == "A0")["text"].to_list() == ["A0"]
