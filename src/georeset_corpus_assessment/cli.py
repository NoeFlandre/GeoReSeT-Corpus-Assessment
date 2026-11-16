"""Command-line entry point. Each command runs on Grid'5000, not on a laptop."""

from __future__ import annotations

import argparse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from huggingface_hub import create_repo, snapshot_download, upload_folder

from georeset_corpus_assessment import settings
from georeset_corpus_assessment.engine import VllmEngine
from georeset_corpus_assessment.legend import load_legend
from georeset_corpus_assessment.pipeline import RunConfig, run, score_results
from georeset_corpus_assessment.prepare import (
    LABEL_COLUMNS,
    SENTENCE_COLUMNS,
    build_pool,
    legend_table_from_csv,
    write_pool,
)
from georeset_corpus_assessment.sampling import round_robin


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Parse the command line and dispatch to one command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


def _build_parser() -> argparse.ArgumentParser:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="georeset")
    commands = parser.add_subparsers(required=True)

    prepare_cmd = commands.add_parser("prepare", help="Build the evaluation pool")
    prepare_cmd.add_argument("--work", type=Path, required=True)
    prepare_cmd.set_defaults(handler=_prepare)

    run_cmd = commands.add_parser("run", help="Label sentences with the model")
    run_cmd.add_argument("--work", type=Path, required=True)
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--limit", type=int, required=True)
    run_cmd.add_argument("--batch-size", type=int, default=500)
    run_cmd.add_argument("--gpu", default="unknown")
    run_cmd.set_defaults(handler=_run)

    score_cmd = commands.add_parser("score", help="Score a results folder per level")
    score_cmd.add_argument("--work", type=Path, required=True)
    score_cmd.add_argument("--out", type=Path, required=True)
    score_cmd.set_defaults(handler=_score)

    publish_cmd = commands.add_parser("publish", help="Upload Parquet files to HF")
    publish_cmd.add_argument("--out", type=Path, required=True)
    publish_cmd.set_defaults(handler=_publish)
    return parser


def _prepare(args: argparse.Namespace) -> None:  # pragma: no cover
    source = Path(
        snapshot_download(
            settings.SOURCE_DATASET,
            repo_type="dataset",
            revision=settings.SOURCE_DATASET_REVISION,
            allow_patterns=["data/*/*/*.parquet", "polygons_labels_shares/*.parquet"],
        ),
    )
    sentences = pl.read_parquet(
        sorted((source / "data").glob("*/*/*.parquet")),
        columns=SENTENCE_COLUMNS,
    )
    labels = pl.read_parquet(
        sorted((source / "polygons_labels_shares").glob("*.parquet")),
        columns=LABEL_COLUMNS,
    )
    write_pool(build_pool(sentences, labels), args.work)
    legend_csv = args.work / "eunis_legend_detailed.csv"
    urllib.request.urlretrieve(settings.LEGEND_URL, legend_csv)  # noqa: S310
    legend_table_from_csv(legend_csv).write_parquet(args.work / "legend.parquet")


def _run(args: argparse.Namespace) -> None:  # pragma: no cover
    pool = pl.read_parquet(args.work / "pool.parquet")
    legend = load_legend(pl.read_parquet(args.work / "legend.parquet"))
    order = round_robin(pool, settings.SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    order.write_parquet(args.out / "manifest.parquet")
    config = RunConfig(
        model_id=settings.MODEL_ID,
        model_revision=settings.MODEL_REVISION,
        dataset_revision=settings.SOURCE_DATASET_REVISION,
        seed=settings.SEED,
        gpu=args.gpu,
    )
    engine = VllmEngine(
        settings.MODEL_ID,
        settings.MODEL_REVISION,
        settings.MAX_NEW_TOKENS,
    )
    written = run(
        order,
        legend,
        engine,
        args.out,
        config,
        limit=args.limit,
        batch_size=args.batch_size,
        created_at=datetime.now(tz=UTC).isoformat(),
    )
    print(f"wrote {written} rows to {args.out}")  # noqa: T201


def _score(args: argparse.Namespace) -> None:  # pragma: no cover
    legend = load_legend(pl.read_parquet(args.work / "legend.parquet"))
    metrics = score_results(args.out, legend)
    metrics.write_parquet(args.out / "metrics.parquet")
    print(metrics)  # noqa: T201


def _publish(args: argparse.Namespace) -> None:  # pragma: no cover
    create_repo(settings.RESULTS_DATASET, repo_type="dataset", exist_ok=True)
    upload_folder(
        folder_path=args.out,
        repo_id=settings.RESULTS_DATASET,
        repo_type="dataset",
        allow_patterns=["*.parquet"],
    )
