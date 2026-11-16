# EUNIS evaluation with LFM3-1.7B

The model reads one sentence and names the finest-level EUNIS classes of the
polygon it describes. Results roll up to coarser levels. Everything runs on
Grid'5000; this machine only edits code and runs lint and tests.

## Pipeline

1. **Pool** (`georeset prepare`): original English sentences flagged as land use,
   linked to exactly one polygon that has an EUNIS label. Gold codes are the
   classes whose share is at least 0.1. Each sentence gets its H3 cell at
   resolution 5, from its coordinates.
2. **Order** (`round_robin`): seeded round-robin across H3 cells. Every round takes
   one sentence per cell that still has sentences, so any prefix stays balanced.
3. **Run** (`georeset run`): greedy generation with reasoning, up to 4,096 tokens,
   zero-shot prompt with the full finest-level legend. Each batch is written as
   one Parquet shard. A rerun skips sentences already in a shard.
4. **Score** (`georeset score`): micro and macro F1 at levels 1, 2 and 3. A parent
   class is positive when any of its children is positive.
5. **Publish** (`georeset publish`): upload the Parquet files to the results
   dataset.

Finest-level classes come from the legend's leaves. Unparseable or truncated
answers count as empty predictions, and the reason is recorded.

## Pinned inputs

- Source dataset `NoeFlandre/GeoReSeT-Corpus`, revision in `settings.py`.
- Model `LiquidAI/LFM3-1.7B`, revision in `settings.py`.
- Seed 42.

## Grid'5000 runbook

Work from the Nancy frontend. The GPUs are Tesla T4s, so the model runs in float16.
Heavy work runs only inside OAR jobs, never on the frontend. Pass a work
directory explicitly, outside the home quota when possible.

```bash
# One time, on the frontend
git clone https://github.com/NoeFlandre/GeoReSeT-Corpus-Assessment.git ~/GeoReSeT-Corpus-Assessment
cd ~/GeoReSeT-Corpus-Assessment && bash scripts/g5k/setup.sh

# Build the pool (night, CPU)
oarsub -n georeset-prepare -l /nodes=1/core=8,walltime=04:00:00 -t night \
  "bash scripts/g5k/prepare.sh <WORK>"

# Smoke test: 1,000 sentences on one GPU
oarsub -n georeset-run -l gpu=1,walltime=01:00:00 \
  "bash scripts/g5k/run.sh <WORK> <OUT> 1000"

# Score, then publish with the token file
oarsub -n georeset-score -l /nodes=1/core=1,walltime=00:15:00 \
  "bash scripts/g5k/score.sh <WORK> <OUT>"
oarsub -n georeset-publish -l /nodes=1/core=1,walltime=00:30:00 \
  "bash scripts/g5k/publish.sh <OUT>"
```

The publish job reads the write token from `~/.config/georeset/hf_token`, a file
you create and keep readable only by you.

Check `usagepolicycheck` before submitting. Daytime work is limited on weekdays
between 09:00 and 19:00, so run large jobs with `-t night`.
