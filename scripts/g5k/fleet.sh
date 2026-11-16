#!/usr/bin/env bash
# Run on a frontend. Applies the fleet rules for this site: cancel waits over 9 minutes,
# then submit workers that fit inside the current time window. Light work only.
# Usage: fleet.sh <site> <want> <owner> <owners> [elsewhere] [failures] [successes] [running 0|1]
set -euo pipefail

SITE="${1:?site}"
WANT="${2:?want}"
OWNER="${3:?owner}"
OWNERS="${4:?owners}"
ELSEWHERE="${5:-0}"
FAILURES="${6:-0}"
SUCCESSES="${7:-0}"
RUNNING="${8:-0}"

STATE="$HOME/georeset-fleet"
WORK="$HOME/georeset-work"
OUT="$HOME/georeset-results-10k"
LIMIT=10000
SHARDS=200
mkdir -p "$STATE"
cd "$HOME/GeoReSeT-Corpus-Assessment"

oarstat -u nflandre -f > "$STATE/jobs-$SITE.txt" 2>/dev/null || true
FLAG=""
[ "$RUNNING" = "1" ] && FLAG="--running"
ACTIONS="$(.venv/bin/georeset fleet --site "$SITE" --jobs "$STATE/jobs-$SITE.txt" \
  --want "$WANT" --health "$STATE/health-$SITE.json" \
  --now "$(date +%Y-%m-%dT%H:%M:%S)" --elsewhere "$ELSEWHERE" \
  --failures "$FAILURES" --successes "$SUCCESSES" $FLAG)"
echo "$ACTIONS"

while read -r kind a b c; do
  case "$kind" in
    cancel)
      oardel "$a" > /dev/null 2>&1 || true
      ;;
    submit)
      WALL="$(printf '%d:%02d:00' $((b / 60)) $((b % 60)))"
      for _ in $(seq "$a"); do
        if [ "$SITE" = "lille" ]; then
          oarsub -q default -n "georeset-w-$SITE" -p "gpu_model = 'A100-SXM4-40GB'" \
            -l "gpu=1,walltime=$WALL" \
            "bash scripts/g5k/work.sh $WORK $OUT $LIMIT $SHARDS $c $OWNER $OWNERS" > /dev/null
        else
          oarsub -q default -t exotic -n "georeset-w-$SITE" \
            -l "gpu=1,walltime=$WALL" \
            "bash scripts/g5k/work.sh $WORK $OUT $LIMIT $SHARDS $c $OWNER $OWNERS" > /dev/null
        fi
      done
      ;;
  esac
done <<< "$ACTIONS"
