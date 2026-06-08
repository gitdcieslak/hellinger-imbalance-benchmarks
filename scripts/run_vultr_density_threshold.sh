#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_SHARDS="${SEED_SHARDS:-0-4 5-9 10-14 15-19}"
MODELS="${MODELS:-mlp_weighted_bce,mlp_weighted_bce_dropout_0_1}"
SKEW_RATIO="${SKEW_RATIO:-100}"
MINORITY_COUNT="${MINORITY_COUNT:-100}"
SCP_TARGET="${SCP_TARGET:-user@your-workstation:/path/to/destination}"

SANITY_CSV="results/topology/density_threshold_sweep_sanity.csv"
FINAL_CSV="results/topology/density_threshold_sweep.csv"
SHARD_DIR="results/topology/density_threshold_shards"
REPORT_MD="reports/topology/density_threshold_sweep_summary.md"
EXPECTED_SANITY_DATA_ROWS=4
EXPECTED_FULL_CSV_LINES=2241

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run scripts/bootstrap_vultr.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p results/topology reports/topology "$SHARD_DIR"

echo "==> Running density-threshold sanity test"
"$PYTHON_BIN" scripts/run_density_separability_factorial.py \
  --preset density_threshold \
  --minority-covs 0.05,0.10 \
  --centroid-distances 0.5,1.0 \
  --models mlp_weighted_bce \
  --seeds 0 \
  --skew-ratio "$SKEW_RATIO" \
  --minority-count 10 \
  --output "$SANITY_CSV"

sanity_rows=$(($(wc -l < "$SANITY_CSV") - 1))
if [ "$sanity_rows" -ne "$EXPECTED_SANITY_DATA_ROWS" ]; then
  echo "ERROR: sanity data row count $sanity_rows != expected $EXPECTED_SANITY_DATA_ROWS"
  exit 1
fi
echo "==> Sanity row count OK: $sanity_rows data rows"

echo "==> Running sharded full density-threshold sweep"
for shard in $SEED_SHARDS; do
  shard_name="$(printf '%s' "$shard" | tr ',' '_' | tr '-' '_')"
  shard_csv="$SHARD_DIR/density_threshold_sweep_seeds_${shard_name}.csv"
  echo "==> Shard seeds=$shard -> $shard_csv"
  "$PYTHON_BIN" scripts/run_density_separability_factorial.py \
    --preset density_threshold \
    --models "$MODELS" \
    --seeds "$shard" \
    --skew-ratio "$SKEW_RATIO" \
    --minority-count "$MINORITY_COUNT" \
    --output "$shard_csv" &
done
wait


echo "==> Merging shard CSVs into $FINAL_CSV"
first=1
rm -f "$FINAL_CSV"
for shard_csv in "$SHARD_DIR"/*.csv; do
  if [ "$first" -eq 1 ]; then
    cat "$shard_csv" > "$FINAL_CSV"
    first=0
  else
    tail -n +2 "$shard_csv" >> "$FINAL_CSV"
  fi
done

full_lines=$(wc -l < "$FINAL_CSV")
if [ "$full_lines" -ne "$EXPECTED_FULL_CSV_LINES" ]; then
  echo "ERROR: full CSV line count $full_lines != expected $EXPECTED_FULL_CSV_LINES"
  echo "Note: expected line count includes the header; expected data rows are 2240."
  exit 1
fi
echo "==> Full row count OK: $full_lines CSV lines including header"

echo "==> Generating density-threshold report"
"$PYTHON_BIN" scripts/report_density_separability_factorial.py \
  --input "$FINAL_CSV" \
  --output-md "$REPORT_MD"

echo ""
echo "Density-threshold sweep complete."
echo "Main outputs:"
echo "  $FINAL_CSV"
echo "  $REPORT_MD"
echo ""
echo "Pull results back with:"
echo "  scp $(hostname):$(pwd)/$FINAL_CSV $SCP_TARGET/"
echo "  scp $(hostname):$(pwd)/$REPORT_MD $SCP_TARGET/"
echo "  rsync -av $(hostname):$(pwd)/reports/topology/ $SCP_TARGET/reports/topology/"
