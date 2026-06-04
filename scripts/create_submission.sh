#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-dataset}"
OUTPUT="${2:-submission.csv}"
CV_REPORT="${3:-cv_report.csv}"

missing=0
for file in train.csv test.csv sample_submission.csv; do
  if [[ ! -f "${DATA_DIR}/${file}" ]]; then
    echo "Missing ${DATA_DIR}/${file}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  cat >&2 <<EOF

Cannot create ${OUTPUT} because the competition CSV files are not present.
Place the files like this and rerun:

  ${DATA_DIR}/train.csv
  ${DATA_DIR}/test.csv
  ${DATA_DIR}/sample_submission.csv

Command:
  bash scripts/create_submission.sh ${DATA_DIR} ${OUTPUT} ${CV_REPORT}
EOF
  exit 1
fi

python src/traffic_demand_solution.py \
  --data-dir "${DATA_DIR}" \
  --output "${OUTPUT}" \
  --cv-report "${CV_REPORT}"

echo "Created ${OUTPUT} for leaderboard submission."
echo "Created ${CV_REPORT} for validation diagnostics."
