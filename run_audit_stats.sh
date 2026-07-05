#!/bin/bash
# run_audit_stats.sh
# Executa a auditoria do pipeline de geração.

DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../evaluation/audit_stats.py" \
  --stats            "$DIR/amive_sintetico_stats.jsonl" \
  --dataset_original "$DIR/amive_sentences.csv" \
  --csv_final        "$DIR/amive_sintetico.csv" \
  --output           "$DIR/results/audit_report.txt" \
  --output_csv       "$DIR/results/audit_detailed.csv"
