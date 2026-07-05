#!/bin/bash
# run.sh
# Executa o pipeline completo de geração de paráfrases sintéticas.

DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../pipeline/main.py" \
  --dataset       "$DIR/exemplo.csv" \
  --doc_type      csv \
  --config        "$DIR/config.yaml" \
  --model         llama-3.3-70b-versatile \
  --client        groq \
  --output        "$DIR/exemplo_sintetico.csv" \
  --output_stats  "$DIR/exemplo_sintetico_stats.jsonl" \
  --num_sequences 5 \
  --batch_size    5 \
  --sleep         1
