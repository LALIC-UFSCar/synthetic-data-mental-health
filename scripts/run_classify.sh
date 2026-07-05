#!/bin/bash
# run_classify.sh
# Executa a classificação binária comparando corpus original e sintético.

DIR="$(cd "$(dirname "$0")" && pwd)"

python "$DIR/../evaluation/classify.py" \
  --dataset                    "$DIR/amive_sentences.csv" \
  --dataset_sintetico          "$DIR/amive_sintetico.csv" \
  --dataset_controle           "$DIR/amive_controle.csv" \
  --dataset_controle_sintetico "$DIR/amive_controle_sintetico.csv" \
  --symptom                    "Tristeza/Humor depressivo" \
  --output                     "$DIR/results/results.csv" \
  --output_jaccard             "$DIR/results/jaccard.csv" \
  --test_size                  0.3 \
  --seed                       42
