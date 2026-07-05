# AMIVE Synthetic Corpus

A back-translation paraphrase pipeline for the AMIVE corpus, enabling its public distribution without exposing the original sensitive texts.

The AMIVE corpus (*Análise de Manifestações de Inteligência e Vulnerabilidades Emocionais* — Analysis of Manifestations of Intelligence and Emotional Vulnerabilities) contains posts from university students on anonymous social media pages, annotated with mental health symptoms. Because it contains sensitive and potentially identifiable data, the original corpus cannot be publicly distributed. This repository implements a synthetic data generation pipeline that preserves the linguistic and semantic characteristics of the corpus, making paraphrased versions available for research.

---

## 🎯 Objective

Generate and evaluate a synthetic version of the AMIVE corpus through automatic paraphrasing, aiming to:

- **Preserve the privacy** of the original post authors
- **Maintain linguistic and semantic quality** of the data
- **Enable reproducibility** of research using the AMIVE corpus
- **Evaluate synthetic corpus quality** via mental health symptom classification

---

## 🔄 Pipeline overview

```
Original text (PT)
        ↓
  Translation PT → EN     (LLM via Groq / OpenAI / MariTalk)
        ↓
  Automatic split         (texts > 60 tokens: split by period and comma)
        ↓
  Paraphrase in EN        (Pegasus: tuner007/pegasus_paraphrase)
        ↓
  Translation EN → PT     (LLM via Groq / OpenAI / MariTalk)
        ↓
  Filters + corrections   (deduplication, punctuation, extreme scores 0/100)
        ↓
  Best selection          (BERTScore: neuralmind/bert-base-portuguese-cased)
        ↓
  Final synthetic CSV
```

All steps — including handling of long texts and corrections — happen inside the same main loop, with no separate auxiliary scripts.

---

## 📂 Repository structure

```
amive-synthetic/
│
├── README.md
├── README_en.md
├── requirements.txt
├── .env.example
│
├── pipeline/
│   └── main.py                   ← full generation pipeline
│
├── evaluation/
│   ├── classify.py               ← binary classification (TF-IDF + multiple classifiers)
│   ├── intrinsic_evaluation.py   ← vocabulary, D-2, inverse BLEU, POS tags
│   ├── audit_stats.py            ← pipeline audit and generation statistics
│   ├── results_visualization.py  ← classification result plots
│   └── wordcloud_comparison.py   ← wordclouds: original vs synthetic
│
├── scripts/
│   ├── run.sh                    ← runs the main pipeline
│   ├── run_classify.sh           ← runs classification
│   └── run_audit_stats.sh        ← runs the audit
│
└── utils/
    ├── io.py
    └── llm_request.py
```

---

## 🛠️ Main technologies and dependencies

- **Python 3.10**
- **[Pegasus](https://huggingface.co/tuner007/pegasus_paraphrase)** — paraphrase generation in English
- **[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)** — best paraphrase selection via BERTScore
- **[spaCy](https://spacy.io/)** (`pt_core_news_sm`) — grammatical category analysis (POS tagging)
- **[Groq](https://console.groq.com)** / OpenAI / MariTalk — translation via LLM
- **scikit-learn** — TF-IDF classifiers for extrinsic evaluation

> **Important:** the `transformers` and `sentencepiece` versions are critical for the Pegasus model to work correctly. Use exactly the versions specified in `requirements.txt`.

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/amive-synthetic.git
cd amive-synthetic
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model

```bash
python -m spacy download pt_core_news_sm
```

### 5. Set up API keys

```bash
cp .env.example .env
```

Fill in `.env` with your keys:

```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here      # optional
MARITALK_API_KEY=your_key_here    # optional
```

Groq is recommended as it offers a free tier. Create an account at [console.groq.com](https://console.groq.com).

---

## ▶️ Running each step

### Step 1 — Synthetic corpus generation

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

Or directly:

```bash
python pipeline/main.py \
  --dataset       data/amive_sentences.csv \
  --doc_type      csv \
  --config        config/config.yaml \
  --model         llama-3.3-70b-versatile \
  --client        groq \
  --output        output/amive_synthetic.csv \
  --output_stats  output/amive_synthetic_stats.jsonl \
  --num_sequences 5 \
  --batch_size    5 \
  --sleep         1
```

Progress is saved after each instance in `--output_stats`, allowing the run to be resumed if interrupted without reprocessing already completed instances.

**Main arguments:**

| Argument | Description |
|---|---|
| `--dataset` | Input file (CSV, JSONL or TXT directory) |
| `--doc_type` | Format: `csv`, `jsonl` or `txt` (default: `txt`) |
| `--config` | YAML file with LLM settings (`temperature`, `top_p`, `max_completion_tokens`) |
| `--model` | LLM model name (e.g. `llama-3.3-70b-versatile`) |
| `--client` | API client: `groq`, `openai` or `maritalk` |
| `--output` | Output CSV with final paraphrases |
| `--output_stats` | JSONL with detailed per-instance statistics |
| `--num_sequences` | Number of paraphrases generated per text (default: `5`) |
| `--batch_size` | Instances per translation batch (default: `5`) |
| `--sleep` | Seconds between LLM requests (default: `1`) |

**CSV input format:**
```
DOCNO,TEXT,SYMPTOM
100_1,"Estou me sentindo muito mal.",Tristeza/Humor depressivo
```

**YAML config format:**
```yaml
temperature: 0.3
top_p: 0.95
max_completion_tokens: 1024
```

---

### Step 2 — Classification

Trains and evaluates binary classifiers (symptom vs. control) with TF-IDF across four scenarios:

| Scenario | Train | Test |
|---|---|---|
| 1 | Original | Original |
| 2 | Synthetic | Synthetic |
| 3 | Synthetic | Original |
| 4 | Original | Synthetic |

In addition to standard metrics (accuracy, precision, recall, F1), computes **Jaccard similarity between pairs of classifiers** — indicating whether models agree on correct and incorrect predictions.

```bash
chmod +x scripts/run_classify.sh
./scripts/run_classify.sh
```

Or directly:

```bash
python evaluation/classify.py \
  --dataset                    data/amive_sentences.csv \
  --dataset_sintetico          data/amive_synthetic.csv \
  --dataset_controle           data/amive_controle.csv \
  --dataset_controle_sintetico data/amive_controle_synthetic.csv \
  --symptom                    "Tristeza/Humor depressivo" \
  --output                     results/results.csv \
  --output_jaccard             results/jaccard.csv \
  --test_size                  0.3 \
  --seed                       42
```

---

### Step 3 — Intrinsic evaluation

Computes quality metrics for the synthetic corpus compared to the original:

| Metric | Description |
|---|---|
| **Vocabulary size** | Number of unique tokens |
| **D-2** | Proportion of unique bigrams — higher means more diverse |
| **Inverse BLEU** | Structural similarity to the original — lower means more diverse |
| **POS distribution** | Proportion of each grammatical category via spaCy |

```bash
python evaluation/intrinsic_evaluation.py
```

> Edit the `ORIGINAL`, `SINTETICO`, `OUTPUT` and `OUTPUT_POS` variables at the top of the file before running.

---

### Step 4 — Pipeline audit

Generates a detailed report on the generation process, including total instances kept and excluded, exclusion reasons, BERTScore statistics, vocabulary difference (Jaccard) and token size comparison.

```bash
chmod +x scripts/run_audit_stats.sh
./scripts/run_audit_stats.sh
```

Or directly:

```bash
python evaluation/audit_stats.py \
  --stats            output/amive_synthetic_stats.jsonl \
  --dataset_original data/amive_sentences.csv \
  --csv_final        output/amive_synthetic.csv \
  --output           results/audit_report.txt \
  --output_csv       results/audit_detailed.csv
```

---

### Step 5 — Visualizations

```bash
# classification result plots
python evaluation/results_visualization.py

# wordclouds: original vs synthetic
python evaluation/wordcloud_comparison.py
```

> Edit the path variables at the top of each file before running.

Files generated in `results/plots/`: comparative table, bars by metric, bars by model, difference heatmaps and POS distribution plots.

---

## ⚠️ API rate limit notes

Groq's free tier has a limit of **1,000 requests per day**. The automatic checkpoint ensures progress is not lost if execution is interrupted — simply run the same command again to resume from where it stopped.

Recommended models:

| Model | Client | Characteristic |
|---|---|---|
| `llama-3.3-70b-versatile` | Groq | Best translation quality |
| `llama-3.1-8b-instant` | Groq | Faster, lower quota consumption |
| `sabia-3` | MariTalk | Good alternative for Brazilian Portuguese |

---

## 👩‍💻 Authors and institution

Developed as Bachelors Thesis at Federal University of São Carlos (B.Sc. Computer Engineering).

**Title (pt):** Geração e Avaliação de Dados Sintéticos para Aplicações em Saúde Mental

**Author:** Vitória Rodrigues Pinto Borelli Figueiredo

**Professor:** Dr. Helena de Medeiros Caseli
