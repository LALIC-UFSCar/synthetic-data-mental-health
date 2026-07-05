# intrinsic_evaluation.py
# Avaliação intrínseca do corpus sintético comparado ao original.
# Métricas: tamanho do vocabulário, D-2 (diversidade de bigramas),
# BLEU invertido e distribuição de categorias gramaticais (POS) via spaCy.

from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
import nltk
nltk.download('stopwords', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import bigrams

# ── caminhos — edite conforme necessário ─────────────────────────────────────
ORIGINAL    = Path('data/amive_sentences.csv')
SINTETICO   = Path('data/amive_sintetico.csv')
OUTPUT      = Path('results/intrinsic_metrics.csv')
OUTPUT_POS  = Path('results/pos_distribution.csv')
# ─────────────────────────────────────────────────────────────────────────────

nlp = spacy.load('pt_core_news_sm')


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text):
    return str(text).lower().split()


def vocabulary_size(texts):
    """Tamanho do vocabulário único do corpus."""
    vocab = set()
    for text in texts:
        vocab.update(tokenize(text))
    return len(vocab)


def distinct_2(texts):
    """D-2: proporção de bigramas únicos sobre o total de bigramas."""
    all_bigrams = []
    for text in texts:
        all_bigrams.extend(list(bigrams(tokenize(text))))
    if not all_bigrams:
        return 0.0
    return round(len(set(all_bigrams)) / len(all_bigrams), 4)


def avg_bleu(originals, synthetics):
    """
    BLEU invertido: cada texto sintético avaliado contra seu original.
    Valores mais baixos indicam que as paráfrases se distanciaram
    estruturalmente dos originais — o que é desejado.
    """
    smoother = SmoothingFunction().method1
    scores = [
        sentence_bleu([tokenize(orig)], tokenize(synt), smoothing_function=smoother)
        for orig, synt in zip(originals, synthetics)
    ]
    return round(sum(scores) / len(scores), 4)


def pos_distribution(texts):
    """
    Proporção de cada categoria gramatical (POS) sobre o total de tokens.
    Exclui pontuação e espaços do cálculo.
    """
    counts = Counter()
    total  = 0
    for text in texts:
        doc = nlp(str(text))
        for token in doc:
            if not token.is_punct and not token.is_space:
                counts[token.pos_] += 1
                total += 1
    if total == 0:
        return {}
    return {pos: round(count / total, 4) for pos, count in counts.items()}


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════════════════

df_orig = pd.read_csv(ORIGINAL)
df_orig = df_orig.rename(columns={'DOCNO': 'id', 'TEXT': 'text', 'SYMPTOM': 'symptom'})
df_orig = df_orig.drop_duplicates(subset='id')

df_sint = pd.read_csv(SINTETICO)
df_sint = df_sint.rename(columns={'DOCNO': 'id', 'TEXT': 'text', 'SYMPTOM': 'symptom'})
df_sint = df_sint.drop_duplicates(subset='id')
df_sint = df_sint.dropna(subset=['text'])
df_sint = df_sint[df_sint['text'].str.strip() != '']

# alinha pelos IDs comuns
ids_comuns = set(df_orig['id']) & set(df_sint['id'])
df_orig = df_orig[df_orig['id'].isin(ids_comuns)].sort_values('id').reset_index(drop=True)
df_sint = df_sint[df_sint['id'].isin(ids_comuns)].sort_values('id').reset_index(drop=True)

print(f"Instâncias comparadas: {len(df_orig)}")
print("=" * 50)

orig_texts = df_orig['text'].tolist()
sint_texts = df_sint['text'].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO E SAÍDA
# ══════════════════════════════════════════════════════════════════════════════

# métricas principais
df_metrics = pd.DataFrame({
    'metric':    ['Vocabulary size', 'D-2 (unique bigrams)', 'BLEU (inverted)'],
    'original':  [vocabulary_size(orig_texts), distinct_2(orig_texts), '—'],
    'synthetic': [vocabulary_size(sint_texts), distinct_2(sint_texts),
                  avg_bleu(orig_texts, sint_texts)],
})
print(df_metrics.to_string(index=False))
df_metrics.to_csv(OUTPUT, index=False)
print(f"\nMétricas salvas em: {OUTPUT}")

# distribuição de POS tags
print("\n" + "=" * 50)
print("Calculando distribuição de categorias gramaticais (spaCy)...")

pos_orig = pos_distribution(orig_texts)
pos_sint = pos_distribution(sint_texts)
all_tags = sorted(set(pos_orig) | set(pos_sint))

df_pos = pd.DataFrame({
    'pos_tag':   all_tags,
    'original':  [pos_orig.get(tag, 0.0) for tag in all_tags],
    'synthetic': [pos_sint.get(tag, 0.0) for tag in all_tags],
})
df_pos['difference'] = (df_pos['synthetic'] - df_pos['original']).round(4)
df_pos = df_pos.sort_values('original', ascending=False).reset_index(drop=True)

print(df_pos.to_string(index=False))
df_pos.to_csv(OUTPUT_POS, index=False)
print(f"\nDistribuição de POS salva em: {OUTPUT_POS}")
