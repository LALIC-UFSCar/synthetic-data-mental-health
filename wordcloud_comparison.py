# wordcloud_comparison.py
# Geração de wordclouds comparando o corpus original e o sintético.

from pathlib import Path

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords', quiet=True)
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud

# ── caminhos — edite conforme necessário ─────────────────────────────────────
ORIGINAL    = Path('data/amive_sentences.csv')
SINTETICO   = Path('data/amive_sintetico.csv')
OUTPUT      = Path('results/wordcloud_comparison.png')
TEXT_COLUMN = 'TEXT'
# ─────────────────────────────────────────────────────────────────────────────

# stopwords em português (a lib wordcloud só tem inglês por padrão)
stopwords_pt = set(STOPWORDS) | set(stopwords.words('portuguese'))


def load_text(path, column):
    df = pd.read_csv(path)
    df = df.dropna(subset=[column])
    return ' '.join(df[column].astype(str).tolist())


text_orig = load_text(ORIGINAL, TEXT_COLUMN)
text_sint = load_text(SINTETICO, TEXT_COLUMN)

wc_orig = WordCloud(
    width=800, height=600,
    background_color='white',
    stopwords=stopwords_pt,
    collocations=False
).generate(text_orig)

wc_sint = WordCloud(
    width=800, height=600,
    background_color='white',
    stopwords=stopwords_pt,
    collocations=False
).generate(text_sint)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(wc_orig, interpolation='bilinear')
axes[0].set_title('Original', fontsize=14)
axes[0].axis('off')

axes[1].imshow(wc_sint, interpolation='bilinear')
axes[1].set_title('Synthetic', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.close()

print(f"Wordcloud salva em: {OUTPUT}")
