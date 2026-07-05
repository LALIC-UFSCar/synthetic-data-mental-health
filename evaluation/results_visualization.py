# results_visualization.py
# Geração de gráficos de avaliação dos classificadores: barras por métrica,
# barras por modelo, heatmaps de diferença e distribuição de POS tags.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── caminhos — edite conforme necessário ─────────────────────────────────────
RESULTS             = Path('results/results.csv')
POS_SYMPTOM         = Path('results/pos_distribution_symptom.csv')
POS_CONTROL         = Path('results/pos_distribution_control.csv')
OUTPUT_DIR          = Path('results/plots')
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

df      = pd.read_csv(RESULTS)
metrics = ['accuracy', 'precision', 'recall', 'f1']
symptom = df['symptom'].iloc[0]
models  = df['model'].unique()

# ordem fixa dos cenários para consistência nos gráficos
corpora = ['original', 'sintetico',
           'sintetico_treino_original_teste', 'original_treino_sintetico_teste']
corpora = [c for c in corpora if c in df['corpus'].unique()]

colors = {
    'original':                       '#2196F3',
    'sintetico':                       '#FF9800',
    'sintetico_treino_original_teste': '#4CAF50',
    'original_treino_sintetico_teste': '#9C27B0',
}
labels = {
    'original':                       'Original → Original',
    'sintetico':                       'Synthetic → Synthetic',
    'sintetico_treino_original_teste': 'Synthetic → Original',
    'original_treino_sintetico_teste': 'Original → Synthetic',
}

# DataFrames por cenário — usados nos heatmaps
df_orig      = df[df['corpus'] == 'original'].set_index('model')[metrics]
df_sint      = df[df['corpus'] == 'sintetico'].set_index('model')[metrics]
df_sint_orig = df[df['corpus'] == 'sintetico_treino_original_teste'].set_index('model')[metrics]
df_orig_sint = df[df['corpus'] == 'original_treino_sintetico_teste'].set_index('model')[metrics]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TABELA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

df_table = pd.concat([
    df_orig.rename(columns={m: f'{m}_orig'      for m in metrics}),
    df_sint.rename(columns={m: f'{m}_sint'      for m in metrics}),
    df_sint_orig.rename(columns={m: f'{m}_sint_orig' for m in metrics}),
    df_orig_sint.rename(columns={m: f'{m}_orig_sint' for m in metrics}),
], axis=1)

cols = [f'{m}_{s}' for m in metrics for s in ['orig', 'sint', 'sint_orig', 'orig_sint']]
df_table = df_table[cols]

print(f"\nSymptom: {symptom}")
print("=" * 80)
print(df_table.to_string())
df_table.to_csv(OUTPUT_DIR / 'comparative_table.csv')
print(f"\nTable saved: {OUTPUT_DIR / 'comparative_table.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BARRAS AGRUPADAS POR MÉTRICA
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'Scenario Comparison\nSymptom: {symptom}', fontsize=14)

x     = np.arange(len(models))
width = 0.18

for ax, metric in zip(axes.flatten(), metrics):
    for i, corpus in enumerate(corpora):
        values = df[df['corpus'] == corpus].set_index('model').loc[models, metric].values
        offset = (i - (len(corpora) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=labels[corpus], color=colors[corpus], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_title(metric.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('TF-IDF + ', '') for m in models], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bars_by_metric.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {OUTPUT_DIR / 'bars_by_metric.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BARRAS AGRUPADAS POR MODELO
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'Metrics by Model\nSymptom: {symptom}', fontsize=14)

x = np.arange(len(metrics))

for ax, model in zip(axes.flatten(), models):
    for i, corpus in enumerate(corpora):
        values = df[(df['corpus'] == corpus) & (df['model'] == model)][metrics].values[0]
        offset = (i - (len(corpora) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=labels[corpus], color=colors[corpus], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_title(model.replace('TF-IDF + ', ''), fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bars_by_model.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {OUTPUT_DIR / 'bars_by_model.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP DE DIFERENÇAS EM RELAÇÃO AO ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f'Differences vs Original Baseline\nSymptom: {symptom}', fontsize=13)

diffs = [
    ('Synthetic − Original',     df_sint - df_orig),
    ('Synt→Orig − Original',     df_sint_orig - df_orig),
    ('Orig→Synt − Original',     df_orig_sint - df_orig),
]

for ax, (title, df_diff) in zip(axes.flatten(), diffs):
    im = ax.imshow(df_diff.values, cmap='RdYlGn', vmin=-0.2, vmax=0.2, aspect='auto')
    plt.colorbar(im, ax=ax, label='Difference')
    ax.set_title(title)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace('TF-IDF + ', '') for m in models])
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{df_diff.values[i, j]:+.3f}',
                    ha='center', va='center', fontsize=10)

# remove o quarto subplot (vazio, pois há apenas 3 heatmaps)
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_differences.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved: {OUTPUT_DIR / 'heatmap_differences.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DISTRIBUIÇÃO DE POS TAGS (sintoma e controle)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pos_distribution(csv_path, output_filename, title_suffix):
    """Gera gráfico de barras comparando distribuição de POS entre original e sintético."""
    df_pos = pd.read_csv(csv_path)

    # compatibilidade: aceita tanto 'sintetico' (pt) quanto 'synthetic' (en)
    col_sint = 'synthetic' if 'synthetic' in df_pos.columns else 'sintetico'

    x     = np.arange(len(df_pos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, df_pos['original'], width,
                   label='Original',   color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width / 2, df_pos[col_sint],  width,
                   label='Synthetic',  color='#FF9800', alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('POS Tag')
    ax.set_ylabel('Proportion')
    ax.set_title(f'POS Tag Distribution: Original vs Synthetic\n{title_suffix}')
    ax.set_xticks(x)
    ax.set_xticklabels(df_pos['pos_tag'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {OUTPUT_DIR / output_filename}")


plot_pos_distribution(POS_SYMPTOM,  'pos_distribution_symptom.png',  '(amive_sentences)')
plot_pos_distribution(POS_CONTROL,  'pos_distribution_control.png',  '(amive_controle)')

print(f"\nAll plots saved in: {OUTPUT_DIR}")
