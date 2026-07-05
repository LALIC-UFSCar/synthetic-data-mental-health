# Auditoria do pipeline: gera relatório de estatísticas a partir do .jsonl
# produzido pelo main.py. Usa o CSV final como referência definitiva de
# quais instâncias sobreviveram, recalculando métricas quando necessário.

import argparse
import re
import statistics
from pathlib import Path

import jsonlines
import pandas as pd
from transformers import PegasusTokenizer
from bert_score import score as bert_score_fn

BERT_MODEL = 'neuralmind/bert-base-portuguese-cased'
peg_tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',  '--stats',           type=Path, required=True,
                        help='.jsonl gerado pelo main.py.')
    parser.add_argument('-d',  '--dataset_original', type=Path, required=True,
                        help='CSV original — fonte de verdade dos textos originais.')
    parser.add_argument('-c',  '--csv_final',        type=Path, required=True,
                        help='CSV final — referência definitiva de quais instâncias sobreviveram.')
    parser.add_argument('-o',  '--output',           type=Path, required=True,
                        help='Arquivo .txt com o relatório.')
    parser.add_argument('-oc', '--output_csv',       type=Path, required=False,
                        help='CSV com estatísticas detalhadas por instância (opcional).')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_for_diff(text):
    """Separa pontuação colada às palavras para comparação justa de tokens."""
    text = re.sub(r'([.,!?;:])', r' \1 ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def count_tokens(text):
    if not text:
        return 0
    return len(peg_tokenizer(str(text))['input_ids'])


def token_diff_jaccard(original, paraphrase):
    """Diferença de vocabulário via distância de Jaccard (sempre entre 0% e 100%)."""
    tokens1 = set(normalize_for_diff(original).split())
    tokens2 = set(normalize_for_diff(paraphrase).split())
    if not tokens1 and not tokens2:
        return 0, 0.0
    diff  = tokens1.symmetric_difference(tokens2)
    union = tokens1.union(tokens2)
    ratio = round(len(diff) / len(union) * 100, 2) if union else 0.0
    return len(diff), ratio


def recompute_score(original, paraphrase):
    """Recalcula BERTScore para instâncias cujo texto foi substituído (score_replaced=True)."""
    if not paraphrase or not paraphrase.strip():
        return 0.0
    P, R, F1 = bert_score_fn(
        [paraphrase], [original],
        model_type=BERT_MODEL, num_layers=12,
        lang='pt', verbose=False, idf=False
    )
    return F1[0].item() * 100


def safe_mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def safe_median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def safe_stdev(values):
    values = [v for v in values if v is not None]
    return round(statistics.stdev(values), 4) if len(values) > 1 else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # carrega o .jsonl indexado por id
    print("Carregando stats...")
    stats_map = {}
    with jsonlines.open(args.stats) as reader:
        for obj in reader:
            stats_map[obj['id']] = obj

    # texto original por id
    df_orig = pd.read_csv(args.dataset_original)
    df_orig = df_orig.rename(columns={'DOCNO': 'id', 'TEXT': 'text'})
    original_text_map = df_orig.drop_duplicates(subset='id').set_index('id')['text'].to_dict()

    # CSV final: referência definitiva de quem sobreviveu e com qual texto
    df_final = pd.read_csv(args.csv_final)
    df_final = df_final.rename(columns={'DOCNO': 'id', 'TEXT': 'text'})
    final_ids      = set(df_final['id'].unique())
    final_text_map = df_final.drop_duplicates(subset='id').set_index('id')['text'].to_dict()

    print(f"IDs no CSV final: {len(final_ids)}")

    # ── monta registros por instância ─────────────────────────────────────────

    records = []

    for doc_id, original in original_text_map.items():
        obj = stats_map.get(doc_id)

        if doc_id not in final_ids:
            # instância excluída — determina o motivo a partir dos campos do .jsonl
            if obj is None:
                exclusion_stage = 'not_processed'
            elif obj.get('no_valid_paraphrase'):
                paras_en = obj.get('paraphrases_en', [])
                split_log = obj.get('split_log', [])
                n_discarded = sum(1 for s in split_log if s.get('action') == 'discarded_still_over_limit')
                if not paras_en or all(not p.strip() for p in paras_en):
                    exclusion_stage = 'pegasus_returned_all_empty'
                elif n_discarded > 0 and not any(
                    s.get('action') in ('kept_after_period_split', 'kept_after_comma_split')
                    for s in split_log
                ):
                    exclusion_stage = 'all_fragments_exceeded_token_limit'
                else:
                    exclusion_stage = 'all_paraphrases_identical_to_original'
            else:
                exclusion_stage = 'removed_extreme_score_no_alternative'

            records.append({
                'id': doc_id,
                'status': 'excluded',
                'exclusion_stage': exclusion_stage,
                'final_score': None,
                'score_replaced': obj.get('score_replaced', False) if obj else False,
                'diff_pct_jaccard': None,
                'n_tokens_original': count_tokens(original),
                'n_tokens_final': None,
                'used_split': bool(obj.get('split_log')) if obj else False,
                'n_fragments_used': sum(
                    1 for s in obj.get('split_log', [])
                    if s.get('action') in ('kept_after_period_split', 'kept_after_comma_split')
                ) if obj else None,
                'n_fragments_discarded': sum(
                    1 for s in obj.get('split_log', [])
                    if s.get('action') == 'discarded_still_over_limit'
                ) if obj else None,
            })
            continue

        # instância sobreviveu
        final_text   = final_text_map.get(doc_id, '')
        score_replaced = obj.get('score_replaced', False) if obj else False

        # recalcula o score apenas se o texto foi substituído (score_replaced=True),
        # pois nesse caso o best_score do .jsonl corresponde ao score antigo (0 ou 100)
        if score_replaced:
            final_score = recompute_score(original, final_text)
        else:
            final_score = obj.get('best_score') if obj else None

        _, diff_jaccard = token_diff_jaccard(original, final_text)
        split_log = obj.get('split_log', []) if obj else []

        records.append({
            'id': doc_id,
            'status': 'kept',
            'exclusion_stage': None,
            'final_score': final_score,
            'score_replaced': score_replaced,
            'diff_pct_jaccard': diff_jaccard,
            'n_tokens_original': count_tokens(original),
            'n_tokens_final': count_tokens(final_text),
            'used_split': bool(split_log),
            'n_fragments_used': sum(
                1 for s in split_log
                if s.get('action') in ('kept_after_period_split', 'kept_after_comma_split')
            ),
            'n_fragments_discarded': sum(
                1 for s in split_log
                if s.get('action') == 'discarded_still_over_limit'
            ),
        })

    df = pd.DataFrame(records)

    # ── estatísticas agregadas ─────────────────────────────────────────────────

    total    = len(df)
    n_kept   = len(df[df['status'] == 'kept'])
    n_excl   = len(df[df['status'] == 'excluded'])
    n_replaced = df['score_replaced'].sum()
    n_split  = len(df[df['used_split'] == True])

    exclusion_counts = df[df['status'] == 'excluded']['exclusion_stage'].value_counts()

    scores_kept       = df[df['status'] == 'kept']['final_score'].tolist()
    diff_jaccard_kept = df[df['status'] == 'kept']['diff_pct_jaccard'].tolist()
    tokens_orig       = df['n_tokens_original'].tolist()
    tokens_final      = df[df['status'] == 'kept']['n_tokens_final'].tolist()

    df_split = df[df['used_split'] == True]
    n_frags_used      = df_split['n_fragments_used'].dropna().sum()
    n_frags_discarded = df_split['n_fragments_discarded'].dropna().sum()
    avg_frags         = safe_mean(df_split['n_fragments_used'].tolist())

    # ── monta relatório ─────────────────────────────────────────────────────────

    lines = []
    lines.append("=" * 70)
    lines.append("RELATÓRIO DE AUDITORIA DO PIPELINE DE PARÁFRASE")
    lines.append("=" * 70)

    lines.append(f"\nTotal de instâncias no dataset original: {total}")
    lines.append(f"Mantidas no CSV final: {n_kept} ({n_kept/total*100:.1f}%)")
    lines.append(f"Excluídas: {n_excl} ({n_excl/total*100:.1f}%)")
    lines.append(f"Instâncias com score corrigido (era 0/100): {int(n_replaced)}")
    lines.append(f"Instâncias que passaram pelo split: {n_split}")

    lines.append("\n--- MOTIVOS DE EXCLUSÃO ---")
    if len(exclusion_counts) > 0:
        for stage, count in exclusion_counts.items():
            lines.append(f"  {stage}: {count}")
    else:
        lines.append("  Nenhuma exclusão registrada.")

    lines.append("\n--- BERTSCORE (instâncias mantidas) ---")
    lines.append(f"Médio: {safe_mean(scores_kept)} | "
                 f"Mediano: {safe_median(scores_kept)} | "
                 f"DP: {safe_stdev(scores_kept)}")

    lines.append("\n--- DIFERENÇA DE VOCABULÁRIO — Jaccard (instâncias mantidas) ---")
    lines.append(f"Médio: {safe_mean(diff_jaccard_kept)} | "
                 f"Mediano: {safe_median(diff_jaccard_kept)}")

    lines.append("\n--- TAMANHO EM TOKENS (tokenizer Pegasus) ---")
    lines.append(f"Original — média: {safe_mean(tokens_orig)}, DP: {safe_stdev(tokens_orig)}")
    lines.append(f"Sintético — média: {safe_mean(tokens_final)}, DP: {safe_stdev(tokens_final)}")

    if n_split > 0:
        lines.append("\n--- SPLIT DE TEXTOS LONGOS ---")
        lines.append(f"Instâncias que usaram split: {n_split}")
        lines.append(f"Fragmentos parafraseados com sucesso: {int(n_frags_used)}")
        lines.append(f"Fragmentos descartados (excediam limite mesmo após split): {int(n_frags_discarded)}")
        lines.append(f"Média de fragmentos por instância: {avg_frags}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nRelatório salvo em: {args.output}")

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"CSV detalhado salvo em: {args.output_csv}")


if __name__ == '__main__':
    main()
