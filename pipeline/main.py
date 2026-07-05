# Pipeline completo de geração de paráfrases sintéticas para o corpus AMIVE.
# Fluxo: PT → EN (LLM) → paráfrase (Pegasus) → EN → PT (LLM)

import argparse
import os
import re
import time
from itertools import islice
from pathlib import Path

import jsonlines
import pandas as pd
import torch
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from bert_score import score as bert_score_fn

from utils.io import parse_yaml
from utils.llm_request import get_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_TOKENS = 60
BERT_MODEL = 'neuralmind/bert-base-portuguese-cased'


# ══════════════════════════════════════════════════════════════════════════════
# PEGASUS
# ══════════════════════════════════════════════════════════════════════════════

def load_pegasus():
    model_name = 'tuner007/pegasus_paraphrase'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model, device


def count_tokens(text, tokenizer):
    return len(tokenizer(str(text))['input_ids'])


def get_paraphrases_short(text, tokenizer, model, device, num_sequences):
    """Gera paráfrases para textos curtos (≤ MAX_TOKENS tokens)."""
    batch = tokenizer(
        [text], truncation=True, padding='longest',
        max_length=MAX_TOKENS, return_tensors="pt"
    ).to(device)
    outputs = model.generate(
        **batch, max_length=MAX_TOKENS,
        num_beams=num_sequences, num_return_sequences=num_sequences,
        num_beam_groups=num_sequences, diversity_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT DE TEXTOS LONGOS
# ══════════════════════════════════════════════════════════════════════════════

def split_preserving_punctuation(text, pattern, sep_type):
    """Divide o texto preservando os separadores para reconstrução posterior."""
    parts = re.split(f'({pattern})', text.strip())
    result = []
    i = 0
    while i < len(parts):
        fragment = parts[i].strip()
        separator = parts[i+1] if i+1 < len(parts) else ''
        if fragment:
            result.append({'text': fragment, 'separator': separator, 'split_type': sep_type})
        i += 2
    return result


def split_by_period(text):
    return split_preserving_punctuation(text, r'[.!?]\s*', 'period')


def split_by_comma(text):
    return split_preserving_punctuation(text, r',\s*', 'comma')


def strip_trailing_punctuation(text):
    """Remove pontuação final adicionada pelo Pegasus antes de juntar fragmentos."""
    return re.sub(r'[.!?]+\s*$', '', text.strip())


def reconstruct(fragments):
    """Junta fragmentos parafraseados preservando a pontuação original."""
    result = ''
    for i, item in enumerate(fragments):
        paraphrase = item['paraphrase']
        if i < len(fragments) - 1:
            paraphrase = strip_trailing_punctuation(paraphrase)
        result += paraphrase
        if i < len(fragments) - 1:
            result += item['separator'] if item['separator'] else ' '
    return result.strip()


def get_paraphrases_long(text_en, tokenizer, model, device, num_sequences):
    """
    Gera paráfrases para textos longos (> MAX_TOKENS tokens).
    Divide por ponto e, se necessário, por vírgula antes do Pegasus.
    Junta os fragmentos parafraseados preservando a pontuação original.
    Retorna (paráfrases_finais, split_log).
    """
    fragments = split_by_period(text_en)
    split_log = []
    fragments_final = []

    for frag in fragments:
        n = count_tokens(frag['text'], tokenizer)
        if n <= MAX_TOKENS:
            frag['action'] = 'kept_after_period_split'
            frag['n_tokens'] = n
            fragments_final.append(frag)
            split_log.append({'fragment': frag['text'], 'n_tokens': n,
                               'action': 'kept_after_period_split'})
        else:
            # fallback: divide por vírgula
            sub_frags = split_by_comma(frag['text'])
            for sf in sub_frags:
                sn = count_tokens(sf['text'], tokenizer)
                if sn <= MAX_TOKENS:
                    sf['action'] = 'kept_after_comma_split'
                    sf['n_tokens'] = sn
                    fragments_final.append(sf)
                    split_log.append({'fragment': sf['text'], 'n_tokens': sn,
                                      'action': 'kept_after_comma_split'})
                else:
                    # fragmento ainda excede o limite após divisão por vírgula: descarta
                    split_log.append({'fragment': sf['text'], 'n_tokens': sn,
                                      'action': 'discarded_still_over_limit'})

    if not fragments_final:
        return [], split_log

    # Pegasus em cada fragmento individualmente
    paraphrased_fragments = []
    for frag in fragments_final:
        paras_en = get_paraphrases_short(frag['text'], tokenizer, model, device, num_sequences)
        paraphrased_fragments.append({
            'paraphrase': paras_en[0],
            'separator': frag['separator'],
        })

    joined = reconstruct(paraphrased_fragments)
    return [joined], split_log


def get_paraphrases(text_en, tokenizer, model, device, num_sequences):
    """
    Decide automaticamente entre paráfrase direta (texto curto)
    ou com split (texto longo), baseado no número de tokens.
    Retorna (paráfrases, split_log).
    """
    n = count_tokens(text_en, tokenizer)
    if n <= MAX_TOKENS:
        paras = get_paraphrases_short(text_en, tokenizer, model, device, num_sequences)
        return paras, []
    else:
        return get_paraphrases_long(text_en, tokenizer, model, device, num_sequences)


# ══════════════════════════════════════════════════════════════════════════════
# BERTSCORE
# ══════════════════════════════════════════════════════════════════════════════

def paraphrase_scores_batch(original, paraphrases):
    """Calcula BERTScore de todas as paráfrases de uma vez (batch)."""
    P, R, F1 = bert_score_fn(
        paraphrases, [original] * len(paraphrases),
        model_type=BERT_MODEL, num_layers=12,
        lang='pt', verbose=False, idf=False
    )
    return [f.item() * 100 for f in F1]


def select_best_paraphrase(original, paraphrases):
    """
    Seleciona a melhor paráfrase pelo BERTScore.
    Retorna (melhor_paráfrase, melhor_score, todos_os_scores).
    """
    scores = paraphrase_scores_batch(original, paraphrases)
    best_idx = scores.index(max(scores))
    return paraphrases[best_idx], scores[best_idx], scores


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_client(client_name):
    load_dotenv()
    if client_name == 'groq':
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    elif client_name == 'openai':
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    else:
        return OpenAI(api_key=os.getenv('MARITALK_API_KEY'),
                      base_url="https://chat.maritaca.ai/api")


def translate_batch(client, texts, direction, config, model_name):
    """Traduz uma lista de textos em uma única chamada à API (batch numerado)."""
    if direction == 'pt_en':
        system = (
            "You are a translator. Translate each numbered item to English. "
            "Return only the translations, keeping the same numbering. "
            "Format: '1. <translation>\\n2. <translation>\\n...'"
        )
    else:
        system = (
            "You are a translator. Translate each numbered item to Brazilian Portuguese. "
            "Return only the translations, keeping the same numbering. "
            "Format: '1. <translation>\\n2. <translation>\\n...'"
        )
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    response = get_answer(client, system, numbered, config, model_name)
    lines = response.strip().split("\n")
    results = []
    for line in lines:
        match = re.match(r'^\d+\.\s*(.*)', line.strip())
        if match:
            results.append(match.group(1).strip())
    # fallback: se o parse falhar, mantém o que veio e completa com originais
    if len(results) != len(texts):
        results = (results + texts)[:len(texts)]
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FILTROS
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text):
    """Normaliza texto para comparação de duplicatas."""
    text = text.strip()
    text = re.sub(r'[.\s]+$', '', text)
    return text.lower()


def same_tokens(text1, text2):
    """Verifica se dois textos têm exatamente os mesmos tokens (ignora ordem)."""
    return set(text1.lower().split()) == set(text2.lower().split())


def different_tokens(original, paraphrase):
    """Calcula a diferença de tokens entre original e paráfrase."""
    tokens1 = set(original.lower().split())
    tokens2 = set(paraphrase.lower().split())
    diff = tokens1.symmetric_difference(tokens2)
    ratio = round(len(diff) / len(tokens1) * 100, 2) if tokens1 else 0.0
    return len(diff), ratio


def fix_punctuation(text):
    """
    Corrige pontuação duplicada causada pelo Pegasus ao parafrasear fragmentos.
    Ex: 'destruída., Não me sinto' → 'destruída, Não me sinto'
    """
    if not text:
        return text
    text = re.sub(r'[.!?]+\s*,', ',', text)    # ponto seguido de vírgula → só vírgula
    text = re.sub(r'([.!?]){2,}', r'\1', text)  # pontuação duplicada → uma só
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    return text


def find_replacement(original, candidates, exclude_text):
    """
    Busca a melhor paráfrase alternativa com BERTScore estritamente entre 0 e 100.
    Usada quando a melhor paráfrase tem score extremo (0 ou 100).
    """
    scored = []
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand == exclude_text.strip():
            continue
        scores = paraphrase_scores_batch(original, [cand])
        scored.append((cand, scores[0]))
    scored.sort(key=lambda x: x[1], reverse=True)
    for cand, score in scored:
        if 0.0 < score < 100.0:
            return cand, score
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO DAS PARÁFRASES
# ══════════════════════════════════════════════════════════════════════════════

def process_paraphrases(text_pt, paraphrases_pt, split_log):
    """
    Aplica os filtros e seleciona a melhor paráfrase.
    Integra: deduplicação, correção de pontuação, BERTScore e correção de scores extremos.
    """
    seen = set()
    paraphrases_unique = []
    count_duplicates = 0
    similar_to_original = False
    tokens_diff = []
    tokens_diff_pct = []

    for para in paraphrases_pt:
        # corrige pontuação duplicada antes de qualquer comparação
        para = fix_punctuation(para)

        key = normalize(para)
        if key in seen:
            count_duplicates += 1
        elif same_tokens(text_pt, para):
            # paráfrase tem os mesmos tokens do original: descarta
            similar_to_original = True
        else:
            seen.add(key)
            paraphrases_unique.append(para)
            n_diff, pct = different_tokens(text_pt, para)
            tokens_diff.append(n_diff)
            tokens_diff_pct.append(pct)

    # nenhuma paráfrase válida sobrou após os filtros
    if not paraphrases_unique:
        return {
            'best_para': '',
            'best_score': 0.0,
            'paraphrases_unique': [],
            'paraphrase_scores': [],
            'is_paraphrase_flags': [],
            'count_duplicates': count_duplicates,
            'similar_to_original': similar_to_original,
            'tokens_diff': tokens_diff,
            'tokens_diff_pct': tokens_diff_pct,
            'no_valid_paraphrase': True,
            'score_replaced': False,
            'split_log': split_log,
        }

    best_para, best_score, all_scores = select_best_paraphrase(text_pt, paraphrases_unique)
    is_paraphrase_flags = [s >= 80.0 for s in all_scores]
    score_replaced = False

    # correção de scores extremos: substitui ou exclui se score == 0 ou == 100
    if best_score == 0.0 or best_score == 100.0:
        new_text, new_score = find_replacement(text_pt, paraphrases_unique, best_para)
        if new_text is not None:
            best_para = new_text
            best_score = new_score
            score_replaced = True
        else:
            # sem alternativa válida: exclui a instância
            best_para = ''
            best_score = 0.0

    return {
        'best_para': best_para,
        'best_score': best_score,
        'paraphrases_unique': paraphrases_unique,
        'paraphrase_scores': all_scores,
        'is_paraphrase_flags': is_paraphrase_flags,
        'count_duplicates': count_duplicates,
        'similar_to_original': similar_to_original,
        'tokens_diff': tokens_diff,
        'tokens_diff_pct': tokens_diff_pct,
        'no_valid_paraphrase': not best_para,
        'score_replaced': score_replaced,
        'split_log': split_log,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_path, doc_type, nrows):
    if doc_type == 'jsonl':
        df = pd.read_json(dataset_path, orient='records', lines=True, encoding='utf-8')
    elif doc_type == 'csv':
        df = pd.read_csv(dataset_path)
        df = df.rename(columns={'DOCNO': 'id', 'TEXT': 'text', 'SYMPTOM': 'symptom'})
    elif doc_type == 'txt':
        txt_files = sorted(Path(dataset_path).glob('*.txt'))
        records = [{'id': f.stem, 'text': f.read_text(encoding='utf-8').strip()}
                   for f in txt_files]
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"doc_type inválido: {doc_type}. Opções: 'jsonl', 'csv' ou 'txt'.")
    if nrows:
        df = df.head(nrows)
    return df


def load_done_ids(output_stats_path):
    """Carrega IDs já processados para retomar execuções interrompidas."""
    done = set()
    p = Path(output_stats_path)
    if p.exists():
        with jsonlines.open(p) as reader:
            for obj in reader:
                done.add(obj['id'])
    return done


def chunked(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline de geração de paráfrases sintéticas para o corpus AMIVE.'
    )
    parser.add_argument('-d',  '--dataset',       type=Path, required=True,
                        help='Arquivo de entrada.')
    parser.add_argument('-t',  '--doc_type',      type=str,  default='txt',
                        help='Formato do arquivo: csv, jsonl ou txt (default: txt).')
    parser.add_argument('-n',  '--nrows',         type=int,  default=None,
                        help='Número de instâncias a processar (default: todas).')
    parser.add_argument('-c',  '--config',        type=Path, required=True,
                        help='YAML com configurações do LLM.')
    parser.add_argument('-m',  '--model',         type=str,  required=True,
                        help='Nome do modelo LLM.')
    parser.add_argument('-C',  '--client',        type=str,  required=True,
                        choices=['openai', 'groq', 'maritalk'])
    parser.add_argument('-o',  '--output',        type=Path, required=True,
                        help='CSV de saída com as paráfrases finais.')
    parser.add_argument('-os', '--output_stats',  type=Path, required=True,
                        help='JSONL com estatísticas detalhadas por instância.')
    parser.add_argument('-N',  '--num_sequences', type=int,  default=5,
                        help='Número de paráfrases por texto (default: 5).')
    parser.add_argument('-z',  '--sleep',         type=int,  default=1,
                        help='Segundos entre requisições ao LLM (default: 1).')
    parser.add_argument('-p',  '--paraphrase',    type=bool, default=True,
                        help='True para paráfrase completa, False para só back-translation.')
    parser.add_argument('-b',  '--batch_size',    type=int,  default=5,
                        help='Instâncias por batch de tradução (default: 5).')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    config = parse_yaml(args.config)
    client = build_client(args.client)

    print("Carregando dataset...")
    df = load_dataset(args.dataset, args.doc_type, args.nrows)
    df_full = df.copy()
    df_unique = df.drop_duplicates(subset='id').reset_index(drop=True)

    print("Carregando Pegasus...")
    tokenizer, model, device = load_pegasus()

    print("Verificando checkpoint...")
    done_ids = load_done_ids(args.output_stats)
    pending = [row for _, row in df_unique.iterrows() if row['id'] not in done_ids]
    print(f"{len(done_ids)} já processados, {len(pending)} pendentes.")

    no_valid_count = 0
    score_replaced_count = 0

    with jsonlines.open(args.output_stats, mode='a') as stats_writer:
        for batch in tqdm(list(chunked(pending, args.batch_size)), desc='Processando batches'):

            texts_pt = [row['text'] for row in batch]

            # 1. tradução PT → EN em batch
            texts_en = translate_batch(client, texts_pt, 'pt_en', config, args.model)
            time.sleep(args.sleep)

            for row, text_pt, text_en in zip(batch, texts_pt, texts_en):

                # 2. paráfrase em EN com split automático para textos longos
                if args.paraphrase:
                    paraphrases_en, split_log = get_paraphrases(
                        text_en, tokenizer, model, device, args.num_sequences
                    )
                else:
                    paraphrases_en, split_log = [text_en], []

                # sem paráfrases geradas (todos os fragmentos excederam o limite)
                if not paraphrases_en:
                    no_valid_count += 1
                    stats_writer.write({
                        'id': row['id'],
                        'original': text_pt,
                        'best_paraphrase': '',
                        'best_score': 0.0,
                        'translated_en': text_en,
                        'paraphrases_en': [],
                        'paraphrases_pt': [],
                        'is_paraphrase': [],
                        'paraphrase_score': [],
                        'duplicates_eliminated': f"0/{args.num_sequences}",
                        'similar_to_original_eliminated': False,
                        'no_valid_paraphrase': True,
                        'score_replaced': False,
                        'split_log': split_log,
                        'tokens_diff': '[]',
                    })
                    continue

                # 3. tradução EN → PT em batch
                paraphrases_pt = translate_batch(
                    client, paraphrases_en, 'en_pt', config, args.model
                )
                time.sleep(args.sleep)

                # 4. filtros + correção de pontuação + BERTScore + correção de scores extremos
                result = process_paraphrases(text_pt, paraphrases_pt, split_log)

                if result['no_valid_paraphrase']:
                    no_valid_count += 1
                if result['score_replaced']:
                    score_replaced_count += 1

                # 5. salva estatísticas
                stats_writer.write({
                    'id': row['id'],
                    'original': text_pt,
                    'best_paraphrase': result['best_para'],
                    'best_score': result['best_score'],
                    'translated_en': text_en,
                    'paraphrases_en': paraphrases_en,
                    'paraphrases_pt': result['paraphrases_unique'],
                    'is_paraphrase': result['is_paraphrase_flags'],
                    'paraphrase_score': result['paraphrase_scores'],
                    'duplicates_eliminated': f"{result['count_duplicates']}/{args.num_sequences}",
                    'similar_to_original_eliminated': result['similar_to_original'],
                    'no_valid_paraphrase': result['no_valid_paraphrase'],
                    'score_replaced': result['score_replaced'],
                    'split_log': result['split_log'],
                    'tokens_diff': (
                        f"{result['tokens_diff']} tokens diferentes "
                        f"({result['tokens_diff_pct']}% do original)"
                    ),
                })

    print(f"\nInstâncias sem paráfrase válida: {no_valid_count}/{len(pending)}")
    print(f"Instâncias com score corrigido (era 0 ou 100): {score_replaced_count}/{len(pending)}")

    # 6. reconstrói o mapa id → melhor paráfrase a partir do .jsonl completo
    paraphrase_map = {}
    with jsonlines.open(args.output_stats) as reader:
        for obj in reader:
            paraphrase_map[obj['id']] = obj['best_paraphrase']

    # 7. gera o CSV final
    df_full['text'] = df_full['id'].map(paraphrase_map)
    df_output = df_full.dropna(subset=['text'])
    df_output = df_output[df_output['text'].str.strip() != '']
    df_output = df_output.rename(columns={'id': 'DOCNO', 'text': 'TEXT', 'symptom': 'SYMPTOM'})
    df_output[['DOCNO', 'TEXT', 'SYMPTOM']].to_csv(args.output, index=False)

    print(f"CSV salvo em: {args.output}")
    print(f"Total de linhas no CSV final: {len(df_output)}")


if __name__ == '__main__':
    main()
