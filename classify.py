# classify.py
# Classificação binária (sintoma vs. controle) com TF-IDF e múltiplos classificadores.
# Avalia corpus original e sintético em 4 cenários de treino/teste.
# Calcula também a similaridade de Jaccard entre pares de classificadores.

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, jaccard_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classificação binária: corpus original vs sintético.'
    )
    parser.add_argument('-d',   '--dataset',                    type=Path, required=True,
                        help='Dataset original com sintomas.')
    parser.add_argument('-ds',  '--dataset_sintetico',          type=Path, required=True,
                        help='Dataset sintético com sintomas.')
    parser.add_argument('-dc',  '--dataset_controle',           type=Path, required=True,
                        help='Dataset de controle original.')
    parser.add_argument('-dcs', '--dataset_controle_sintetico', type=Path, required=True,
                        help='Dataset de controle sintético.')
    parser.add_argument('-s',   '--symptom',                    type=str,  required=True,
                        help='Sintoma de interesse (string exata conforme coluna SYMPTOM).')
    parser.add_argument('-o',   '--output',                     type=Path, required=True,
                        help='CSV com métricas de classificação.')
    parser.add_argument('-oj',  '--output_jaccard',             type=Path, required=True,
                        help='CSV com similaridade de Jaccard entre classificadores.')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proporção do conjunto de teste (default: 0.3).')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Seed para reprodutibilidade (default: 42).')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO E PREPARAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_path, dataset_sintetico_path, dataset_controle_path,
              dataset_controle_sintetico_path, symptom, seed):

    df_original = pd.read_csv(dataset_path)
    df_original = df_original.rename(columns={'DOCNO': 'id', 'TEXT': 'text', 'SYMPTOM': 'symptom'})

    df_sintetico = pd.read_csv(dataset_sintetico_path)
    df_sintetico = df_sintetico.rename(columns={'DOCNO': 'id', 'TEXT': 'text', 'SYMPTOM': 'symptom'})
    df_sintetico = df_sintetico.dropna(subset=['text'])
    df_sintetico = df_sintetico[df_sintetico['text'].str.strip() != '']

    df_controle = pd.read_csv(dataset_controle_path)
    df_controle = df_controle.rename(columns={'DOCNO': 'id', 'TEXT': 'text'})

    df_controle_sint = pd.read_csv(dataset_controle_sintetico_path)
    df_controle_sint = df_controle_sint.rename(columns={'DOCNO': 'id', 'TEXT': 'text'})
    df_controle_sint = df_controle_sint.dropna(subset=['text'])
    df_controle_sint = df_controle_sint[df_controle_sint['text'].str.strip() != '']

    # filtra e deduplica pelo sintoma de interesse
    df_sint_orig = df_original[df_original['symptom'] == symptom][['id', 'text']].drop_duplicates(subset='id')
    df_sint_sint = df_sintetico[df_sintetico['symptom'] == symptom][['id', 'text']].drop_duplicates(subset='id')

    # usa apenas os IDs presentes no sintético
    ids_sint = set(df_sint_sint['id'].unique())
    df_sint_orig = df_sint_orig[df_sint_orig['id'].isin(ids_sint)]

    n_orig = len(df_sint_orig)
    n_sint = len(df_sint_sint)
    assert n_orig == n_sint, f"Tamanhos diferentes (sintoma): original={n_orig}, sintético={n_sint}."
    print(f"Instâncias com sintoma '{symptom}': {n_orig}")

    # controle: filtra pelos IDs que têm par sintético
    df_controle      = df_controle[['id', 'text']].drop_duplicates(subset='id')
    df_controle_sint = df_controle_sint[['id', 'text']].drop_duplicates(subset='id')

    ids_ctrl_sint = set(df_controle_sint['id'].unique())
    df_ctrl_orig_filtrado = df_controle[df_controle['id'].isin(ids_ctrl_sint)]

    n_ctrl_orig = len(df_ctrl_orig_filtrado)
    n_ctrl_sint = len(df_controle_sint)
    assert n_ctrl_orig == n_ctrl_sint, \
        f"Tamanhos diferentes (controle): original={n_ctrl_orig}, sintético={n_ctrl_sint}."

    if n_ctrl_orig < n_orig:
        raise ValueError(f"Controle com menos instâncias pareadas ({n_ctrl_orig}) do que necessário ({n_orig}).")

    # amostra x instâncias de controle (x = tamanho do dataset com sintoma), mantendo o par
    ids_amostrados       = df_ctrl_orig_filtrado.sample(n=n_orig, random_state=seed)['id'].tolist()
    df_ctrl_orig_sample  = df_ctrl_orig_filtrado[df_ctrl_orig_filtrado['id'].isin(ids_amostrados)]
    df_ctrl_sint_sample  = df_controle_sint[df_controle_sint['id'].isin(ids_amostrados)]

    print(f"Controle selecionado: {len(df_ctrl_orig_sample)} instâncias")
    print(f"Total por dataset: {n_orig + len(df_ctrl_orig_sample)} (50% sintoma, 50% controle)")

    return df_sint_orig, df_sint_sint, df_ctrl_orig_sample, df_ctrl_sint_sample


def build_dataset(df_sintoma, df_controle, seed):
    df_pos = df_sintoma.copy()
    df_pos['label'] = 1
    df_neg = df_controle.copy()
    df_neg['label'] = 0
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    n_before = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    if len(df) < n_before:
        print(f"Atenção: {n_before - len(df)} linhas removidas por texto vazio/nulo.")

    # embaralha para evitar bias de ordem
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_by_id(df, test_size, seed):
    """Divide treino/teste por ID para evitar vazamento de dados."""
    ids = df['id'].unique()
    ids_train, ids_test = train_test_split(ids, test_size=test_size, random_state=seed)
    return df[df['id'].isin(ids_train)], df[df['id'].isin(ids_test)]


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICAÇÃO E AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, corpus, model_name):
    return {
        'corpus':    corpus,
        'model':     model_name,
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_tfidf_classifiers(X_train, y_train, X_test, y_test, corpus):
    """Treina e avalia os classificadores TF-IDF. Retorna métricas e predições."""
    vectorizer    = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM':                 SVC(kernel='linear', random_state=42),
        'Naive Bayes':         MultinomialNB(),
        'Random Forest':       RandomForestClassifier(random_state=42),
    }

    results, predictions = [], {}
    for name, clf in classifiers.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        results.append(evaluate(y_test, y_pred, corpus, name))
        predictions[name] = y_pred

    return results, predictions


def compute_jaccard_matrix(predictions, corpus, y_true):
    """
    Calcula a similaridade de Jaccard entre pares de classificadores.
    Mede concordância geral nas predições, nos acertos e nos erros.
    """
    rows        = []
    y_true_arr  = np.array(y_true)

    for m1, m2 in combinations(predictions.keys(), 2):
        y1, y2 = predictions[m1], predictions[m2]

        # Jaccard entre os vetores de predição (concordância geral)
        j_pred = round(jaccard_score(y1, y2, zero_division=0), 4)

        # Jaccard apenas nos acertos
        correct1          = (y1 == y_true_arr)
        correct2          = (y2 == y_true_arr)
        both_correct      = (correct1 & correct2).sum()
        at_least_one_corr = (correct1 | correct2).sum()
        j_acertos = round(both_correct / at_least_one_corr, 4) if at_least_one_corr > 0 else 0.0

        # Jaccard apenas nos erros
        wrong1             = ~correct1
        wrong2             = ~correct2
        both_wrong         = (wrong1 & wrong2).sum()
        at_least_one_wrong = (wrong1 | wrong2).sum()
        j_erros = round(both_wrong / at_least_one_wrong, 4) if at_least_one_wrong > 0 else 0.0

        rows.append({
            'corpus':             corpus,
            'modelo_1':           m1,
            'modelo_2':           m2,
            'jaccard_predicoes':  j_pred,
            'jaccard_acertos':    j_acertos,
            'jaccard_erros':      j_erros,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\nSintoma: {args.symptom}")
    print("=" * 60)

    df_sint_orig, df_sint_sint, df_ctrl_orig, df_ctrl_sint = load_data(
        args.dataset, args.dataset_sintetico,
        args.dataset_controle, args.dataset_controle_sintetico,
        args.symptom, args.seed
    )

    # original usa controle original; sintético usa controle sintético (mesmos IDs)
    df_orig = build_dataset(df_sint_orig, df_ctrl_orig, args.seed)
    df_sint = build_dataset(df_sint_sint, df_ctrl_sint, args.seed)

    # divide pelos mesmos IDs para garantir comparabilidade entre cenários
    df_orig_train, df_orig_test = split_by_id(df_orig, args.test_size, args.seed)
    ids_train     = df_orig_train['id'].unique()
    ids_test      = df_orig_test['id'].unique()
    df_sint_train = df_sint[df_sint['id'].isin(ids_train)]
    df_sint_test  = df_sint[df_sint['id'].isin(ids_test)]

    print(f"\nTreino: {len(df_orig_train)} | Teste: {len(df_orig_test)}")

    all_results, all_jaccard = [], []
    y_test_orig = df_orig_test['label'].tolist()
    y_test_sint = df_sint_test['label'].tolist()

    cenarios = [
        ('ORIGINAL → ORIGINAL',  'original',
         df_orig_train, df_orig_test, y_test_orig),
        ('SINTÉTICO → SINTÉTICO', 'sintetico',
         df_sint_train, df_sint_test, y_test_sint),
        ('SINTÉTICO → ORIGINAL',  'sintetico_treino_original_teste',
         df_sint_train, df_orig_test, y_test_orig),
        ('ORIGINAL → SINTÉTICO',  'original_treino_sintetico_teste',
         df_orig_train, df_sint_test, y_test_sint),
    ]

    for label, corpus, train_df, test_df, y_test in cenarios:
        print(f"\nCenário: {label}")
        results, preds = run_tfidf_classifiers(
            train_df['text'].tolist(), train_df['label'].tolist(),
            test_df['text'].tolist(), y_test, corpus
        )
        all_results.extend(results)
        all_jaccard.extend(compute_jaccard_matrix(preds, corpus, y_test))

    # salva métricas de classificação
    df_results = pd.DataFrame(all_results)
    df_results['symptom'] = args.symptom
    df_results = df_results[['symptom', 'corpus', 'model', 'accuracy', 'precision', 'recall', 'f1']]
    print("\nResultados:")
    print(df_results.to_string(index=False))
    df_results.to_csv(args.output, index=False)
    print(f"\nResultados salvos em: {args.output}")

    # salva matrizes de Jaccard entre classificadores
    df_jaccard = pd.DataFrame(all_jaccard)
    df_jaccard['symptom'] = args.symptom
    df_jaccard = df_jaccard[['symptom', 'corpus', 'modelo_1', 'modelo_2',
                              'jaccard_predicoes', 'jaccard_acertos', 'jaccard_erros']]
    print("\nJaccard entre classificadores:")
    print(df_jaccard.to_string(index=False))
    df_jaccard.to_csv(args.output_jaccard, index=False)
    print(f"\nJaccard salvo em: {args.output_jaccard}")


if __name__ == '__main__':
    main()
