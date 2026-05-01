#!/usr/bin/env python3
"""
Script principal para execucao de um experimento completo de classificacao
de texto em portugues brasileiro.

O fluxo segue estas etapas:
  1. Carregamento do dataset (HateBR por padrao)
  2. Divisao estratificada em treino/validacao/teste (70/15/15)
  3. Definicao de multiplas configs de pre-processamento, vetorizacao e classificacao
  4. Execucao do produto cartesiano de todas as combinacoes
  5. Exibicao e persistencia dos resultados

Uso:
    python main.py
    python main.py --dataset toldbr
    python main.py --dataset b2w --data-path data/B2W-Reviews01.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from sentiment_pipeline.data_loader import DatasetLoader
from sentiment_pipeline.pipeline_manager import PipelineManager

# Silencia avisos internos do scikit-learn para manter a saida limpa
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Logger global com formato padronizado: hora, nivel, modulo, mensagem
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Cada lista abaixo define variacoes de parametros combinadas entre si
# pelo PipelineManager no formato produto cartesiano.

# Config 1 - Basica: lowercase, remove URLs/mencoes/pontuacao/numeros/stopwords,
# preserva negacoes, sem stemming, remove emojis.
# Config 2 - Agressiva: mesma base + stemming RSLP + marcacao _NEG + demojize.
# Config 3 - Minima: apenas lowercase e remocao de URLs/mencoes. Baseline.
PREPROCESS_CONFIGS = [
    {
        "lowercase": True, "remove_urls": True, "remove_mentions": True,
        "remove_punctuation": True, "remove_numbers": True,
        "remove_stopwords": True, "keep_negations": True,
        "handle_negations": False, "stemming": False,
        "handle_emojis": "remove", "min_token_length": 2,
    },
    {
        "lowercase": True, "remove_urls": True, "remove_mentions": True,
        "remove_punctuation": True, "remove_numbers": True,
        "remove_stopwords": True, "keep_negations": True,
        "handle_negations": True, "stemming": True,
        "handle_emojis": "demojize", "min_token_length": 2,
    },
    {
        "lowercase": True, "remove_urls": True, "remove_mentions": True,
        "remove_punctuation": False, "remove_numbers": False,
        "remove_stopwords": False, "keep_negations": False,
        "handle_negations": False, "stemming": False,
        "handle_emojis": "keep", "min_token_length": 1,
    },
]

# Vetorizacao: BoW, TF-IDF, e TF-IDF + SVD com 50/100/300 componentes
VECTORIZER_CONFIGS = [
    {"method": "bow", "max_features": 10_000, "ngram_range": (1, 1)},
    {"method": "tfidf", "max_features": 10_000, "ngram_range": (1, 2)},
    {"method": "tfidf", "max_features": 10_000, "ngram_range": (1, 2), "svd_components": 50},
    {"method": "tfidf", "max_features": 10_000, "ngram_range": (1, 2), "svd_components": 100},
    {"method": "tfidf", "max_features": 10_000, "ngram_range": (1, 2), "svd_components": 300},
]

# Classificadores: manual = defaults fixos, auto = RandomizedSearchCV (F1-macro)
CLASSIFIER_CONFIGS = [
    {"model_name": "naive_bayes", "mode": "manual"},
    {"model_name": "logistic_regression", "mode": "manual"},
    {"model_name": "logistic_regression", "mode": "auto", "n_iter": 20},
    {"model_name": "linear_svc", "mode": "manual"},
    {"model_name": "linear_svc", "mode": "auto", "n_iter": 20},
    {"model_name": "random_forest", "mode": "manual"},
    {"model_name": "lightgbm", "mode": "manual"},
]


def parse_args() -> argparse.Namespace:
    """Processa os argumentos recebidos pela linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline Dinamico de Classificacao de Texto PT-BR",
    )
    parser.add_argument(
        "--dataset", type=str, default="hatebr",
        choices=["hatebr", "toldbr", "b2w"],
        help="Dataset a ser utilizado (default: hatebr)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Caminho para o arquivo CSV do dataset.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Diretorio para salvar resultados (default: results)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semente para reprodutibilidade (default: 42)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rapido: usa apenas 1 config de cada tipo",
    )
    return parser.parse_args()


def get_default_path(dataset: str) -> str:
    """Retorna o caminho padrao para o CSV do dataset escolhido.

    Args:
        dataset: Identificador do dataset ("hatebr", "toldbr" ou "b2w").

    Returns:
        Caminho absoluto para o arquivo CSV correspondente.
    """
    base = Path(__file__).parent
    paths = {
        "hatebr": base / "hatebr.csv",
        "toldbr": base / "told_br_multilabel.csv",
        "b2w": base / "data" / "B2W-Reviews01.csv",
    }
    return str(paths[dataset])


def main() -> None:
    """Funcao principal que orquestra a execucao do experimento.

    Carrega dados, divide em particoes estratificadas, executa todas
    as combinacoes de pipeline e exibe os resultados ordenados por F1-macro.
    """
    args = parse_args()

    logger.info("PIPELINE DINAMICO DE CLASSIFICACAO DE TEXTO (PT-BR)")

    # Carregamento dos dados
    data_path = args.data_path or get_default_path(args.dataset)
    logger.info("Dataset: %s | Path: %s", args.dataset, data_path)

    loader = DatasetLoader(random_state=args.random_state)
    df = loader.load_and_standardize(data_path, args.dataset)

    # Divisao estratificada: mantem proporcao das classes em cada particao
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df)

    logger.info(
        "Distribuicao - Train: %s | Val: %s | Test: %s",
        y_train.value_counts().to_dict(),
        y_val.value_counts().to_dict(),
        y_test.value_counts().to_dict(),
    )

    # No modo rapido, usa apenas uma amostra de cada tipo para validacao
    if args.quick:
        pp_configs = PREPROCESS_CONFIGS[:1]
        vec_configs = VECTORIZER_CONFIGS[:2]
        clf_configs = CLASSIFIER_CONFIGS[:2]
        logger.info("MODO RAPIDO: configs reduzidas")
    else:
        pp_configs = PREPROCESS_CONFIGS
        vec_configs = VECTORIZER_CONFIGS
        clf_configs = CLASSIFIER_CONFIGS

    total = len(pp_configs) * len(vec_configs) * len(clf_configs)
    logger.info(
        "Configuracoes: %d preprocess x %d vectorizer x %d classifier = %d runs",
        len(pp_configs), len(vec_configs), len(clf_configs), total,
    )

    # O PipelineManager executa o produto cartesiano de todas as configs
    manager = PipelineManager(
        experiment_name=f"exp_{args.dataset}",
        results_dir=args.results_dir,
        random_state=args.random_state,
    )

    results_df = manager.run_experiment(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocess_configs=pp_configs,
        vectorizer_configs=vec_configs,
        classifier_configs=clf_configs,
    )

    # Exibicao dos resultados
    logger.info("RESULTADOS FINAIS")

    display_cols = [
        "run_id", "preprocessor", "vectorizer", "classifier",
        "f1_macro", "accuracy", "val_f1_macro", "train_time_s",
    ]
    available = [c for c in display_cols if c in results_df.columns]

    # Tenta formatar com tabulate; caso nao instalado, usa pandas
    try:
        from tabulate import tabulate
        print("\n" + tabulate(
            results_df[available].head(15),
            headers="keys", tablefmt="grid",
            floatfmt=".4f", showindex=False,
        ))
    except ImportError:
        print("\n" + results_df[available].head(15).to_string(index=False))

    # Destaca o melhor pipeline por F1-macro no teste
    if not results_df.empty and not results_df["f1_macro"].isna().all():
        best = results_df.iloc[0]
        logger.info(
            "MELHOR RESULTADO - F1-macro: %.4f | %s | %s | %s",
            best["f1_macro"], best["preprocessor"],
            best["vectorizer"], best["classifier"],
        )

    # Relatorio textual consolidado
    report = manager.generate_report(results_df)
    print("\n" + report)


if __name__ == "__main__":
    main()
