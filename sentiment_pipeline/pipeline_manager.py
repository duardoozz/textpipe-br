"""Módulo de gestão de experimentos com scikit-learn Pipeline.

Utiliza ``sklearn.pipeline.Pipeline`` para encadear pré-processamento,
vetorização e classificação, e executa o produto cartesiano de todas
as configurações de forma estruturada.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

from sentiment_pipeline.preprocessor import TextPreprocessor
from sentiment_pipeline.vectorizer import TextVectorizer
from sentiment_pipeline.classifier import TextClassifier

logger = logging.getLogger(__name__)


class PipelineManager:
    """Orquestra experimentos de classificação de texto.

    Executa o produto cartesiano de múltiplas configurações de
    pré-processamento, vetorização e classificação, registrando
    métricas em um DataFrame comparativo.

    Cada combinação é montada como um ``sklearn.pipeline.Pipeline``
    com três estágios: ``preprocessor → vectorizer → classifier``.

    Args:
        experiment_name: Nome do experimento (usado em logs e arquivos).
        results_dir: Diretório para salvar resultados.
        random_state: Semente global para reprodutibilidade.

    Example:
        >>> manager = PipelineManager("exp_hatebr_v1")
        >>> results = manager.run_experiment(
        ...     X_train, X_val, X_test,
        ...     y_train, y_val, y_test,
        ...     preprocess_configs=[...],
        ...     vectorizer_configs=[...],
        ...     classifier_configs=[...],
        ... )
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        results_dir: str = "results",
        random_state: int = 42,
    ) -> None:
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.random_state = random_state

    def run_experiment(
        self,
        X_train: pd.Series,
        X_val: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        preprocess_configs: List[Dict[str, Any]],
        vectorizer_configs: List[Dict[str, Any]],
        classifier_configs: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Executa o experimento completo sobre todas as combinações.

        Itera sobre o produto cartesiano de todas as configurações,
        construindo um pipeline scikit-learn para cada combinação e
        avaliando no conjunto de teste.

        Args:
            X_train: Textos de treino (raw).
            X_val: Textos de validação (raw).
            X_test: Textos de teste (raw).
            y_train: Labels de treino.
            y_val: Labels de validação.
            y_test: Labels de teste.
            preprocess_configs: Lista de dicts com parâmetros para
                ``TextPreprocessor``.
            vectorizer_configs: Lista de dicts com parâmetros para
                ``TextVectorizer``.
            classifier_configs: Lista de dicts com parâmetros para
                ``TextClassifier``.

        Returns:
            DataFrame com métricas de cada combinação.
        """
        total = (
            len(preprocess_configs)
            * len(vectorizer_configs)
            * len(classifier_configs)
        )
        logger.info(
            "=== Experimento '%s' | %d combinações ===",
            self.experiment_name,
            total,
        )

        results: List[Dict[str, Any]] = []
        run_id = 0

        for pp_cfg, vec_cfg, clf_cfg in product(
            preprocess_configs,
            vectorizer_configs,
            classifier_configs,
        ):
            run_id += 1
            run_label = self._make_run_label(pp_cfg, vec_cfg, clf_cfg)
            logger.info(
                "[%d/%d] %s", run_id, total, run_label
            )

            try:
                row = self._run_single(
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    pp_cfg, vec_cfg, clf_cfg,
                    run_id,
                )
                results.append(row)
            except Exception as exc:
                logger.error(
                    "[%d/%d] ERRO: %s — %s",
                    run_id, total, run_label, exc,
                )
                results.append({
                    "run_id": run_id,
                    "preprocessor": str(pp_cfg),
                    "vectorizer": str(vec_cfg),
                    "classifier": str(clf_cfg),
                    "f1_macro": np.nan,
                    "accuracy": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "val_f1_macro": np.nan,
                    "train_time_s": np.nan,
                    "error": str(exc),
                })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            "f1_macro", ascending=False
        ).reset_index(drop=True)

        # Salvar resultados
        self._save_results(results_df)

        return results_df

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Gera um relatório textual comparativo dos resultados.

        Args:
            results_df: DataFrame retornado por ``run_experiment``.

        Returns:
            Relatório formatado em Markdown.
        """
        lines = [
            f"# Relatório — {self.experiment_name}",
            f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Total de runs**: {len(results_df)}",
            "",
            "## Top 10 Combinações (F1-macro)",
            "",
        ]

        # Colunas para exibição
        display_cols = [
            "run_id", "preprocessor", "vectorizer", "classifier",
            "f1_macro", "accuracy", "val_f1_macro", "train_time_s",
        ]
        available_cols = [
            c for c in display_cols if c in results_df.columns
        ]

        top10 = results_df.head(10)[available_cols]
        lines.append(top10.to_markdown(index=False))
        lines.append("")

        # Estatísticas resumidas
        valid = results_df.dropna(subset=["f1_macro"])
        if not valid.empty:
            lines.extend([
                "## Estatísticas Gerais",
                "",
                f"- **Melhor F1-macro**: {valid['f1_macro'].max():.4f}",
                f"- **Pior F1-macro**: {valid['f1_macro'].min():.4f}",
                f"- **Média F1-macro**: {valid['f1_macro'].mean():.4f}",
                f"- **Desvio padrão**: {valid['f1_macro'].std():.4f}",
                "",
            ])

            # Melhor combinação
            best = valid.iloc[0]
            lines.extend([
                "## Melhor Configuração",
                "",
                f"- **Preprocessor**: {best.get('preprocessor', 'N/A')}",
                f"- **Vectorizer**: {best.get('vectorizer', 'N/A')}",
                f"- **Classifier**: {best.get('classifier', 'N/A')}",
                f"- **F1-macro (teste)**: {best['f1_macro']:.4f}",
                f"- **Accuracy**: {best.get('accuracy', 'N/A')}",
                "",
            ])

        # Erros
        errors = results_df[results_df["f1_macro"].isna()]
        if not errors.empty:
            lines.extend([
                "## Erros",
                f"**{len(errors)} combinações falharam.**",
                "",
            ])

        return "\n".join(lines)

    # Execucao de um unico pipeline (preprocess -> vectorize -> classify)

    def _run_single(
        self,
        X_train: pd.Series,
        X_val: pd.Series,
        X_test: pd.Series,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        pp_cfg: Dict[str, Any],
        vec_cfg: Dict[str, Any],
        clf_cfg: Dict[str, Any],
        run_id: int,
    ) -> Dict[str, Any]:
        """Executa um único pipeline (preprocess → vectorize → classify).

        Args:
            X_train, X_val, X_test: Textos brutos.
            y_train, y_val, y_test: Labels.
            pp_cfg: Config do preprocessor.
            vec_cfg: Config do vectorizer.
            clf_cfg: Config do classifier.
            run_id: ID sequencial do run.

        Returns:
            Dicionário com métricas e metadados do run.
        """
        start = time.time()

        # 1. Pré-processamento
        preprocessor = TextPreprocessor(**pp_cfg)
        X_train_pp = preprocessor.fit_transform(X_train)
        X_val_pp = preprocessor.transform(X_val)
        X_test_pp = preprocessor.transform(X_test)

        # 2. Vetorização
        vectorizer = TextVectorizer(**vec_cfg)
        X_train_vec = vectorizer.fit_transform(X_train_pp)
        X_val_vec = vectorizer.transform(X_val_pp)
        X_test_vec = vectorizer.transform(X_test_pp)

        # 3. Classificação
        classifier = TextClassifier(**clf_cfg)

        # Naive Bayes requer valores não-negativos
        if clf_cfg.get("model_name") == "naive_bayes":
            import scipy.sparse
            if scipy.sparse.issparse(X_train_vec):
                X_train_vec = X_train_vec.toarray()
                X_val_vec = X_val_vec.toarray()
                X_test_vec = X_test_vec.toarray()
            X_train_vec = np.abs(X_train_vec)
            X_val_vec = np.abs(X_val_vec)
            X_test_vec = np.abs(X_test_vec)

        # LightGBM requer float32/float64 (BoW produz int)
        if clf_cfg.get("model_name") == "lightgbm":
            import scipy.sparse
            if scipy.sparse.issparse(X_train_vec):
                X_train_vec = X_train_vec.astype(np.float32)
                X_val_vec = X_val_vec.astype(np.float32)
                X_test_vec = X_test_vec.astype(np.float32)
            else:
                X_train_vec = np.asarray(X_train_vec, dtype=np.float32)
                X_val_vec = np.asarray(X_val_vec, dtype=np.float32)
                X_test_vec = np.asarray(X_test_vec, dtype=np.float32)

        classifier.fit(X_train_vec, y_train, X_val_vec, y_val)

        elapsed = time.time() - start

        # 4. Avaliação
        y_pred_test = classifier.predict(X_test_vec)
        y_pred_val = classifier.predict(X_val_vec)

        return {
            "run_id": run_id,
            "preprocessor": self._cfg_to_short_label(pp_cfg),
            "vectorizer": self._cfg_to_short_label(vec_cfg),
            "classifier": self._cfg_to_short_label(clf_cfg),
            "f1_macro": f1_score(y_test, y_pred_test, average="macro"),
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision_macro": precision_score(
                y_test, y_pred_test, average="macro"
            ),
            "recall_macro": recall_score(
                y_test, y_pred_test, average="macro"
            ),
            "val_f1_macro": f1_score(
                y_val, y_pred_val, average="macro"
            ),
            "train_time_s": round(elapsed, 2),
            "best_params": str(classifier.get_best_params()),
            "error": None,
        }

    # Metodos utilitarios para formatacao e persistencia de resultados

    @staticmethod
    def _make_run_label(
        pp_cfg: Dict, vec_cfg: Dict, clf_cfg: Dict
    ) -> str:
        """Gera label legível para um run."""
        pp = pp_cfg.get("stemming", False)
        neg = pp_cfg.get("handle_negations", False)
        vec = vec_cfg.get("method", "?")
        svd = vec_cfg.get("svd_components")
        clf = clf_cfg.get("model_name", "?")
        mode = clf_cfg.get("mode", "manual")

        parts = [
            f"stem={'Y' if pp else 'N'}",
            f"neg={'Y' if neg else 'N'}",
            f"vec={vec}" + (f"+svd{svd}" if svd else ""),
            f"clf={clf}({mode})",
        ]
        return " | ".join(parts)

    @staticmethod
    def _cfg_to_short_label(cfg: Dict[str, Any]) -> str:
        """Converte config em label curta para o DataFrame."""
        parts = []
        for k, v in cfg.items():
            if isinstance(v, bool):
                if v:
                    parts.append(k)
            elif v is not None:
                parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else "default"

    def _save_results(self, results_df: pd.DataFrame) -> None:
        """Salva resultados em CSV.

        Args:
            results_df: DataFrame com resultados de todos os runs.
        """
        out_dir = Path(self.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.csv"
        filepath = out_dir / filename

        results_df.to_csv(filepath, index=False)
        logger.info("Resultados salvos em: %s", filepath)

        # Salva relatório
        report = self.generate_report(results_df)
        report_path = out_dir / f"{self.experiment_name}_{timestamp}_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Relatório salvo em: %s", report_path)
