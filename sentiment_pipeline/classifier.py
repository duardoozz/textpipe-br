"""Módulo de classificação de texto.

Fornece um wrapper unificado para múltiplos classificadores com dois
modos de operação: manual (hiperparâmetros fixos) e auto (otimização
via RandomizedSearchCV com F1-macro).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform

logger = logging.getLogger(__name__)

# Espaços de busca para otimização automática de hiperparâmetros
SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "naive_bayes": {
        "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    },
    "logistic_regression": {
        "C": loguniform(1e-3, 1e2),
        "max_iter": [500, 1000, 2000],
        "solver": ["lbfgs", "saga"],
    },
    "linear_svc": {
        "C": loguniform(1e-3, 1e2),
        "max_iter": [1000, 3000, 5000],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20],
    },
}

class TextClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper unificado para classificadores de texto.

    Suporta cinco algoritmos e dois modos de operação:
        - **manual**: treina com hiperparâmetros fixos.
        - **auto**: otimiza hiperparâmetros via ``RandomizedSearchCV``
          maximizando F1-macro.

    Args:
        model_name: Nome do classificador. Valores aceitos:
            ``"naive_bayes"``, ``"logistic_regression"``,
            ``"linear_svc"``, ``"random_forest"``, ``"lightgbm"``.
        mode: Modo de operação (``"manual"`` ou ``"auto"``).
        params: Hiperparâmetros fixos para modo manual. Se ``None``,
            usa defaults razoáveis.
        random_state: Semente para reprodutibilidade.
        n_iter: Número de iterações do RandomizedSearchCV (modo auto).
        cv: Número de folds para cross-validation (modo auto).

    Example:
        >>> clf = TextClassifier("logistic_regression", mode="auto")
        >>> clf.fit(X_train_vec, y_train)
        >>> results = clf.evaluate(X_test_vec, y_test)
        >>> print(results["f1_macro"])
    """

    def __init__(
        self,
        model_name: str = "logistic_regression",
        mode: str = "manual",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_iter: int = 30,
        cv: int = 3,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.params = params
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv = cv

        self._model: Optional[Any] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._search: Optional[RandomizedSearchCV] = None

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> "TextClassifier":
        """Treina o classificador.

        No modo ``"manual"``, treina diretamente com os parâmetros
        fornecidos. No modo ``"auto"``, usa RandomizedSearchCV para
        encontrar os melhores hiperparâmetros.

        Args:
            X: Features de treino (matriz densa ou esparsa).
            y: Labels de treino.
            X_val: Features de validação (para report adicional).
            y_val: Labels de validação (para report adicional).

        Returns:
            self
        """
        base_model = self._create_base_model()

        if self.mode == "manual":
            if self.params:
                base_model.set_params(**self.params)
            self._model = base_model
            self._model.fit(X, y)
            self._best_params = self._model.get_params()

        elif self.mode == "auto":
            search_space = SEARCH_SPACES.get(self.model_name, {})
            self._search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=search_space,
                n_iter=min(self.n_iter, self._max_combinations(search_space)),
                scoring="f1_macro",
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
                error_score="raise",
            )
            self._search.fit(X, y)
            self._model = self._search.best_estimator_
            self._best_params = self._search.best_params_

            logger.info(
                "[%s] Melhor F1-macro (CV): %.4f | Params: %s",
                self.model_name,
                self._search.best_score_,
                self._best_params,
            )

        else:
            raise ValueError(
                f"Modo '{self.mode}' não reconhecido. "
                f"Use 'manual' ou 'auto'."
            )

        # Report de validação opcional
        if X_val is not None and y_val is not None:
            val_pred = self._model.predict(X_val)
            val_f1 = f1_score(y_val, val_pred, average="macro")
            logger.info(
                "[%s] F1-macro na validação: %.4f",
                self.model_name,
                val_f1,
            )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Gera predições para os dados fornecidos.

        Args:
            X: Features (matriz densa ou esparsa).

        Returns:
            Array de predições.
        """
        if self._model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self._model.predict(X)

    def evaluate(
        self,
        X_test: Any,
        y_test: Any,
    ) -> Dict[str, Any]:
        """Avalia o modelo no conjunto de teste.

        Args:
            X_test: Features de teste.
            y_test: Labels de teste.

        Returns:
            Dicionário com métricas:
                - ``f1_macro``: F1-score macro.
                - ``classification_report``: Relatório detalhado (str).
                - ``confusion_matrix``: Matriz de confusão (np.ndarray).
                - ``best_params``: Melhores hiperparâmetros.
        """
        y_pred = self.predict(X_test)
        return {
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "classification_report": classification_report(
                y_test, y_pred, digits=4
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "best_params": self._best_params,
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Retorna os melhores hiperparâmetros encontrados.

        Returns:
            Dicionário de hiperparâmetros.
        """
        return self._best_params or {}

    # Fabrica de modelos: cria a instancia base do classificador escolhido

    def _create_base_model(self) -> Any:
        """Cria a instância base do classificador.

        Returns:
            Instância do classificador scikit-learn ou compatível.

        Raises:
            ValueError: Se ``model_name`` não for reconhecido.
        """
        if self.model_name == "naive_bayes":
            from sklearn.naive_bayes import MultinomialNB
            return MultinomialNB()

        elif self.model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
            )

        elif self.model_name == "linear_svc":
            from sklearn.svm import LinearSVC
            return LinearSVC(
                random_state=self.random_state,
                max_iter=3000,
                dual="auto",
            )

        elif self.model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif self.model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1,
                )
            except ImportError:
                raise ImportError(
                    "LightGBM não instalado. "
                    "Instale com: pip install lightgbm"
                )

        else:
            raise ValueError(
                f"Modelo '{self.model_name}' não reconhecido. "
                f"Valores aceitos: {list(SEARCH_SPACES.keys())}"
            )

    @staticmethod
    def _max_combinations(space: Dict[str, Any]) -> int:
        """Estima o número máximo de combinações no espaço de busca.

        Args:
            space: Dicionário de distribuições/listas.

        Returns:
            Número estimado de combinações (capped para contínuas).
        """
        total = 1
        for v in space.values():
            if isinstance(v, list):
                total *= len(v)
            else:
                total *= 50  # Estimativa para distribuições contínuas
        return total
