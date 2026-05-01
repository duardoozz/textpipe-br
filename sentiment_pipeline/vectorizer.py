"""Módulo de vetorização de texto.

Implementa um transformador compatível com scikit-learn que converte
textos pré-processados em representações numéricas usando Bag-of-Words,
TF-IDF (com SVD opcional) ou Word2Vec (média de vetores).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class TextVectorizer(BaseEstimator, TransformerMixin):
    """Vetorizador de texto flexível, compatível com scikit-learn.

    Suporta três métodos de vetorização:
        - ``"bow"``: Bag-of-Words via ``CountVectorizer``.
        - ``"tfidf"``: TF-IDF via ``TfidfVectorizer``, com redução de
          dimensionalidade opcional via ``TruncatedSVD``.
        - ``"word2vec"``: Média dos vetores Word2Vec de cada documento.
          Aceita modelos pré-treinados ou treina no próprio corpus.

    Args:
        method: Método de vetorização (``"bow"``, ``"tfidf"``,
            ``"word2vec"``).
        max_features: Número máximo de features (BoW/TF-IDF).
        ngram_range: Faixa de n-grams (BoW/TF-IDF).
        svd_components: Se definido, aplica TruncatedSVD com esse
            número de componentes após TF-IDF. Valores comuns: 50, 100,
            300.
        w2v_size: Dimensão dos vetores Word2Vec.
        w2v_window: Janela de contexto do Word2Vec.
        w2v_min_count: Frequência mínima para inclusão no vocabulário
            Word2Vec.
        w2v_pretrained_path: Caminho para modelo pré-treinado
            (formato ``KeyedVectors``). Se ``None``, treina no corpus.
        random_state: Semente para reprodutibilidade.

    Example:
        >>> vec = TextVectorizer(method="tfidf", svd_components=100)
        >>> X_train_vec = vec.fit_transform(X_train_processed)
        >>> X_test_vec = vec.transform(X_test_processed)
    """

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 10_000,
        ngram_range: tuple = (1, 1),
        svd_components: Optional[int] = None,
        w2v_size: int = 300,
        w2v_window: int = 5,
        w2v_min_count: int = 2,
        w2v_pretrained_path: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.svd_components = svd_components
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count
        self.w2v_pretrained_path = w2v_pretrained_path
        self.random_state = random_state

        # Modelos internos (populados durante fit)
        self._vectorizer: Optional[Any] = None
        self._svd: Optional[TruncatedSVD] = None
        self._w2v_model: Optional[Any] = None

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> "TextVectorizer":
        """Ajusta o vetorizador nos dados de treino.

        Args:
            X: Iterável de strings pré-processadas.
            y: Labels (ignorados).

        Returns:
            self
        """
        texts = list(X)

        if self.method == "bow":
            self._vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            self._vectorizer.fit(texts)

        elif self.method == "tfidf":
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            self._vectorizer.fit(texts)

            # SVD opcional
            if self.svd_components is not None:
                self._svd = TruncatedSVD(
                    n_components=self.svd_components,
                    random_state=self.random_state,
                )
                tfidf_matrix = self._vectorizer.transform(texts)
                self._svd.fit(tfidf_matrix)
                logger.info(
                    "SVD aplicado: %d componentes | "
                    "Variância explicada: %.2f%%",
                    self.svd_components,
                    self._svd.explained_variance_ratio_.sum() * 100,
                )

        elif self.method == "word2vec":
            self._fit_word2vec(texts)

        else:
            raise ValueError(
                f"Método '{self.method}' não reconhecido. "
                f"Valores aceitos: 'bow', 'tfidf', 'word2vec'."
            )

        return self

    def transform(
        self,
        X: Any,
        y: Any = None,
    ) -> np.ndarray:
        """Transforma textos em representações numéricas.

        Args:
            X: Iterável de strings pré-processadas.
            y: Labels (ignorados).

        Returns:
            Matriz numpy (densa) ou scipy sparse com as
            representações vetoriais.
        """
        texts = list(X)

        if self.method in ("bow", "tfidf"):
            result = self._vectorizer.transform(texts)
            if self._svd is not None:
                result = self._svd.transform(result)
            return result

        elif self.method == "word2vec":
            return self._transform_word2vec(texts)

        else:
            raise ValueError(f"Método '{self.method}' não reconhecido.")

    # Metodos auxiliares para treinamento e transformacao Word2Vec

    def _fit_word2vec(self, texts: List[str]) -> None:
        """Treina ou carrega modelo Word2Vec.

        Args:
            texts: Lista de textos pré-processados.
        """
        if self.w2v_pretrained_path:
            self._load_pretrained_w2v()
        else:
            self._train_w2v(texts)

    def _load_pretrained_w2v(self) -> None:
        """Carrega vetores Word2Vec pré-treinados."""
        try:
            from gensim.models import KeyedVectors

            logger.info(
                "Carregando vetores pré-treinados de: %s",
                self.w2v_pretrained_path,
            )
            self._w2v_model = KeyedVectors.load(
                self.w2v_pretrained_path
            )
            # Atualiza dimensão para coincidir com modelo carregado
            self.w2v_size = self._w2v_model.vector_size
        except Exception as exc:
            raise RuntimeError(
                f"Erro ao carregar vetores pré-treinados: {exc}"
            ) from exc

    def _train_w2v(self, texts: List[str]) -> None:
        """Treina Word2Vec no corpus fornecido.

        Args:
            texts: Lista de textos pré-processados.
        """
        try:
            from gensim.models import Word2Vec

            tokenized = [text.split() for text in texts]
            model = Word2Vec(
                sentences=tokenized,
                vector_size=self.w2v_size,
                window=self.w2v_window,
                min_count=self.w2v_min_count,
                workers=1,  # Reprodutibilidade
                seed=self.random_state,
            )
            self._w2v_model = model.wv
            logger.info(
                "Word2Vec treinado: vocabulário de %d palavras | "
                "dimensão %d",
                len(self._w2v_model),
                self.w2v_size,
            )
        except ImportError:
            raise ImportError(
                "A biblioteca 'gensim' é necessária para Word2Vec. "
                "Instale com: pip install gensim"
            )

    def _transform_word2vec(self, texts: List[str]) -> np.ndarray:
        """Transforma textos usando média dos vetores Word2Vec.

        Para cada documento, calcula a média dos vetores das palavras
        presentes no vocabulário. Documentos sem nenhuma palavra
        conhecida recebem vetor zero.

        Args:
            texts: Lista de textos pré-processados.

        Returns:
            Matriz numpy (n_docs, w2v_size).
        """
        result = np.zeros((len(texts), self.w2v_size))

        for i, text in enumerate(texts):
            tokens = text.split()
            vectors = [
                self._w2v_model[token]
                for token in tokens
                if token in self._w2v_model
            ]
            if vectors:
                result[i] = np.mean(vectors, axis=0)

        return result
