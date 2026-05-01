"""Módulo de pré-processamento de texto para PT-BR.

Implementa um transformador compatível com scikit-learn que aplica
uma sequência configurável de etapas de limpeza e normalização a
textos em Português Brasileiro.
"""

from __future__ import annotations

import re
import string
import logging
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Palavras de negação em Português Brasileiro
NEGATION_WORDS = frozenset({
    "não", "nao", "nunca", "jamais", "nem", "nenhum",
    "nenhuma", "ninguém", "ninguem", "nada", "tampouco",
})

# Pontuações que encerram escopo de negação
NEGATION_BREAKERS = frozenset({".", ",", ";", "!", "?", ":"})


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Pré-processador de texto configurável, compatível com scikit-learn.

    Aceita parâmetros que controlam cada etapa do pipeline de limpeza.
    Pode ser usado diretamente em um ``sklearn.pipeline.Pipeline``.

    Args:
        lowercase: Converter texto para minúsculas.
        remove_urls: Remover URLs (http/https/www).
        remove_mentions: Remover menções (@usuario) e hashtags.
        remove_punctuation: Remover sinais de pontuação.
        remove_numbers: Remover dígitos numéricos.
        remove_stopwords: Remover stopwords do NLTK (pt).
        keep_negations: Se ``True``, preserva palavras de negação mesmo
            quando ``remove_stopwords=True``.
        handle_negations: Se ``True``, adiciona sufixo ``_NEG`` a todos
            os tokens após uma palavra de negação até a próxima
            pontuação ou fim da frase.
        stemming: Aplicar RSLP Stemmer (Português).
        handle_emojis: Estratégia para emojis. Valores:
            ``"remove"`` — remove todos os emojis;
            ``"demojize"`` — converte para descrição textual (PT);
            ``"keep"`` — mantém como estão.
        min_token_length: Comprimento mínimo de token para manter.

    Example:
        >>> from sentiment_pipeline.preprocessor import TextPreprocessor
        >>> pp = TextPreprocessor(handle_negations=True, stemming=True)
        >>> pp.transform(["Eu não gostei do produto"])
        ['nao gost_NEG produt_NEG']
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
        keep_negations: bool = True,
        handle_negations: bool = False,
        stemming: bool = False,
        handle_emojis: str = "remove",
        min_token_length: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.keep_negations = keep_negations
        self.handle_negations = handle_negations
        self.stemming = stemming
        self.handle_emojis = handle_emojis
        self.min_token_length = min_token_length

        # Lazy-loaded resources
        self._stopwords: Optional[frozenset] = None
        self._stemmer: Optional[Any] = None

    def fit(
        self,
        X: Any,
        y: Any = None,
    ) -> "TextPreprocessor":
        """Ajusta o pré-processador (no-op, transformador stateless).

        Args:
            X: Dados de entrada (ignorados).
            y: Labels (ignorados).

        Returns:
            self
        """
        return self

    def transform(self, X: Any, y: Any = None) -> List[str]:
        """Aplica o pipeline de pré-processamento a uma coleção de textos.

        Args:
            X: Iterável de strings (list, pd.Series, np.array).
            y: Labels (ignorados).

        Returns:
            Lista de strings pré-processadas.
        """
        return [self._process_single(str(text)) for text in X]

    def _process_single(self, text: str) -> str:
        """Processa um único texto pela sequência de etapas.

        Args:
            text: Texto bruto.

        Returns:
            Texto pré-processado.
        """
        # 1. Emojis (antes de lowercase para preservar padrões)
        text = self._apply_emoji_strategy(text)

        # 2. Lowercase
        if self.lowercase:
            text = text.lower()

        # 3. URLs
        if self.remove_urls:
            text = re.sub(
                r"https?://\S+|www\.\S+", " ", text
            )

        # 4. Menções e hashtags
        if self.remove_mentions:
            text = re.sub(r"@\w+", " ", text)
            text = re.sub(r"#\w+", " ", text)

        # 5. Números
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # 6. Pontuação (preserva temporariamente se handle_negations
        #    para usar como breaker)
        if self.remove_punctuation and not self.handle_negations:
            text = text.translate(
                str.maketrans("", "", string.punctuation)
            )

        # 7. Tokenização simples
        tokens = text.split()

        # 8. Handle negations — adiciona _NEG
        if self.handle_negations:
            tokens = self._apply_negation_handling(tokens)

        # Remove pontuação após negation handling
        if self.remove_punctuation and self.handle_negations:
            tokens = [
                t.translate(str.maketrans("", "", string.punctuation))
                for t in tokens
            ]
            tokens = [t for t in tokens if t]

        # 9. Stopwords
        if self.remove_stopwords:
            stopwords = self._get_stopwords()
            if self.keep_negations:
                # Remove stopwords mas mantém negações
                tokens = [
                    t for t in tokens
                    if t.rstrip("_NEG").lower() in NEGATION_WORDS
                    or t.lower() not in stopwords
                ]
            else:
                tokens = [
                    t for t in tokens if t.lower() not in stopwords
                ]

        # 10. Stemming
        if self.stemming:
            stemmer = self._get_stemmer()
            tokens = [
                self._stem_token(t, stemmer) for t in tokens
            ]

        # 11. Filtro por comprimento mínimo
        if self.min_token_length > 1:
            tokens = [
                t for t in tokens
                if len(t.rstrip("_NEG")) >= self.min_token_length
            ]

        return " ".join(tokens)

    # Metodos auxiliares de estrategias de pre-processamento

    def _apply_emoji_strategy(self, text: str) -> str:
        """Aplica a estratégia de tratamento de emojis.

        Args:
            text: Texto bruto.

        Returns:
            Texto com emojis tratados.
        """
        if self.handle_emojis == "keep":
            return text

        try:
            import emoji
        except ImportError:
            logger.warning(
                "Biblioteca 'emoji' não instalada. "
                "Emojis serão mantidos."
            )
            return text

        if self.handle_emojis == "demojize":
            return emoji.demojize(text, language="pt")
        elif self.handle_emojis == "remove":
            return emoji.replace_emoji(text, replace="")
        else:
            return text

    def _apply_negation_handling(
        self,
        tokens: List[str],
    ) -> List[str]:
        """Adiciona sufixo ``_NEG`` a tokens após uma negação.

        O escopo da negação vai da palavra de negação até a próxima
        pontuação (.,;!?:) ou fim da sequência.

        Args:
            tokens: Lista de tokens.

        Returns:
            Lista de tokens com sufixo ``_NEG`` aplicado.
        """
        result: List[str] = []
        in_negation = False

        for token in tokens:
            clean = token.lower().strip(string.punctuation)

            # Verifica se é um breaker de negação
            if any(c in NEGATION_BREAKERS for c in token):
                in_negation = False
                result.append(token)
                continue

            # Verifica se é uma palavra de negação
            if clean in NEGATION_WORDS:
                in_negation = True
                result.append(token)
                continue

            # Aplica sufixo se em escopo de negação
            if in_negation:
                result.append(f"{token}_NEG")
            else:
                result.append(token)

        return result

    def _stem_token(self, token: str, stemmer: Any) -> str:
        """Aplica stemming preservando sufixo _NEG.

        Args:
            token: Token a ser processado.
            stemmer: Instância do RSLP Stemmer.

        Returns:
            Token com stem aplicado.
        """
        if token.endswith("_NEG"):
            base = token[:-4]
            return stemmer.stem(base) + "_NEG"
        return stemmer.stem(token)

    def _get_stopwords(self) -> frozenset:
        """Carrega stopwords do NLTK (lazy loading).

        Returns:
            Conjunto de stopwords em português.
        """
        if self._stopwords is None:
            try:
                import nltk
                try:
                    from nltk.corpus import stopwords
                    self._stopwords = frozenset(
                        stopwords.words("portuguese")
                    )
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                    from nltk.corpus import stopwords
                    self._stopwords = frozenset(
                        stopwords.words("portuguese")
                    )
            except ImportError:
                logger.warning(
                    "NLTK não instalado. Stopwords não serão removidas."
                )
                self._stopwords = frozenset()
        return self._stopwords

    def _get_stemmer(self) -> Any:
        """Carrega o RSLP Stemmer do NLTK (lazy loading).

        Returns:
            Instância de ``nltk.stem.RSLPStemmer``.
        """
        if self._stemmer is None:
            try:
                import nltk
                try:
                    self._stemmer = nltk.stem.RSLPStemmer()
                except LookupError:
                    nltk.download("rslp", quiet=True)
                    self._stemmer = nltk.stem.RSLPStemmer()
            except ImportError:
                logger.warning(
                    "NLTK não instalado. Stemming não será aplicado."
                )
                # Retorna stemmer dummy
                class DummyStemmer:
                    def stem(self, word: str) -> str:
                        return word
                self._stemmer = DummyStemmer()
        return self._stemmer
