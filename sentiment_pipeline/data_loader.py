"""Módulo de carregamento e preparação de datasets.

Fornece uma interface unificada para carregar, padronizar e dividir
datasets de classificação de texto em Português Brasileiro.

Datasets:
    - HateBR: detecção binária de discurso de ódio.
    - ToLD-BR: detecção multilabel de toxicidade (binarizado).
    - B2W-Reviews01: reviews de e-commerce (ratings binarizados).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Carrega, padroniza e divide datasets de texto PT-BR.

    Todos os datasets são convertidos para um ``DataFrame`` com duas
    colunas padronizadas: ``text`` (string) e ``label`` (int 0/1).

    Attributes:
        random_state: Semente para reprodutibilidade dos splits.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # Carregamento padronizado
    def load_and_standardize(
        self,
        path: str,
        dataset_type: str,
    ) -> pd.DataFrame:

        # Dicionário de loaders
        loaders = {
            "hatebr": self._load_hatebr,
            "toldbr": self._load_toldbr,
            "b2w": self._load_b2w,
        }

        # Verifica se o dataset type é válido
        if dataset_type not in loaders:
            raise ValueError(
                f"dataset_type '{dataset_type}' não reconhecido. "
                f"Valores aceitos: {list(loaders.keys())}"
            )
        
        # Carrega o dataset
        df = loaders[dataset_type](path)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        logger.info(
            "Dataset '%s' carregado: %d amostras | Distribuição: %s",
            dataset_type,
            len(df),
            df["label"].value_counts().to_dict(),
        )
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        text_col: str = "text",
        label_col: str = "label",
    ) -> Tuple[
        pd.Series, pd.Series, pd.Series,
        pd.Series, pd.Series, pd.Series,
    ]:
        """Divide os dados em treino/validação/teste com estratificação.

        Realiza um split 70/15/15 por padrão, estratificado pela coluna
        de labels para manter a proporção das classes em todas as partições.

        Args:
            df: DataFrame com pelo menos as colunas ``text_col`` e
                ``label_col``.
            test_size: Proporção do conjunto de teste (default 0.15).
            val_size: Proporção do conjunto de validação (default 0.15).
            text_col: Nome da coluna de texto.
            label_col: Nome da coluna de label.

        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        X = df[text_col]
        y = df[label_col]

        # Primeiro split: separa teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Segundo split: separa validação do restante
        # Ajusta proporção relativa: val_size / (1 - test_size)
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val_size,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(
            "Split realizado — Treino: %d | Validação: %d | Teste: %d",
            len(X_train), len(X_val), len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Carregamento genérico (qualquer dataset)
    def load_generic(
        self,
        path: str,
        text_col: str,
        label_col: str,
        sep: str = ",",
    ) -> pd.DataFrame:
        """Carrega qualquer dataset CSV com colunas de texto e label.

        Permite utilizar o pipeline com datasets arbitrários,
        bastando indicar qual coluna contém o texto e qual
        contém os labels.

        Args:
            path: Caminho para o arquivo CSV.
            text_col: Nome da coluna de texto.
            label_col: Nome da coluna de label.
            sep: Separador do CSV (default: ``","``).

        Returns:
            DataFrame com colunas ``text`` (str) e ``label`` (int).
        """
        df = pd.read_csv(path, sep=sep)

        if text_col not in df.columns:
            raise ValueError(
                f"Coluna de texto '{text_col}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            )
        if label_col not in df.columns:
            raise ValueError(
                f"Coluna de label '{label_col}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        result = pd.DataFrame({
            "text": df[text_col].astype(str),
            "label": df[label_col],
        })
        result = result.dropna(subset=["text", "label"]).reset_index(drop=True)
        result["label"] = result["label"].astype(int)

        logger.info(
            "Dataset genérico carregado: %d amostras | Distribuição: %s",
            len(result),
            result["label"].value_counts().to_dict(),
        )
        return result

    # Loaders internos
    def _load_hatebr(self, path: str) -> pd.DataFrame:
        """Carrega o dataset HateBR.

        O HateBR possui anotações de 3 anotadores e um ``label_final``
        binário (0 = não-ofensivo, 1 = ofensivo).

        Args:
            path: Caminho para ``hatebr.csv``.

        Returns:
            DataFrame padronizado com colunas ``text`` e ``label``.
        """
        df = pd.read_csv(path)
        return pd.DataFrame({
            "text": df["comentario"].astype(str),
            "label": df["label_final"].astype(int),
        })

    def _load_toldbr(self, path: str) -> pd.DataFrame:
        """Carrega e binariza o dataset ToLD-BR.

        Binarização: se *qualquer* coluna de anotação (homophobia,
        obscene, insult, racism, misogyny, xenophobia) for > 0,
        o texto é classificado como tóxico (1); caso contrário, 0.

        Args:
            path: Caminho para ``told_br_multilabel.csv``.

        Returns:
            DataFrame padronizado com colunas ``text`` e ``label``.
        """
        df = pd.read_csv(path)
        annotation_cols = [
            "homophobia", "obscene", "insult",
            "racism", "misogyny", "xenophobia",
        ]
        toxic_mask = df[annotation_cols].sum(axis=1) > 0
        return pd.DataFrame({
            "text": df["text"].astype(str),
            "label": toxic_mask.astype(int),
        })

    def _load_b2w(self, path: str) -> pd.DataFrame:
        """Carrega e binariza o dataset B2W-Reviews01.

        Binarização de ratings:
            - 1, 2 → negativo (0)
            - 4, 5 → positivo (1)
            - 3 → descartado (neutro)

        Se o arquivo não existir localmente, tenta baixá-lo do GitHub.

        Args:
            path: Caminho para ``B2W-Reviews01.csv`` ou diretório onde
                salvá-lo.

        Returns:
            DataFrame padronizado com colunas ``text`` e ``label``.
        """
        csv_path = Path(path)

        # Se 'path' é diretório, assume nome padrão do arquivo
        if csv_path.is_dir():
            csv_path = csv_path / "B2W-Reviews01.csv"

        # Tenta download se não existir
        if not csv_path.exists():
            csv_path = self._download_b2w(csv_path)

        df = pd.read_csv(csv_path, sep=";", quotechar='"')

        # Filtra colunas necessárias e remove nulos
        df = df[["review_text", "overall_rating"]].dropna()

        # Binariza: descarta rating 3
        df = df[df["overall_rating"] != 3].copy()
        df["label"] = (df["overall_rating"] >= 4).astype(int)

        return pd.DataFrame({
            "text": df["review_text"].astype(str),
            "label": df["label"],
        })

    @staticmethod
    def _download_b2w(target_path: Path) -> Path:
        """Tenta baixar o dataset B2W-Reviews01 do GitHub.

        Args:
            target_path: Caminho de destino para salvar o CSV.

        Returns:
            Path do arquivo baixado.

        Raises:
            FileNotFoundError: Se o download falhar.
        """
        url = (
            "https://raw.githubusercontent.com/"
            "b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv"
        )
        logger.info("Baixando B2W-Reviews01 de %s ...", url)

        try:
            import urllib.request

            target_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(target_path))
            logger.info("Download concluído: %s", target_path)
            return target_path

        except Exception as exc:
            raise FileNotFoundError(
                f"Não foi possível baixar o B2W-Reviews01: {exc}\n"
                f"Por favor, baixe manualmente de:\n"
                f"  https://github.com/b2wdigital/b2w-reviews01\n"
                f"e coloque o CSV em: {target_path}"
            ) from exc
