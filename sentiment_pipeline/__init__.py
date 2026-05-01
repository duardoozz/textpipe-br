"""
sentiment_pipeline — Pacote modular para classificação de texto (PT-BR).

Oferece componentes reutilizáveis e compatíveis com scikit-learn para
pré-processamento, vetorização e modelagem de texto, permitindo a
experimentação dinâmica de diferentes estratégias de pipeline.
"""

from sentiment_pipeline.data_loader import DatasetLoader
from sentiment_pipeline.preprocessor import TextPreprocessor
from sentiment_pipeline.vectorizer import TextVectorizer
from sentiment_pipeline.classifier import TextClassifier
from sentiment_pipeline.pipeline_manager import PipelineManager

__all__ = [
    "DatasetLoader",
    "TextPreprocessor",
    "TextVectorizer",
    "TextClassifier",
    "PipelineManager",
]

__version__ = "1.0.0"
