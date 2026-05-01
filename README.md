<div align="center">

# TextPipe BR

**Pipeline dinâmico e modular para classificação de texto em Português Brasileiro**

Teste dezenas de combinações de pré-processamento, vetorização e modelos de ML em um único comando.
Descubra automaticamente a melhor configuração para o seu dataset.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## Por que usar este projeto?

A maioria dos tutoriais de NLP em português ensina a testar **uma única combinação** de técnicas (ex: TF-IDF + Logistic Regression). Mas como saber se essa é realmente a melhor opção para o seu dataset?

O **TextPipe BR** resolve esse problema. Ele testa **todas as combinações possíveis** entre diferentes formas de limpar, representar e classificar textos, e entrega um ranking objetivo mostrando qual configuração realmente funciona melhor.

**Em números:** com 3 configs de pré-processamento, 5 de vetorização e 7 classificadores, o pipeline executa **105 experimentos automaticamente** e entrega o resultado ordenado por F1-macro.

### O que torna este projeto diferente

| Característica | Outros projetos | TextPipe BR |
|:---|:---|:---|
| Linguagem alvo | Inglês | **Português Brasileiro** |
| Abordagem | Pipeline único e fixo | **Produto cartesiano de configs** |
| Interface | Apenas terminal | **Terminal + Dashboard interativo** |
| Hiperparâmetros | Fixos | **Manual ou busca automática** |
| Compatibilidade | Código solto | **API scikit-learn nativa** |
| Reprodutibilidade | Variável | **`random_state=42` em todas as etapas** |

---

## Sumário

1. [Início Rápido](#1-início-rápido)
2. [Arquitetura](#2-arquitetura)
3. [Módulos do Pipeline](#3-módulos-do-pipeline)
4. [Dashboard Interativo](#4-dashboard-interativo)
5. [Justificativa das Escolhas Técnicas](#5-justificativa-das-escolhas-técnicas)
6. [Datasets Suportados](#6-datasets-suportados)
7. [Resultados e Relatórios](#7-resultados-e-relatórios)
8. [Uso como Biblioteca](#8-uso-como-biblioteca)
9. [Configuração Avançada](#9-configuração-avançada)

---

## 1. Início Rápido

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/textpipe-br.git
cd textpipe-br

# Crie um ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Instale as dependências
pip install -r requirements.txt
```

### Executar via Terminal

```bash
# Experimento completo (todas as combinações)
python main.py

# Modo rápido (poucas combinações, ideal para testar)
py main.py --quick

# Com outro dataset
python main.py --dataset toldbr
```

### Executar o Dashboard

```bash
streamlit run dashboard.py
```

O navegador abre automaticamente em `http://localhost:8501` com a interface completa.

### Argumentos do Terminal

| Argumento         | Default    | Descrição                              |
|:---|:---|:---|
| `--dataset`       | `hatebr`   | Dataset: `hatebr`, `toldbr`, `b2w`     |
| `--data-path`     | (auto)     | Caminho para o CSV do dataset          |
| `--results-dir`   | `results`  | Diretório para salvar resultados       |
| `--random-state`  | `42`       | Semente de reprodutibilidade           |
| `--quick`         | `false`    | Modo rápido (configs reduzidas)        |

---

## 2. Arquitetura

```
textpipe-br/
├── sentiment_pipeline/           # Pacote Python principal
│   ├── __init__.py               # Exporta classes públicas
│   ├── data_loader.py            # Carregamento e split de datasets
│   ├── preprocessor.py           # Pré-processamento configurável
│   ├── vectorizer.py             # Vetorização (BoW / TF-IDF / Word2Vec)
│   ├── classifier.py             # Classificadores (manual / auto-tuning)
│   └── pipeline_manager.py       # Orquestrador de experimentos
├── configs/
│   └── experiment_config.yaml    # Template de configuração YAML
├── results/                      # Resultados gerados automaticamente
├── main.py                       # Execução via terminal
├── dashboard.py                  # Interface Streamlit
├── requirements.txt              # Dependências
├── hatebr.csv                    # Dataset HateBR
└── told_br_multilabel.csv        # Dataset ToLD-BR
```

### Fluxo de Dados

O pipeline segue uma arquitetura linear com ramificação por configuração:

```
  Dataset (CSV)
       │
       ▼
  DatasetLoader ──► DataFrame padronizado (text, label)
       │
       ▼
  split_data() ──► Treino 70% │ Validação 15% │ Teste 15%  (estratificado)
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │         PRODUTO CARTESIANO DE CONFIGS          │
  │                                                │
  │  Para cada combinação:                         │
  │    1. TextPreprocessor  ──► Texto limpo        │
  │    2. TextVectorizer    ──► Matriz numérica    │
  │    3. TextClassifier    ──► Modelo treinado    │
  │    4. Evaluate          ──► Métricas (F1, Acc) │
  └────────────────────────────────────────────────┘
       │
       ▼
  PipelineManager ──► Ranking CSV + Relatório Markdown
```

---

## 3. Módulos do Pipeline

### 3.1. DatasetLoader — Carregamento de Dados

Responsável por carregar datasets em formatos diversos e padronizá-los em um DataFrame com duas colunas: `text` e `label` (0 ou 1).

```python
from sentiment_pipeline import DatasetLoader

loader = DatasetLoader(random_state=42)
df = loader.load_and_standardize("hatebr.csv", "hatebr")
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df)
```

**Loaders especializados:** Cada dataset possui tratamento próprio porque os formatos brutos são diferentes. O ToLD-BR, por exemplo, possui 6 colunas de anotação que precisam ser binarizadas; o B2W precisa converter ratings 1-5 em classes binárias. O `load_generic()` serve como alternativa para qualquer CSV arbitrário.

**Split estratificado:** A divisão 70/15/15 utiliza `stratify=y` para garantir que a proporção das classes seja mantida em cada partição, evitando resultados enviesados em datasets desbalanceados.

---

### 3.2. TextPreprocessor — Pré-processamento Configurável

Cada etapa de limpeza é controlada individualmente por parâmetros booleanos. O usuário pode ligar ou desligar qualquer combinação:

```python
from sentiment_pipeline import TextPreprocessor

pp = TextPreprocessor(
    stemming=True,
    handle_negations=True,
    handle_emojis="demojize",
)
texts_clean = pp.fit_transform(X_train)
```

| Parâmetro | Default | O que faz |
|:---|:---|:---|
| `lowercase` | `True` | Converte para minúsculas |
| `remove_urls` | `True` | Remove links http/https/www |
| `remove_mentions` | `True` | Remove @usuário e #hashtags |
| `remove_punctuation` | `True` | Remove sinais de pontuação |
| `remove_numbers` | `True` | Remove dígitos numéricos |
| `remove_stopwords` | `True` | Remove palavras comuns ("de", "o", "que") via NLTK |
| `keep_negations` | `True` | Preserva "não", "nunca", "jamais" mesmo ao remover stopwords |
| `handle_negations` | `False` | Marca tokens após negação com sufixo `_NEG` |
| `stemming` | `False` | Reduz palavras ao radical via RSLP Stemmer |
| `handle_emojis` | `"remove"` | `"remove"`, `"demojize"` (texto em PT), `"keep"` |
| `min_token_length` | `2` | Descarta tokens com menos de N caracteres |

**Por que a ordem das etapas importa:** Emojis são tratados primeiro porque podem ser perdidos ao converter para minúsculas. A pontuação é removida depois do tratamento de negações, pois os pontos finais servem como delimitadores para saber onde a negação termina.

---

### 3.3. TextVectorizer — Vetorização

Transforma textos limpos em representações numéricas (matrizes) que os classificadores conseguem processar:

```python
from sentiment_pipeline import TextVectorizer

vec = TextVectorizer(method="tfidf", svd_components=100)
X_vec = vec.fit_transform(texts_clean)
```

| Método | Descrição | Quando usar |
|:---|:---|:---|
| **BoW** (Bag-of-Words) | Conta a frequência de cada palavra | Baseline rápido |
| **TF-IDF** | Pondera pela raridade da palavra no corpus | Melhor que BoW na maioria dos casos |
| **TF-IDF + SVD** | Reduz dimensionalidade após TF-IDF | Datasets grandes, reduz ruído e memória |
| **Word2Vec** | Média dos vetores semânticos de cada palavra | Captura significado, requer gensim |

**SVD vs Word2Vec:** O SVD é uma técnica de álgebra linear que comprime a matriz TF-IDF (ex: de 10.000 colunas para 100), mantendo apenas os padrões mais importantes. O Word2Vec é uma rede neural que aprende relações semânticas entre palavras a partir do contexto. São abordagens complementares: o SVD trabalha sobre frequências, o Word2Vec sobre significado.

---

### 3.4. TextClassifier — Classificação

Treina e avalia modelos de machine learning com dois modos de operação:

```python
from sentiment_pipeline import TextClassifier

clf = TextClassifier("logistic_regression", mode="auto", n_iter=30)
clf.fit(X_train_vec, y_train)
results = clf.evaluate(X_test_vec, y_test)
```

| Modelo | Tipo | Por que está incluído |
|:---|:---|:---|
| **Naive Bayes** | Probabilístico | Baseline clássico para texto, extremamente rápido |
| **Logistic Regression** | Linear | Equilíbrio entre velocidade e performance |
| **Linear SVC** | SVM Linear | Excelente para dados de alta dimensionalidade |
| **Random Forest** | Ensemble | Robusto a overfitting, não assume linearidade |
| **LightGBM** | Gradient Boosting | Estado da arte em dados tabulares |

**Modo Manual vs Automático:**
- **Manual:** Usa hiperparâmetros fixos. Rápido e determinístico. Ideal para comparações rápidas.
- **Automático:** Usa `RandomizedSearchCV` com cross-validation (cv=3) para encontrar os melhores hiperparâmetros. Mais lento, mas pode melhorar significativamente o resultado.

**F1 de validação:** É calculado separadamente para detectar **overfitting**. Se o F1 no treino for muito superior ao F1 de validação, o modelo está "decorando" em vez de aprendendo.

---

### 3.5. PipelineManager — Orquestrador

Coordena todo o experimento: calcula o produto cartesiano das configurações, executa cada combinação, registra métricas e gera relatórios.

```python
from sentiment_pipeline import PipelineManager

manager = PipelineManager("meu_experimento")
results_df = manager.run_experiment(
    X_train, X_val, X_test, y_train, y_val, y_test,
    preprocess_configs=[...],
    vectorizer_configs=[...],
    classifier_configs=[...],
)
```

**Tolerância a falhas:** Se uma combinação específica falha (ex: Naive Bayes recebe valores negativos de SVD), o Manager registra o erro e continua com a próxima combinação sem interromper o experimento inteiro.

**Tratamentos automáticos:**
- Naive Bayes recebe `np.abs()` nos dados (não aceita valores negativos)
- LightGBM recebe conversão para `float32` (BoW produz inteiros)

---

## 4. Dashboard Interativo

O `dashboard.py` oferece uma interface gráfica completa via Streamlit, permitindo executar experimentos sem escrever código.

### Como executar

```bash
# Opção padrão
streamlit run dashboard.py

# Alternativa recomendada (especialmente no Windows, se der erro de comando não encontrado)
python -m streamlit run dashboard.py
```

### Funcionalidades

**Sidebar (Barra Lateral) — Configuração:**
- Seleção de dataset (HateBR, ToLD-BR, B2W) ou upload de qualquer CSV
- Checkboxes individuais para cada etapa de pré-processamento
- Seleção de métodos de vetorização e parâmetros (n-grams, SVD)
- Seleção de classificadores e modo (manual/automático)
- Slider para Top N (quantas melhores combinações exibir)

**Área Principal — Resultados:**
- Tabela de ranking colorida das melhores combinações (ordenada por F1-macro)
- Gráficos comparativos de métricas
- Justificativa experimental automática: análise textual gerada pelo sistema explicando por que a melhor configuração venceu
- Linha de configuração final pronta para reprodução

**Upload de dataset personalizado:**
O usuário pode fazer upload de qualquer CSV e indicar o nome da coluna de texto e da coluna de label. Internamente, o dashboard usa a função `load_generic()` do DatasetLoader.

---

## 5. Justificativa das Escolhas Técnicas

### Por que scikit-learn como base?

O scikit-learn oferece uma API padronizada (`fit` / `transform` / `predict`) que permite trocar qualquer componente do pipeline sem alterar o restante do código. Todas as classes do projeto herdam de `BaseEstimator` e `TransformerMixin`, garantindo compatibilidade nativa com ferramentas do ecossistema (GridSearch, Pipeline, cross-validation).

### Por que F1-macro como métrica principal?

Datasets de classificação de texto em português frequentemente são **desbalanceados** (ex: muito mais textos "não ofensivos" do que "ofensivos"). A acurácia sozinha pode ser enganosa: um modelo que sempre diz "não ofensivo" teria 90% de acurácia em um dataset 90/10, mas seria inútil. O F1-macro trata todas as classes com peso igual, penalizando modelos que ignoram a classe minoritária.

### Por que split 70/15/15 com estratificação?

- **70% treino:** Volume suficiente para o modelo aprender padrões.
- **15% validação:** Permite detectar overfitting comparando F1 de treino vs validação.
- **15% teste:** Avaliação final em dados completamente inéditos.
- **Estratificação:** Garante que cada partição mantenha a proporção original das classes.

### Por que produto cartesiano em vez de busca aleatória?

Para o escopo deste projeto (dezenas a centenas de combinações), o produto cartesiano é viável e garante **cobertura completa**. Nenhuma combinação é ignorada. Em cenários com milhares de combinações, uma Random Search de alto nível seria mais eficiente.

### Por que `random_state=42` em todo lugar?

Reprodutibilidade. Qualquer pessoa que execute o mesmo experimento com os mesmos dados obterá **exatamente os mesmos resultados**. Isso é fundamental em contexto acadêmico e científico.

### Por que LightGBM além dos modelos clássicos?

Os modelos clássicos (Naive Bayes, Logistic Regression, SVC) são ótimos baselines para texto. O LightGBM representa o estado da arte em dados tabulares e frequentemente supera modelos lineares em datasets maiores, mas com custo computacional maior. Incluir ambos permite uma comparação justa.

### Por que RSLP Stemmer em vez de Lemmatization?

O RSLP é o stemmer mais estabelecido para português brasileiro e está disponível no NLTK sem dependências externas. Lemmatizadores para português (como spaCy pt) exigem modelos adicionais e são significativamente mais lentos, sem ganho proporcional em tarefas de classificação binária.

---

## 6. Datasets Suportados

| Dataset | Tarefa | Amostras | Binarização |
|:---|:---|:---|:---|
| **HateBR** | Discurso de ódio | ~7.000 | Direto (0/1) |
| **ToLD-BR** | Toxicidade | ~28.000 | Qualquer anotação > 0 = tóxico |
| **B2W-Reviews** | Sentimento | ~130.000 | Rating 1-2 = neg, 4-5 = pos, 3 descartado |
| **Personalizado** | Qualquer | Variável | Upload via dashboard (indicar colunas) |

---

## 7. Resultados e Relatórios

Cada execução gera dois arquivos na pasta `results/`:

**Arquivo CSV** — para análise programática:
- `run_id`: Identificador da combinação
- `preprocessor`, `vectorizer`, `classifier`: Configurações utilizadas
- `f1_macro`, `accuracy`, `precision_macro`, `recall_macro`: Métricas de teste
- `val_f1_macro`: Métrica de validação (para detectar overfitting)
- `train_time_s`: Tempo de treino em segundos
- `best_params`: Hiperparâmetros otimizados (modo auto)

**Relatório Markdown** — para leitura humana:
- Top 10 melhores combinações em tabela formatada
- Estatísticas gerais (melhor, pior, média, desvio padrão do F1)
- Destaque do vencedor com configurações detalhadas
- Seção de erros (se houve falhas)

---

## 8. Uso como Biblioteca

O pacote `sentiment_pipeline` pode ser importado em qualquer projeto Python:

```python
from sentiment_pipeline import (
    DatasetLoader,
    TextPreprocessor,
    TextVectorizer,
    TextClassifier,
)

# 1. Carregar dados
loader = DatasetLoader()
df = loader.load_and_standardize("meu_dataset.csv", "hatebr")

# 2. Pré-processar
pp = TextPreprocessor(stemming=True, handle_negations=True)
texts = pp.fit_transform(df["text"])

# 3. Vetorizar
vec = TextVectorizer(method="tfidf", svd_components=100)
X = vec.fit_transform(texts)

# 4. Classificar
clf = TextClassifier("lightgbm", mode="auto")
clf.fit(X, df["label"])
results = clf.evaluate(X_test, y_test)
print(f"F1-macro: {results['f1_macro']:.4f}")
```

---

## 9. Configuração Avançada

### Via Python (main.py)

```python
PREPROCESS_CONFIGS = [
    {"stemming": False, "handle_negations": False},
    {"stemming": True, "handle_negations": True, "handle_emojis": "demojize"},
]

VECTORIZER_CONFIGS = [
    {"method": "tfidf", "ngram_range": (1, 2)},
    {"method": "tfidf", "svd_components": 100},
    {"method": "tfidf", "svd_components": 300},
]

CLASSIFIER_CONFIGS = [
    {"model_name": "logistic_regression", "mode": "manual"},
    {"model_name": "logistic_regression", "mode": "auto"},
    {"model_name": "lightgbm", "mode": "auto"},
]
```

### Via YAML (template)

O arquivo `configs/experiment_config.yaml` contém um template completo com todas as opções disponíveis. Ele não é lido diretamente pelo código atual, mas serve como referência e pode ser integrado para carregamento dinâmico.

### Via Dashboard (interativo)

Na interface Streamlit, todas as configurações são definidas pela sidebar usando checkboxes, dropdowns e sliders, sem necessidade de editar código.
