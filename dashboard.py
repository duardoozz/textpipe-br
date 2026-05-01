"""
Dashboard interativo para o Pipeline Dinâmico de Classificação de Texto.

Permite ao usuário configurar pré-processamento, vetorização e classificadores
pela sidebar, executar todas as combinações e visualizar os resultados
com métricas, gráficos e justificativa experimental automática.

Execute com:
    streamlit run dashboard.py
"""

from __future__ import annotations

import io
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sentiment_pipeline.data_loader import DatasetLoader
from sentiment_pipeline.preprocessor import TextPreprocessor
from sentiment_pipeline.vectorizer import TextVectorizer
from sentiment_pipeline.classifier import TextClassifier
from sentiment_pipeline.pipeline_manager import PipelineManager

warnings.filterwarnings("ignore")

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Pipeline Dinâmico - Classificação de Texto",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para estilização dos componentes visuais
# Utilizando uma paleta de cores padrão, profissional e moderna
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }
    .metric-card {
        background: #ffffff;
        padding: 1.2rem; border-radius: 10px; text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-card h3 {
        margin: 0; font-size: 0.9rem; color: #64748b;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1e293b; margin: 0.3rem 0; }
    .config-line {
        background: #1e293b; color: #38bdf8; padding: 1rem 1.5rem;
        border-radius: 8px; font-family: 'Courier New', monospace;
        font-size: 0.85rem; overflow-x: auto; border-left: 4px solid #38bdf8;
    }
    .justification {
        background: #f8fafc; border-left: 4px solid #3b82f6;
        padding: 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
        font-size: 1.05rem; line-height: 1.6; color: #334155;
    }
    .section-title { border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; margin-bottom: 1.5rem; color: #0f172a; }
    .html-table-container {
        overflow-x: auto;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .html-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .html-table-container th {
        background-color: #f1f5f9;
        text-align: left;
        padding: 12px;
        color: #475569;
        font-weight: 600;
        border-bottom: 2px solid #cbd5e1;
    }
    .html-table-container td {
        padding: 10px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #334155;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    """Função principal do dashboard Streamlit.

    Estrutura:
      - Sidebar: configuração de dataset, pré-processamento, vetorização,
        classificadores e parâmetros gerais.
      - Área principal: exibe a config atual, executa o experimento
        e mostra resultados, gráficos e justificativa.
    """

    # Cabeçalho visual
    st.markdown("""
    <div class="main-header">
        <h1>Pipeline Dinâmico de Classificação de Texto</h1>
        <p>Experimentação interativa com pré-processamento, vetorização e modelagem (PT-BR)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar com todos os controles de configuração do experimento
    with st.sidebar:
        st.markdown("## Configuração do Experimento")

        # Seção: seleção do dataset (local ou upload)
        st.markdown("### Dataset")
        dataset_source = st.radio(
            "Fonte do Dataset",
            ["Dataset local (pré-definido)", "Upload de CSV"],
            help="Escolha um dataset local ou faça upload do seu próprio CSV.",
        )

        df = None
        if dataset_source == "Dataset local (pré-definido)":
            dataset_type = st.selectbox(
                "Dataset", ["hatebr", "toldbr"],
                help="HateBR (~7K) ou ToLD-BR (~28K).",
            )
            paths = {"hatebr": "hatebr.csv", "toldbr": "told_br_multilabel.csv"}
            data_path = paths[dataset_type]
        else:
            uploaded = st.file_uploader(
                "Upload CSV", type=["csv"],
                help="Faça upload de qualquer CSV com colunas de texto e label.",
            )
            if uploaded is not None:
                # Salva o arquivo temporariamente para leitura posterior
                temp_path = Path("data") / uploaded.name
                temp_path.parent.mkdir(exist_ok=True)
                temp_path.write_bytes(uploaded.read())
                data_path = str(temp_path)

                # Mostra colunas detectadas para que o usuário mapeie texto/label
                preview = pd.read_csv(data_path, nrows=5)
                st.markdown("**Colunas detectadas:**")
                st.code(", ".join(preview.columns.tolist()))

                text_col = st.selectbox("Coluna de Texto", preview.columns.tolist())
                label_col = st.selectbox(
                    "Coluna de Label",
                    [c for c in preview.columns if c != text_col],
                )
                sep = st.text_input("Separador CSV", value=",")
            else:
                data_path = None

        st.markdown("---")

        # Seção: opções de pré-processamento de texto
        st.markdown("### Pré-processamento")
        st.caption("Configure as opções de limpeza de texto.")

        pp_lowercase = st.checkbox("Lowercase", value=True)
        pp_remove_urls = st.checkbox("Remover URLs", value=True)
        pp_remove_mentions = st.checkbox("Remover @menções/#hashtags", value=True)
        pp_remove_punct = st.checkbox("Remover pontuação", value=True)
        pp_remove_numbers = st.checkbox("Remover números", value=True)
        pp_remove_stopwords = st.checkbox("Remover stopwords", value=True)
        pp_keep_negations = st.checkbox(
            "Preservar negações", value=True,
            help="Mantém 'não', 'nunca', 'jamais' mesmo ao remover stopwords.",
        )
        pp_handle_negations = st.checkbox(
            "Handle negações (_NEG)", value=False,
            help="Adiciona sufixo _NEG a tokens após palavras de negação.",
        )
        pp_stemming = st.checkbox(
            "Stemming (RSLP)", value=False,
            help="Aplica o stemmer RSLP para português.",
        )
        pp_emojis = st.selectbox(
            "Tratamento de Emojis",
            ["remove", "demojize", "keep"],
            help="remove: apaga | demojize: converte para texto | keep: mantém",
        )
        pp_min_token = st.slider("Comprimento mínimo de token", 1, 5, 2)

        # Opção para adicionar automaticamente duas variações extras
        # (com stemming e mínima) além da configuração manual definida acima
        pp_add_variations = st.checkbox(
            "Adicionar variações de pré-processamento", value=True,
            help="Testa também uma config com stemming e outra mínima.",
        )

        st.markdown("---")

        # Seção: seleção dos métodos de vetorização
        st.markdown("### Vetorização")
        vec_methods = st.multiselect(
            "Métodos",
            ["bow", "tfidf", "tfidf+svd50", "tfidf+svd100", "tfidf+svd300"],
            default=["bow", "tfidf", "tfidf+svd100", "tfidf+svd300"],
            help="Selecione os métodos de vetorização a testar.",
        )
        vec_max_features = st.number_input("Max features", 1000, 50000, 10000, step=1000)
        vec_ngram = st.selectbox(
            "N-gram range", [(1, 1), (1, 2), (1, 3)],
            index=1, format_func=lambda x: f"({x[0]}, {x[1]})",
        )

        st.markdown("---")

        # Seção: seleção dos classificadores e modo de tuning
        st.markdown("### Classificadores")
        clf_models = st.multiselect(
            "Modelos",
            ["naive_bayes", "logistic_regression", "linear_svc", "random_forest", "lightgbm"],
            default=["naive_bayes", "logistic_regression", "linear_svc"],
        )
        clf_mode = st.radio(
            "Modo de Tuning", ["manual", "auto", "ambos"],
            help="manual: defaults | auto: RandomizedSearchCV | ambos: testa os dois",
        )

        st.markdown("---")

        # Seção: parâmetros gerais do experimento
        st.markdown("### Parâmetros Gerais")
        top_n = st.slider(
            "Top N combinações para comparar", 5, 50, 20,
            help="Número de melhores combinações a exibir nos resultados.",
        )
        random_state = st.number_input("Random State", 0, 9999, 42)
        test_size = st.slider("Tamanho do teste (%)", 5, 40, 15) / 100
        val_size = st.slider("Tamanho da validação (%)", 5, 40, 15) / 100

        st.markdown("---")
        run_button = st.button("Executar Experimento", type="primary", use_container_width=True)

    # Área principal: mostra a configuração atual como JSON formatado
    st.markdown("### Configuração Atual")

    # Monta o dicionário de pré-processamento a partir dos checkboxes
    pp_config_manual = {
        "lowercase": pp_lowercase, "remove_urls": pp_remove_urls,
        "remove_mentions": pp_remove_mentions, "remove_punctuation": pp_remove_punct,
        "remove_numbers": pp_remove_numbers, "remove_stopwords": pp_remove_stopwords,
        "keep_negations": pp_keep_negations, "handle_negations": pp_handle_negations,
        "stemming": pp_stemming, "handle_emojis": pp_emojis,
        "min_token_length": pp_min_token,
    }

    config_str = str(pp_config_manual).replace("'", '"')
    st.markdown(f'<div class="config-line">{config_str}</div>', unsafe_allow_html=True)

    # Execução do experimento quando o botão é clicado
    if run_button:
        if data_path is None:
            st.error("Faça upload de um dataset CSV para continuar.")
            return

        # Aviso sobre tempo de execução
        st.info("⏳ **Atenção:** Este processo pode demorar alguns minutos, dependendo do tamanho do dataset e da quantidade de combinações selecionadas. Por favor, aguarde a conclusão do progresso abaixo.")

        # Carrega e padroniza o dataset
        with st.spinner("Carregando dataset..."):
            loader = DatasetLoader(random_state=random_state)
            try:
                if dataset_source == "Dataset local (pré-definido)":
                    df = loader.load_and_standardize(data_path, dataset_type)
                else:
                    df = loader.load_generic(data_path, text_col, label_col, sep)
            except Exception as e:
                st.error(f"Erro ao carregar dataset: {e}")
                return

        # Exibe métricas resumidas do dataset carregado
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total de Amostras</h3>
                <div class="value">{len(df):,}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            n_classes = df["label"].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Classes</h3>
                <div class="value">{n_classes}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            balance = df["label"].value_counts(normalize=True).max() * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Classe Majoritária</h3>
                <div class="value">{balance:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Divisão estratificada dos dados
        with st.spinner("Dividindo dados (estratificado)..."):
            X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
                df, test_size=test_size, val_size=val_size,
            )

        st.success(
            f"Split: **Treino** {len(X_train)} | **Validação** {len(X_val)} | **Teste** {len(X_test)}"
        )

        # Monta a lista de configs de pré-processamento.
        # Se o checkbox de variações está ativo, adiciona automaticamente
        # uma config com stemming+negação e uma config mínima.
        preprocess_configs = [pp_config_manual]
        if pp_add_variations:
            pp_stemming_cfg = {
                **pp_config_manual,
                "stemming": True, "handle_negations": True,
                "handle_emojis": "demojize",
            }
            pp_minimal_cfg = {
                "lowercase": True, "remove_urls": True, "remove_mentions": True,
                "remove_punctuation": False, "remove_numbers": False,
                "remove_stopwords": False, "keep_negations": False,
                "handle_negations": False, "stemming": False,
                "handle_emojis": "keep", "min_token_length": 1,
            }
            # Evita duplicatas caso a config manual seja idêntica a alguma variação
            for cfg in [pp_stemming_cfg, pp_minimal_cfg]:
                if cfg != pp_config_manual:
                    preprocess_configs.append(cfg)

        # Monta configs de vetorização a partir dos métodos selecionados
        vectorizer_configs = []
        for m in vec_methods:
            cfg: Dict[str, Any] = {
                "max_features": vec_max_features,
                "ngram_range": vec_ngram,
            }
            if m == "bow":
                cfg["method"] = "bow"
            elif m == "tfidf":
                cfg["method"] = "tfidf"
            elif m.startswith("tfidf+svd"):
                n_comp = int(m.split("svd")[1])
                cfg["method"] = "tfidf"
                cfg["svd_components"] = n_comp
            vectorizer_configs.append(cfg)

        # Monta configs de classificadores de acordo com o modo selecionado
        classifier_configs = []
        for model in clf_models:
            if clf_mode in ("manual", "ambos"):
                classifier_configs.append({"model_name": model, "mode": "manual"})
            if clf_mode in ("auto", "ambos"):
                classifier_configs.append({"model_name": model, "mode": "auto", "n_iter": 20})

        total = len(preprocess_configs) * len(vectorizer_configs) * len(classifier_configs)
        st.info(
            f"**{len(preprocess_configs)}** pré-processamento x "
            f"**{len(vectorizer_configs)}** vetorizadores x "
            f"**{len(classifier_configs)}** classificadores = **{total}** combinações no total."
        )

        # Executa o experimento completo
        progress = st.progress(0, text="Iniciando experimento...")
        manager = PipelineManager(
            experiment_name="dashboard_exp",
            results_dir="results",
            random_state=random_state,
        )

        results_df = manager.run_experiment(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            preprocess_configs=preprocess_configs,
            vectorizer_configs=vectorizer_configs,
            classifier_configs=classifier_configs,
        )
        progress.progress(100, text="Experimento concluído!")

        # Armazena resultados no session_state para persistir entre reruns
        st.session_state["results_df"] = results_df
        st.session_state["top_n"] = top_n

    # Seção de resultados: só aparece se já houver resultados calculados
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        top_n = st.session_state.get("top_n", 20)
        valid = results_df.dropna(subset=["f1_macro"])

        st.markdown("---")
        st.markdown('<h2 class="section-title">Resultados do Experimento</h2>', unsafe_allow_html=True)

        # Métricas resumo: melhor, pior, média e desvio padrão do F1-macro
        if not valid.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Melhor F1-macro", f"{valid['f1_macro'].max():.4f}")
            with col2:
                st.metric("Pior F1-macro", f"{valid['f1_macro'].min():.4f}")
            with col3:
                st.metric("Média F1-macro", f"{valid['f1_macro'].mean():.4f}")
            with col4:
                st.metric("Desvio Padrão", f"{valid['f1_macro'].std():.4f}")

        # Tabela com as top N combinações ordenadas por F1-macro
        st.markdown(f"### Top {top_n} Combinações")

        display_cols = [
            "run_id", "preprocessor", "vectorizer", "classifier",
            "f1_macro", "accuracy", "precision_macro", "recall_macro",
            "val_f1_macro", "train_time_s",
        ]
        available = [c for c in display_cols if c in valid.columns]
        top_df = valid.head(top_n)[available]

        # Renderizando a tabela usando HTML puro do pandas para contornar a falta do pyarrow no Streamlit
        html_table = top_df.style.format({
            "f1_macro": "{:.4f}", "accuracy": "{:.4f}",
            "precision_macro": "{:.4f}", "recall_macro": "{:.4f}",
            "val_f1_macro": "{:.4f}", "train_time_s": "{:.2f}s",
        }).background_gradient(subset=["f1_macro"], cmap="Greens").to_html()
        
        st.markdown(f'<div class="html-table-container">{html_table}</div>', unsafe_allow_html=True)

        # Gráfico de barras comparando F1-macro das top combinações
        st.markdown("<br>### Comparação Visual (F1-macro)", unsafe_allow_html=True)
        chart_df = top_df[["run_id", "f1_macro", "classifier"]].copy()
        chart_df["label"] = chart_df["run_id"].astype(str) + " - " + chart_df["classifier"]
        chart_data = chart_df.set_index("label")["f1_macro"]
        st.bar_chart(chart_data, color="#3b82f6")

        # Detalhes da melhor configuração encontrada
        st.markdown("---")
        st.markdown('<h2 class="section-title">Melhor Configuração</h2>', unsafe_allow_html=True)

        if not valid.empty:
            best = valid.iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("F1-macro (teste)", f"{best['f1_macro']:.4f}")
                st.metric("Accuracy", f"{best.get('accuracy', 0):.4f}")
            with col2:
                st.metric("F1-macro (validação)", f"{best.get('val_f1_macro', 0):.4f}")
                st.metric("Tempo de treino", f"{best.get('train_time_s', 0):.2f}s")

            # Linha de configuração final: resume preprocessor+vectorizer+classifier
            st.markdown("**Linha de Configuração Final:**")
            config_final = {
                "preprocessor": best["preprocessor"],
                "vectorizer": best["vectorizer"],
                "classifier": best["classifier"],
            }
            config_str = str(config_final).replace("'", '"')
            st.markdown(f'<div class="config-line">{config_str}</div>', unsafe_allow_html=True)

            # Exibe hiperparâmetros otimizados (se modo auto foi usado)
            if "best_params" in best and best["best_params"]:
                st.markdown("**Melhores Hiperparâmetros:**")
                st.code(best["best_params"])

        # Justificativa experimental gerada automaticamente
        st.markdown("---")
        st.markdown('<h2 class="section-title">Justificativa Experimental</h2>', unsafe_allow_html=True)

        if not valid.empty:
            justification = _generate_justification(valid, top_n)
            st.markdown(f'<div class="justification">{justification}</div>', unsafe_allow_html=True)

        # Combinações que falharam (se houver)
        errors = results_df[results_df["f1_macro"].isna()]
        if not errors.empty:
            with st.expander(f"{len(errors)} combinações falharam"):
                error_html = errors[["run_id", "classifier", "vectorizer", "error"]].to_html(index=False)
                st.markdown(f'<div class="html-table-container">{error_html}</div>', unsafe_allow_html=True)

        # Botão de download dos resultados completos em CSV
        st.markdown("---")
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download resultados (CSV)",
            data=csv, file_name="experiment_results.csv",
            mime="text/csv", use_container_width=True,
        )

    # Estado inicial: instruções exibidas antes de qualquer execução
    elif not run_button:
        st.markdown("""
        ### Configure o experimento na barra lateral

        1. **Selecione ou faça upload** de um dataset CSV
        2. **Configure o pré-processamento** (negações, stemming, emojis...)
        3. **Escolha os vetorizadores** (BoW, TF-IDF, TF-IDF+SVD)
        4. **Selecione os classificadores** e modo de tuning
        5. **Clique em "Executar Experimento"**

        O pipeline irá testar **todas as combinações** e exibir as **top N melhores**
        com justificativa experimental automática.
        """)


def _generate_justification(valid: pd.DataFrame, top_n: int) -> str:
    """Gera uma justificativa experimental automática baseada nos resultados.

    Analisa os resultados por componente (vetorizador, classificador,
    pré-processamento) e verifica sinais de overfitting comparando
    F1 de teste vs validação.

    Args:
        valid: DataFrame com resultados válidos (sem NaN em f1_macro).
        top_n: Número de top combinações consideradas.

    Returns:
        Texto HTML com a justificativa estruturada.
    """
    best = valid.iloc[0]
    total_runs = len(valid)

    # Calcula média e máximo de F1 por componente do pipeline
    by_vec = valid.groupby("vectorizer")["f1_macro"].agg(["mean", "max", "count"])
    best_vec = by_vec["mean"].idxmax()

    by_clf = valid.groupby("classifier")["f1_macro"].agg(["mean", "max", "count"])
    best_clf = by_clf["mean"].idxmax()

    by_pp = valid.groupby("preprocessor")["f1_macro"].agg(["mean", "max", "count"])
    best_pp = by_pp["mean"].idxmax()

    # Separa runs com e sem SVD para comparação
    svd_runs = valid[valid["vectorizer"].str.contains("svd", case=False, na=False)]
    non_svd = valid[~valid["vectorizer"].str.contains("svd", case=False, na=False)]

    lines = []
    lines.append(f"<strong>Escopo:</strong> {total_runs} combinações foram testadas, "
                 f"avaliando as top {min(top_n, total_runs)} configurações.<br><br>")

    lines.append(f"<strong>1. Melhor vetorizador (média):</strong> <code>{best_vec}</code> - "
                 f"Média F1 = {by_vec.loc[best_vec, 'mean']:.4f}. ")

    if not svd_runs.empty and not non_svd.empty:
        svd_mean = svd_runs["f1_macro"].mean()
        non_svd_mean = non_svd["f1_macro"].mean()
        if svd_mean > non_svd_mean:
            lines.append(f"A redução de dimensionalidade (SVD) trouxe ganho médio de "
                         f"{(svd_mean - non_svd_mean)*100:.2f}pp sobre vetorizadores sem SVD.<br>")
        else:
            lines.append(f"Vetorizadores sem SVD superaram os com SVD em "
                         f"{(non_svd_mean - svd_mean)*100:.2f}pp na média, sugerindo que "
                         f"a redução perdeu informação relevante neste dataset.<br>")

    lines.append(f"<br><strong>2. Melhor classificador (média):</strong> <code>{best_clf}</code> - "
                 f"Média F1 = {by_clf.loc[best_clf, 'mean']:.4f}.<br>")

    lines.append(f"<br><strong>3. Melhor pré-processamento (média):</strong> <code>{best_pp[:80]}...</code> - "
                 f"Média F1 = {by_pp.loc[best_pp, 'mean']:.4f}.<br>")

    # Verifica gap entre teste e validação como indicador de overfitting
    if "val_f1_macro" in valid.columns:
        best_gap = abs(best["f1_macro"] - best.get("val_f1_macro", best["f1_macro"]))
        if best_gap < 0.02:
            lines.append("<br><strong>4. Generalização:</strong> A diferença entre F1 de teste e "
                         f"validação é de apenas {best_gap*100:.2f}pp, indicando <strong>boa "
                         "generalização</strong> sem sinais de overfitting.")
        else:
            lines.append(f"<br><strong>4. Atenção:</strong> Gap de {best_gap*100:.2f}pp entre "
                         "teste e validação pode indicar <strong>variabilidade</strong> nos dados.")

    return "".join(lines)


if __name__ == "__main__":
    main()
