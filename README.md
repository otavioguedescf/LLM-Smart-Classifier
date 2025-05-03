# 🤖 LLM Smart Classifier

**Objetivo:** Desenvolver um pipeline completo de classificação de tweets relacionados a desastres humanitários, utilizando tecnologias modernas de Big Data (PySpark), NLP tradicional, LLMs, MLOps e deploy em nuvem, com visualização interativa e documentação profissional.

---

## 📊 Dataset

- **Fonte:** [HumAID – Event Wise Dataset (Set1)](https://crisisnlp.qcri.org/humaid_dataset.html)
- **Descrição:** 47.868 tweets rotulados em 11 categorias humanitárias.
- **Formato:** CSV, multilinha, com texto livre e classes como `infrastructure_damage`, `sympathy`, `rescue_effort`, entre outras.

---

## 🚀 Tecnologias Utilizadas

- `PySpark / Spark` – Processamento distribuído em larga escala
- `HuggingFace / OpenAI API` – Aplicação de LLMs (GPT, BERT, T5)
- `Scikit-learn / TF-IDF / LogisticRegression` – Modelos tradicionais (baseline)
- `MLFlow` – Tracking e versionamento de modelos
- `FastAPI` – API de deploy do classificador
- `Azure ML` ou `Databricks` – Ambiente cloud escalável
- `GitHub Actions` – CI/CD automatizado
- `Apache Airflow` – Orquestração de pipeline (MLOps)
- `Streamlit` ou `Gradio` – Visualização e interação com o modelo
- `SciPy / statsmodels` – Validação estatística e testes A/B

---

## 🧱 Estrutura do Projeto

### 🔹 Módulo 1 – Ingestão e Pré-processamento com PySpark
- Leitura do dataset no ambiente distribuído
- Limpeza textual, tokenização e stopwords
- Conversão de colunas e tratamento de tipos

### 🔹 Módulo 2 – Análise Exploratória de Dados (EDA)
- 2.1 Visão Geral dos Dados
- 2.2 Distribuição das Categorias
- 2.3 Frequência de Palavras por Categoria (Análise Quantitativa)
- 2.4 Frequência de N-Gramas (Bigramas e Trigramas)
- 2.5 Comprimento dos Textos
- 2.6 Correlações entre Rótulos e Padrões Linguísticos

### 🔹 Módulo 3 – Engenharia de Features e Modelagem
- Vetorização com TF-IDF
- Treinamento de modelos supervisionados (LogReg, Random Forest)
- Métricas de desempenho: F1, ROC AUC, Matriz de Confusão
- Tracking de experimentos com MLFlow

### 🔹 Módulo 4 – Aplicação de LLMs
- Geração de embeddings com OpenAI / HuggingFace
- Classificação zero-shot e few-shot
- Comparação entre modelos tradicionais e LLMs
- Geração de explicações automáticas por LLM

### 🔹 Módulo 5 – MLOps e Deploy
- Pipeline de treinamento com MLFlow
- API com FastAPI
- Containerização com Docker
- Deploy em Azure ML ou local
- CI/CD com GitHub Actions
- Orquestração com Airflow (simulação de pipeline)

### 🔹 Módulo 6 – Validação Avançada e A/B Testing
- Comparação entre modelos
- Testes estatísticos (t-test, bootstrap, etc.)
- Interpretação dos resultados

### 🔹 Módulo 7 – Interface Visual Interativa
- App em Streamlit ou Gradio
- Campo de input para classificação de novos tweets
- Retorno explicável (LLM ou SHAP)

### 🔹 Módulo 8 – Documentação e Storytelling Técnico
- README completo e visual
- Prints e artefatos do pipeline
- Apresentação clara e objetiva no GitHub

---

## 📌 Status do Projeto

- 🔧 Em andamento – Módulo 2: Análise Exploratória de Dados (EDA)
- ✅ Módulos concluídos: 1. Ingestão e Pré-processamento
- 🚧 Próximo passo: Frequência de N-Gramas (2.4)

---

## 📎 Licença

Este projeto é de código aberto sob a licença MIT. Sinta-se à vontade para utilizar e adaptar!

---

## 📅 Autor
Otávio Guedes
Cientista de Dados em transição de carreira, focado em projetos práticos de ponta a ponta.

Acesse o [meu perfil no LinkedIn](https://www.linkedin.com/in/otaviomendesguedes/)
