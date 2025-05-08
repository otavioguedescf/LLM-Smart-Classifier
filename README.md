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

- **1.1** Leitura do dataset no ambiente distribuído
- **1.2** Limpeza textual e normalização
- **1.3** Tokenização e remoção de stopwords
- **1.4** Salvar em parquet

---

### 🔹 Módulo 2 – Análise Exploratória de Dados (EDA)

- **2.1** Visão Geral dos Dados
- **2.2** Distribuição das Categorias (Balanceamento)
- **2.3** Frequência de Palavras por Categoria (Análise Quantitativa)
- **2.4** Frequência de N-Gramas (Bigramas e Trigramas)
- **2.5** Comprimento dos Textos
- **2.6** Correlações entre Rótulos e Padrões Linguísticos

---

### 🔹 Módulo 3 – Engenharia de Features e Modelagem

- **3.1** Vetorização com TF-IDF (n-gramas de 1 a 3)
- **3.2** Treinamento de modelos supervisionados (LogReg, Random Forest)
- **3.3** Avaliação com F1-Score e Matriz de Confusão
- **3.4** Tracking de experimentos com MLFlow
- **3.5** Seleção e salvamento do melhor modelo baseline

---

### 🔹 Módulo 4 – Aplicação de LLMs

- **4.1** Geração de embeddings com OpenAI ou HuggingFace
- **4.2** Classificação zero-shot e few-shot
- **4.3** Comparação com modelos tradicionais
- **4.4** Geração de explicações automáticas por LLM

---

### 🔹 Módulo 5 – MLOps e Deploy

- **5.1** Criação do pipeline com MLFlow
- **5.2** API REST com FastAPI
- **5.3** Containerização com Docker
- **5.4** Deploy local ou em Azure ML
- **5.5** CI/CD com GitHub Actions
- **5.6** Orquestração simulada com Airflow

---

### 🔹 Módulo 6 – Validação Avançada e A/B Testing

- **6.1** Comparação entre modelos tradicionais e LLMs
- **6.2** Testes estatísticos (t-test, bootstrap)
- **6.3** Interpretação dos resultados com base em significância

---

### 🔹 Módulo 7 – Interface Visual Interativa

- **7.1** Desenvolvimento de app com Streamlit ou Gradio
- **7.2** Campo de input para novos tweets
- **7.3** Exibição da categoria e explicação via SHAP ou LLM
- **7.4** Deploy acessível para demonstração

---

### 🔹 Módulo 8 – Documentação e Storytelling Técnico

- **8.1** Estruturação do README no GitHub
- **8.2** Registro visual das etapas (prints, imagens)
- **8.3** Apresentação didática para portfólio técnico

---

## 📌 Status do Projeto

- ✅ Módulo 1 concluído
- 🟡 Módulo 2 em andamento (atualmente no item 2.4)
- ⏳ Próximo passo: Análise de N-Gramas e Comprimento dos Textos

---

## 📎 Licença

Este projeto é de código aberto sob a licença MIT. Sinta-se à vontade para utilizar, referenciar e adaptar.

---

## 📅 Autor
Otávio Guedes
Cientista de Dados em transição de carreira, focado em projetos práticos de ponta a ponta.

Acesse o [meu perfil no LinkedIn](https://www.linkedin.com/in/otaviomendesguedes/)
