# 🧠 LLM Smart Classifier

Projeto completo de classificação inteligente de tweets humanitários, utilizando Big Data com PySpark, Machine Learning supervisionado, Modelos de Linguagem (LLMs), MLOps e deploy em nuvem.

---

## 🚀 Visão Geral

Este projeto tem como objetivo construir uma solução capaz de **entender, categorizar e explicar automaticamente milhares de mensagens humanitárias** postadas em redes sociais durante crises e emergências.

Por meio de uma abordagem moderna e escalável, unimos técnicas de NLP com modelos clássicos de machine learning, modelos de linguagem (LLMs), pipelines de MLOps e interfaces de uso para entregar um sistema robusto e aplicável em cenários reais.

---

## 🧩 Estrutura Modular do Projeto

| Módulo | Descrição |
|--------|-----------|
| [1. Ingestão e Pré-processamento](#modulo-1--ingestao-e-pre-processamento-com-pyspark) | Leitura distribuída dos dados com PySpark, limpeza e tokenização |
| [2. Análise Exploratória de Dados](#modulo-2--analise-exploratoria-de-dados-eda) | Exploração, frequência de palavras, n-gramas e padrões |
| [3. Engenharia de Features e Modelagem](#modulo-3--engenharia-de-features-e-modelagem) | Vetorização, modelos supervisionados e tracking com MLFlow |
| [4. Aplicação de LLMs](#modulo-4--aplicacao-de-llms) | Embeddings, zero-shot, few-shot, explicações e comparação |
| [5. MLOps e Deploy](#modulo-5--mlops-e-deploy) | Pipeline, API, Docker, Azure ML, CI/CD |
| [6. Validação Avançada](#modulo-6--validacao-avancada-e-ab-testing) | Testes estatísticos e comparação entre abordagens |
| [7. Interface Visual Interativa](#modulo-7--interface-visual-interativa) | App com input e explicações acessíveis |
| [8. Documentação e Storytelling Técnico](#modulo-8--documentacao-e-storytelling-tecnico) | README, prints, impacto e visão estratégica |

---

## 🔹 Módulo 1 – Ingestão e Pré-processamento com PySpark

**Objetivo:**  
Ingerir os dados brutos no ambiente distribuído do PySpark, realizar o pré-processamento textual necessário para análise posterior e salvar os dados limpos em formato otimizado (Parquet).

---

### 1.1 Leitura do dataset no ambiente distribuído

- O dataset foi carregado diretamente no ambiente Databricks com PySpark, contendo aproximadamente **47.868 tweets** humanitários.
- Cada linha representa uma mensagem classificada por categoria humanitária (como saúde, abrigo, comida, etc.).

*Trecho de código:*
```python
# Copiando o ZIP da DBFS para o sistema de arquivos local do cluster
dbutils.fs.cp("dbfs:/FileStore/tables/humaid_eventwise.zip", "file:/tmp/humaid_eventwise.zip")

# Extraindo
with zipfile.ZipFile("/tmp/humaid_eventwise.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp/humaid_eventwise")
```

---

### 1.2 Limpeza textual e normalização

**Objetivo:**  
Preparar os textos dos tweets para processamento linguístico, removendo ruídos e padronizando a estrutura textual. Esta etapa é essencial para melhorar a qualidade dos dados e garantir resultados mais precisos nos próximos módulos de análise e modelagem.

---

**Transformações aplicadas:**

- ✅ Conversão de todos os caracteres para minúsculas;
- ✅ Remoção de:
  - **Links/URLs** (ex: `http://...`);
  - **Menções** (ex: `@username`);
  - **Hashtags** (ex: `#emergência`);
  - **Pontuação e caracteres especiais** (ex: `!`, `?`, `...`);
  - **Emojis** e **símbolos não textuais**;
  - **Múltiplos espaços em branco**.

---

**Nova coluna criada:** `text_clean`  
Essa coluna representa o conteúdo textual limpo de cada tweet.

---

*Trecho de código:*
```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import re

# Função Python para limpeza
def limpar_texto(texto):
    if texto:
        texto = texto.lower()
        texto = re.sub(r"http\S+", "", texto)                     # Remove links
        texto = re.sub(r"rt\s+@[\w_]+:? ?", "", texto)            # Remove RTs com @usuario
        texto = re.sub(r"@\w+", "", texto)                        # Remove todas as outras menções
        texto = re.sub(r"#\w+", "", texto)                        # Remove hashtags
        texto = re.sub(r"[^\w\s]", "", texto)                     # Remove pontuação
        texto = re.sub(r"\s+", " ", texto).strip()                # Espaços extras
    return texto

# Registrando a função como UDF do PySpark
limpar_udf = udf(limpar_texto, StringType())

# Criando nova coluna com texto limpo
df_limpo = df_total.withColumn("text_clean", limpar_udf(col("tweet_text")))

# Visualizando resultado
df_limpo.select("tweet_text", "text_clean").show(10, truncate=False)
```

---

### 1.3 Tokenização e remoção de stopwords

**Objetivo:**  
Transformar os textos limpos em **listas de palavras relevantes**, removendo palavras comuns (como "the", "and", "is", etc.) que não agregam valor semântico. Essa etapa é essencial para preparar os dados para análise textual e modelagem supervisionada.

---

**Transformações aplicadas:**

- ✅ **Tokenização:** separação do texto em palavras individuais (*tokens*);
- ✅ **Remoção de stopwords:** exclusão de palavras muito frequentes e com pouco significado no contexto de NLP;
- ✅ **Nova coluna gerada:** `tokens_filtrados`, contendo as listas de palavras úteis.

---

*Trecho de código:*
```python
# Etapa 1: Tokenização
from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="text_clean", outputCol="tokens")
df_tokens = tokenizer.transform(df_limpo)

# Etapa 2: Remoção de stopwords
from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_filtrados")
df_tokens_filtrados = remover.transform(df_tokens)

# Visualizando o resultado
display(df_tokens_filtrados.head(5))
```

---

### 1.4 Salvamento em formato Parquet

**Objetivo:**  
Persistir os dados pré-processados em um formato otimizado (`Parquet`), garantindo **eficiência de leitura**, **compactação de armazenamento** e **compatibilidade com processamento distribuído** em etapas futuras do projeto.

---

**Por que Parquet?**

- ✅ Formato **colunar**, ideal para consultas analíticas;
- ✅ Compatível com ferramentas distribuídas como PySpark;
- ✅ Permite leitura seletiva de colunas, economizando tempo e recursos;
- ✅ Otimiza a performance de etapas como vetorização e modelagem.

---

*Trecho de código:*
```python
# Salvar DataFrame como .parquet
df_tokens_filtrados.write.mode("overwrite").parquet("/tmp/humaid_dados_limpos")

# Leitura futura:
df_pronto= spark.read.parquet("/tmp/humaid_dados_limpos")
df_pronto.show(5, truncate=False)
```

---

## 🔹 Módulo 2 – Análise Exploratória de Dados (EDA)

**Objetivo:**  
Compreender a estrutura, distribuição e principais características dos tweets humanitários após o pré-processamento. Essa etapa permite identificar padrões, anomalias e orientar decisões futuras na modelagem.

---

### 2.1 Visão Geral dos Dados

**Objetivo da etapa:**  
Realizar uma análise inicial para entender:
- O volume total de registros disponíveis;
- A estrutura e os tipos das colunas;
- Presença de valores nulos;
- Exemplo de registros após limpeza.

---

*Trecho de código:*
```python
# Contar o total de linhas
print("Total de tweets:", df_tokens_filtrados.count())

# Ver categorias únicas
df_tokens_filtrados.select("class_label").distinct().show(truncate=False)

# Verificando a contagem de linhas por categoria
df_tokens_filtrados.groupby("class_label").count().orderBy("count", ascending=False).show(truncate=False)
```

---

### 2.2 Distribuição das Categorias (Balanceamento)

**Objetivo:**  
Analisar a variável-alvo `class_label` para entender o **balanceamento entre as categorias humanitárias**. Isso é crucial para identificar possíveis desequilíbrios que possam influenciar negativamente os modelos de classificação supervisionada.

---

**Transformações e ações aplicadas:**

- ✅ Agrupamento por categoria com contagem de ocorrências;
- ✅ Ordenação decrescente para facilitar a análise;
- ✅ Geração de gráfico de barras para visualização do desequilíbrio.

---

*Trecho de código:*
```python
# Agrupar em categorias
df_labels = df_tokens_filtrados.groupBy("class_label").count()

# Ordenar categorias do maior para o menor
df_labels = df_labels.orderBy("count",ascending=False)

# Converter para o pandas
df_labels = df_labels.toPandas()
```

---

### 2.3 Frequência de Palavras por Categoria (Análise Quantitativa)

**Objetivo:**  
Identificar as **palavras mais frequentes dentro de cada categoria humanitária**, a fim de entender os termos mais característicos de cada tipo de situação. Essa análise ajuda na construção de vetores de texto mais representativos para os modelos.

---

**Transformações e ações aplicadas:**

- ✅ Explosão da coluna `tokens_filtrados` em palavras individuais;
- ✅ Agrupamento por `class_label` + `token`;
- ✅ Contagem de ocorrências de cada palavra por categoria;
- ✅ Filtragem das palavras mais frequentes por classe.

---

*Trecho de código:*
```python
# Selecionar categorias dominantes
categorias_dominantes = ["rescue_volunteering_or_donation_effort","other_relevant_information","infrastructure_and_utility_damage","sympathy_and_support","injured_or_dead  _people"]

# Filtrar nosso DataFrame com as categorias dominantes
df_filtrado = df_tokens_filtrados.filter(df_tokens_filtrados["class_label"].isin(categorias_dominantes))

df_filtrado.show(5)

# Gerar texto concatenado com categoria
from pyspark.sql.functions import explode, collect_list, concat_ws

# Vamos explodir as palavras
df_explode = df_filtrado.select("class_label", explode("tokens_filtrados").alias("token"))

df_explode.show(5)
```

---

### 2.4 Frequência de N-Gramas (Bigramas e Trigramas)

**Objetivo:**  
Explorar **composições de palavras mais comuns** (bigramas e trigramas) nos tweets humanitários. Isso permite capturar estruturas linguísticas relevantes, como “need food”, “medical help”, ou “people are trapped”, que carregam mais contexto do que palavras isoladas.

---

**Transformações e ações aplicadas:**

- ✅ Utilização do `NGram` do PySpark para gerar bigramas e trigramas a partir da coluna `tokens_filtrados`;
- ✅ Explosão e contagem de n-gramas;
- ✅ Análise das expressões mais frequentes por categoria.

---

*Trecho de código (Bigramas):*
```python
from pyspark.sql.functions import size

df_tokens_filtrados = df_tokens_filtrados.withColumn("text_length", size(col("tokens_filtrados")))
df_tokens_filtrados.show(5)

# Estatísticas das palavras
df_tokens_filtrados.select("text_length").describe().show()
```

---

### 2.5 Comprimento dos Textos

**Objetivo:**  
Analisar a **distribuição do comprimento dos textos**, medido em número de tokens, para entender o nível de complexidade linguística e possíveis padrões associados a diferentes categorias humanitárias.

---

**Transformações e ações aplicadas:**

- ✅ Cálculo da quantidade de tokens em cada tweet (coluna `tokens_filtrados`);
- ✅ Geração de histogramas para visualizar a distribuição geral e por categoria;
- ✅ Identificação de outliers e padrões relevantes (ex: categorias com textos mais curtos ou longos).

---

*Trecho de código:*
```python
# Explodir palavras por linha
from pyspark.sql.functions import explode

df_exploded_limpo = df_tokens_filtrados_limpos.select("class_label", explode("tokens_filtrados").alias("token"))

# Agrupa por categoria
from pyspark.sql.functions import collect_list

df_grouped_tokens = df_exploded_limpo.groupBy("class_label") \
    .agg(collect_list("token").alias("all_tokens"))

# Gerar unigramas, bigramas e trigramas

import pandas as pd
from nltk.util import ngrams
from collections import Counter

df_tokens_pd = df_grouped_tokens.toPandas()

def extrair_top_ngrams(tokens, n, top_n=10):
    return Counter(ngrams(tokens, n)).most_common(top_n)

rows = []

for _, row in df_tokens_pd.iterrows():
    categoria = row["class_label"]
    tokens = row["all_tokens"]

    # Top palavras, bigramas e trigramas
    top_unigrams = [uni[0] for uni in Counter(tokens).most_common(10)]
    top_bigrams = [" ".join(bi) for bi, _ in extrair_top_ngrams(tokens, 2)]
    top_trigrams = [" ".join(tri) for tri, _ in extrair_top_ngrams(tokens, 3)]

    rows.append({
        "class_label": categoria,
        "top_10_unigrams": ", ".join(top_unigrams),
        "top_10_bigrams": ", ".join(top_bigrams),
        "top_10_trigrams": ", ".join(top_trigrams),
    })

df_ngrams_limpo = pd.DataFrame(rows)
```
---

### 2.6 Correlações entre Rótulos e Padrões Linguísticos

**Objetivo:**  
Investigar se existem **correlações entre características linguísticas** (como comprimento do texto e presença de determinadas palavras) e os rótulos das categorias humanitárias. Essa análise ajuda a entender **quais padrões o modelo pode aprender para distinguir entre as classes**.

---

**Transformações e ações aplicadas:**

- ✅ Uso da coluna `text_length` criada anteriormente;
- ✅ Análise de palavras-chave mais frequentes por categoria;
- ✅ Cálculo de medidas estatísticas básicas para comparar os grupos.

---

*Trecho de código – Distribuição do comprimento por categoria:*
```python
# Frequência de palavras por categoria

from pyspark.sql.functions import count

df_freq = df_exploded_limpo.groupBy("class_label", "token").agg(count("*").alias("freq"))
df_freq.show(5, truncate=False)

# Converter para pandas
df_freq_pd = df_freq.toPandas()

# Obter top 10 palavras por categoria
top_tokens_por_categoria = (
    df_freq_pd.sort_values("freq", ascending=False)
    .groupby("class_label")
    .head(10)
    .reset_index(drop=True)
)
```

---

## 🔹 Módulo 3 – Engenharia de Features e Modelagem

**Objetivo:**  
Transformar os textos em representações numéricas (vetores) que possam ser compreendidas por algoritmos de Machine Learning. Em seguida, treinar modelos supervisionados para prever a categoria humanitária de novos tweets com base em seus conteúdos.

---

### 3.1 Vetorização com TF-IDF (Scikit-learn)

**Objetivo:**  
Transformar os textos dos tweets em vetores numéricos utilizando a técnica **TF-IDF (Term Frequency – Inverse Document Frequency)** com o `TfidfVectorizer` do Scikit-learn. Essa técnica permite que os modelos identifiquem quais palavras (ou sequências de palavras) são mais relevantes para classificar corretamente cada tipo de emergência.

---

**Transformações e ações aplicadas:**

- ✅ Reconstrução do texto original a partir dos tokens filtrados;
- ✅ Conversão do DataFrame PySpark para Pandas;
- ✅ Aplicação do `TfidfVectorizer` com n-gramas (1 a 3 palavras);
- ✅ Criação da matriz esparsa `X` com até 5.000 features.

---

*Trecho de código:*
```python
# Juntar palavras no texto
from pyspark.sql.functions import concat_ws

df_texto = df_tokens_filtrados_limpos.select(
    "class_label",
    concat_ws(" ", "tokens_filtrados").alias("texto")
)

# Converter para Pandas
df_modelo = df_texto.toPandas()

# Vetorização com TF-IDF + N-gramas

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
X = vectorizer.fit_transform(df_modelo["texto"])
```

---

### 3.2 Treinamento de modelos supervisionados (LogReg, Random Forest)

**Objetivo:**  
Treinar modelos de classificação supervisionada utilizando os vetores TF-IDF gerados anteriormente, com o objetivo de prever a categoria (`class_label`) de cada tweet humanitário.

---

**Modelos utilizados:**

- ✅ **Logistic Regression (LogReg):** modelo linear bastante eficiente para classificações multiclasses com vetores esparsos.
- ✅ **Random Forest Classifier:** modelo de ensemble baseado em árvores de decisão, mais robusto a não-linearidades e variações.

---

*Trecho de código:*
```python
# Separar treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Treinar o modelo Regressão Logistica
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
```

---

### 3.3 Avaliação com F1-Score e Matriz de Confusão

**Objetivo:**  
Avaliar o desempenho dos modelos Logistic Regression e Random Forest utilizando métricas apropriadas para classificação multiclasse, com foco em **F1-Score**, **Acurácia** e **Matriz de Confusão**.

---

**Métricas utilizadas:**

- ✅ **F1-Score (macro):** média harmônica entre precisão e recall, ponderada igualmente entre as classes;
- ✅ **Acurácia:** proporção de classificações corretas sobre o total;
- ✅ **Matriz de Confusão:** mostra visualmente os acertos e erros por classe.

---

**Ações aplicadas:**

- Geração de predições no conjunto de teste;
- Visualização da matriz de confusão com Pandas e Seaborn.

---

*Trecho de código:*
```python
# Avaliar os dois modelos
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Avaliação - Logistic Regression
y_pred_logreg = logreg.predict(X_test)
print("📊 Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))

# Avaliação - Random Forest
y_pred_rf = rf.predict(X_test)
print("📊 Random Forest:")
print(classification_report(y_test, y_pred_rf))
```

---

### 3.4 Tracking de experimentos com MLflow

**Objetivo:**  
Rastrear os experimentos de machine learning com o MLflow, registrando automaticamente métricas, parâmetros, modelos e artefatos. Isso permite **comparar modelos, reproduzir resultados e documentar a evolução do projeto de forma profissional**.

---

**Recursos monitorados:**

- ✅ Métricas: F1-score, acurácia;
- ✅ Parâmetros dos modelos (ex: `maxIter`, `numTrees`);
- ✅ Artefatos: modelo treinado, matriz de confusão, configurações;
- ✅ Tags e anotações manuais para melhor organização.

---

**Configuração inicial:**
```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Caminho válido dentro do Databricks
experiment_name = "/Shared/llm-smart-classifier"

# Cria o experimento se não existir
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Inicia o tracking com o experimento criado
with mlflow.start_run(experiment_id=experiment_id):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("ngram_range", (1, 3))
    mlflow.log_param("max_features", 5000)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average="weighted"))

    mlflow.sklearn.log_model(model, "model")
```

---



