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
