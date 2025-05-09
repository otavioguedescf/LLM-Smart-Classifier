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
