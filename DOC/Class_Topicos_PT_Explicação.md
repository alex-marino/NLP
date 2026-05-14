## Bloco 0 — Configuração inicial

```python
import os
from dotenv import load_dotenv
load_dotenv("../.env")
print("DeepSeek configurado:", bool(os.environ.get("DEEPSEEK_API_KEY")))
```

**O que faz:** carrega o arquivo `.env` (que tem a chave da API do DeepSeek) e injeta as variáveis no ambiente do Python. O `print` no final é uma verificação rápida: se imprimir `True`, a chave está disponível para o resto do código; se `False`, o `.env` não foi encontrado ou está vazio.

**Por que é importante:** todas as chamadas ao DeepSeek mais à frente dependem dessa chave. Falhar aqui significa que tudo do Bloco 10 em diante quebra.

---

## Bloco 1 — Carregando o dataset

```python
import pandas as pd
URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
df_full = pd.read_csv(URL)

df = df_full.dropna(subset=["review_text"]).copy()
df = df[df["review_text"].str.len() >= 30].reset_index(drop=True)
df = df.sample(n=20_000, random_state=42).reset_index(drop=True)

textos  = df["review_text"].tolist()
titulos = df["review_title"].fillna("").tolist()
```

**O que faz:** três operações em sequência.

1. **Download:** baixa ~40 MB com 130 mil avaliações de produtos da Americanas/Submarino.
2. **Limpeza:** descarta linhas vazias e reviews muito curtos (< 30 caracteres) — textos como "bom" ou "kkk" não viram cluster útil, só ruído.
3. **Amostragem:** pega 20 mil reviews aleatórios. O `random_state=42` garante que toda execução produz a mesma amostra (reprodutibilidade).

**Por que amostrar:** os 130 mil completos rodariam em ~20 minutos. Com 20 mil, todo o pipeline roda em ~3 minutos — suficiente para a aula sem perder qualidade dos clusters.

**Saída:** duas listas de strings — `textos` (corpo do review) e `titulos` (título curto). Daqui em diante o código só usa `textos`.

---

## Bloco 2 — Embeddings com modelo multilíngue

```python
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedding_model.encode(textos, show_progress_bar=True)
```

**O que faz:** transforma cada string em um vetor de 384 números (float). Cada vetor "codifica" o significado semântico do texto. Textos parecidos viram vetores próximos no espaço de 384 dimensões.

**Detalhe importante:** o modelo `paraphrase-multilingual-MiniLM-L12-v2` foi treinado em 50+ idiomas. O modelo do livro original (`thenlper/gte-small`) só funciona bem em inglês — se você usasse com português, os clusters ficariam ruins porque o modelo "não entende" bem as palavras.

**Saída:** `embeddings` é um array NumPy com shape `(20000, 384)` — 20 mil vetores, cada um com 384 números.

**Tempo:** ~1-2 minutos. A barra de progresso aparece porque tem 20 mil textos para processar.

---

## Bloco 3 — Redução de dimensionalidade com UMAP

```python
from umap import UMAP

umap_model = UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

viz_embeddings = UMAP(n_components=2, min_dist=0.0, metric="cosine",
                     random_state=42).fit_transform(embeddings)
```

**O que faz:** comprime os vetores de 384 dimensões para versões menores, **duas vezes**:

- **`reduced_embeddings`** (5 dim): usado pelo HDBSCAN no próximo bloco para clusterizar. 5 dimensões é o "ponto doce" — preserva estrutura sem o problema da "maldição da dimensionalidade" (em alta dimensão, distâncias perdem significado).
- **`viz_embeddings`** (2 dim): só serve para plotar no Bloco 5. Não tem como plotar 384 dimensões num gráfico bidimensional.

**Parâmetros importantes:**
- `min_dist=0.0`: força os pontos próximos a ficarem **bem** próximos, criando clusters mais densos (HDBSCAN adora isso).
- `metric="cosine"`: distância apropriada para embeddings (mede similaridade angular, não euclidiana).
- `random_state=42`: reprodutibilidade. UMAP é estocástico — sem isso, cada rodada gera resultados diferentes.

**Por que UMAP e não PCA:** PCA é linear e perde estrutura não-linear. UMAP preserva relações de vizinhança local, que é o que importa para clustering.

---

## Bloco 4 — Clustering com HDBSCAN

```python
from hdbscan import HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    metric="euclidean",
    cluster_selection_method="eom",
    core_dist_n_jobs=1,
).fit(reduced_embeddings)

clusters = hdbscan_model.labels_
```

**O que faz:** olha para os 20 mil pontos no espaço 5-D e agrupa os que estão próximos. Cada ponto recebe um número:
- `0, 1, 2, ...` → identificador do cluster
- `-1` → outlier (ponto isolado, não pertence a cluster nenhum)

**Parâmetros:**
- `min_cluster_size=30`: um grupo só "vira cluster" se tiver pelo menos 30 reviews. Menor → muitos clusters pequenos e granulares. Maior → poucos clusters grandes e genéricos. 30 é bom para reviews porque eles tendem a se agrupar em temas específicos (defeitos de tela, atraso na entrega, etc.).
- `cluster_selection_method="eom"`: "Excess of Mass" — método mais conservador, prefere clusters maiores quando há ambiguidade.
- `core_dist_n_jobs=1`: roda em processo único. É **a correção do bug do BrokenProcessPool** que você teve antes (descompasso numpy/joblib). Sem isso, o HDBSCAN pode quebrar.

**Por que HDBSCAN e não k-means:**
- HDBSCAN descobre o número de clusters sozinho. K-means exige você definir `k` antes.
- HDBSCAN aceita outliers (cluster -1). K-means força todo ponto a algum cluster, distorcendo os resultados.
- HDBSCAN encontra clusters com formas arbitrárias. K-means só faz "esferas".

**Saída esperada:** algo como 80-150 clusters detectados, com 30-40% dos reviews indo para outliers.

---

## Bloco 5 — Visualização 2D

```python
import matplotlib.pyplot as plt

df_plot = pd.DataFrame(viz_embeddings, columns=["x", "y"])
df_plot["cluster"] = [str(c) for c in clusters]

to_plot  = df_plot.loc[df_plot.cluster != "-1", :]
outliers = df_plot.loc[df_plot.cluster == "-1", :]

plt.scatter(outliers.x, outliers.y, alpha=0.05, s=2, c="grey")
plt.scatter(to_plot.x, to_plot.y, c=to_plot.cluster.astype(int),
            alpha=0.6, s=2, cmap="tab20b")
```

**O que faz:** plota os pontos no plano 2D (usando `viz_embeddings` do Bloco 3), com:
- **Pontos cinzas, quase transparentes:** outliers — formam o "fundo".
- **Pontos coloridos:** reviews em clusters. A cor codifica o ID do cluster.

**Por que dois `scatter` separados:** os outliers são muitos (milhares) e poluiriam o gráfico. Com `alpha=0.05` (95% transparente), eles formam uma "nuvem de fundo" sutil, e os clusters reais aparecem em destaque.

**Cuidado interpretativo:** essa visualização é só **ilustrativa**. A clusterização real aconteceu em 5 dimensões (Bloco 4); o 2D distorce as distâncias. Dois clusters que parecem juntos no gráfico podem estar bem separados no espaço original.

---

## Bloco 6 — BERTopic com vetorizador em português

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stopwords_pt = stopwords.words("portuguese")
stopwords_extras = ["produto", "comprei", ...]
stopwords_total = list(set(stopwords_pt + stopwords_extras))

vectorizer_model = CountVectorizer(
    stop_words=stopwords_total,
    min_df=5,
    ngram_range=(1, 2),
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
    language="multilingual",
).fit(textos, embeddings)
```

**Conceito chave:** o BERTopic não faz o clustering — ele **reaproveita** o que você já calculou (embeddings, UMAP, HDBSCAN) e adiciona uma camada nova: **descobrir quais palavras descrevem cada cluster**.

**Como funciona:**
1. Junta todos os reviews de um cluster num "super-documento".
2. Conta frequência de palavras em cada super-documento.
3. Aplica **c-TF-IDF** (uma versão modificada do TF-IDF): palavras que aparecem muito num cluster e pouco nos outros recebem peso alto.
4. As palavras com maior peso viram a "descrição" do cluster.

**Por que o vetorizador customizado é crucial:**
- Sem `stop_words`, palavras como "que", "para", "muito", "bom" dominam todos os clusters — vira lixo.
- `ngram_range=(1, 2)` captura bigramas: "entrega rápida", "produto chegou", "muito ruim" — mais informativos que palavras isoladas.
- `min_df=5` ignora palavras que aparecem em menos de 5 reviews — elimina typos e termos raros demais.

**Stop words extras:** palavras como "produto", "comprei", "americanas" aparecem em quase todo review, então não diferenciam clusters. Adicioná-las à lista de bloqueio melhora muito a qualidade dos tópicos.

**Saída:** `topic_model.get_topic_info()` mostra cada tópico com suas top-keywords. Exemplo do que você deve ver:

```
Topic  Count  Name
-1     7800   -1_que_não_para_uma  (outliers)
0      450    0_celular_tela_bateria_carregador
1      380    1_entrega_chegou_prazo_correios
...
```

---

## Bloco 7 — Visualizações nativas do BERTopic

```python
topic_model.visualize_documents(titulos, reduced_embeddings=viz_embeddings, ...)
topic_model.visualize_barchart()
topic_model.visualize_heatmap(n_clusters=20)
topic_model.visualize_hierarchy()
```

**O que cada uma faz:**

- **`visualize_documents`:** versão interativa do gráfico do Bloco 5. Cada ponto tem hover com o texto. Útil para apresentações.
- **`visualize_barchart`:** gráficos de barras das top-keywords de cada tópico. Ótimo para slides — mostra de relance "do que cada cluster fala".
- **`visualize_heatmap`:** matriz de similaridade entre tópicos. Tópicos que aparecem "quentes" entre si são candidatos a fusão (talvez sejam o mesmo tema dividido).
- **`visualize_hierarchy`:** dendrograma. Mostra como os tópicos se agrupariam se você "subisse o nível" — útil para descobrir super-temas.

**Para a aula:** rode todas, mas apresente principalmente `visualize_documents` (visual impactante) e `visualize_barchart` (mostra o conteúdo de cada cluster).

---

## Bloco 8 — Refinamento com KeyBERTInspired

```python
from bertopic.representation import KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(textos, representation_model=representation_model,
                          vectorizer_model=vectorizer_model)
```

**Problema que resolve:** o c-TF-IDF do Bloco 6 só olha frequência de palavras. Ele não "entende" o significado. Resultado: às vezes pega palavras frequentes mas pouco descritivas (tipo "comprado", "loja").

**O que o KeyBERTInspired faz:** para cada cluster:
1. Calcula o embedding "médio" do cluster.
2. Pega as top-30 palavras candidatas (via c-TF-IDF).
3. Reordena: as que têm embedding mais próximo do embedding-médio sobem no ranking.

**Efeito prático:** as keywords ficam **mais semanticamente representativas**, não só estatisticamente frequentes.

**Compare antes/depois:**
- Antes: `0_celular_smartphone_aparelho_marca_funciona`
- Depois: `0_smartphone_câmera_bateria_processador_desempenho`

---

## Bloco 9 — Diversificação com MMR

```python
from bertopic.representation import MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=0.3)
topic_model.update_topics(...)
```

**Problema que resolve:** os tópicos ainda têm redundância. Aparecem coisas como "celular | smartphone | aparelho" — três palavras que significam quase a mesma coisa, ocupando slots valiosos das top-10 keywords.

**O que o MMR faz:** escolhe keywords que são **relevantes** ao tópico **mas distantes umas das outras**. O parâmetro `diversity` (0 a 1) controla esse trade-off:
- `0.0` → escolha só por relevância (vai dar redundância).
- `1.0` → escolha só por diversidade (pode pegar palavras irrelevantes).
- `0.3` → equilíbrio bom.

**Efeito prático:**
- Antes: `celular | smartphone | aparelho | telefone | móvel`
- Depois: `celular | câmera | bateria | tela | desempenho`

A representação fica **muito mais informativa** com o mesmo número de palavras.

---

## Bloco 10 — Labels com DeepSeek (via LiteLLM)

```python
from bertopic.representation import LiteLLM

representation_model = LiteLLM(
    model="deepseek/deepseek-chat",
    prompt=prompt_deepseek_ptbr,
    nr_docs=4,
    delay_in_seconds=1,
)
topic_model.update_topics(textos, representation_model=representation_model,
                          vectorizer_model=vectorizer_model)
```

**O salto qualitativo:** keywords são bons para "máquina ler", mas humanos preferem **labels descritivos** tipo "Reclamações sobre atraso na entrega" em vez de `entrega | prazo | atraso | correios | demora`.

**Como o LLM ajuda:** para cada tópico, o BERTopic monta um prompt com:
- As top keywords do tópico.
- 4 reviews representativos.
- Pede ao DeepSeek que gere um label curto.

**Parâmetros:**
- `nr_docs=4`: número de documentos passados ao LLM. Mais documentos = mais contexto, mais tokens consumidos.
- `delay_in_seconds=1`: 1 segundo entre chamadas. Suficiente para o DeepSeek (não tem rate limit agressivo como o Gemini free tier).
- `prompt_deepseek_ptbr`: o prompt em português garante labels naturais em PT-BR.

**Custo:** rotular 100-150 tópicos com `deepseek-chat` custa menos de $0.01.

---

## Bloco 10b — DeepSeek direto com JSON estruturado

```python
from openai import OpenAI
deepseek_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# Para cada tópico:
resp = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt_estruturado.format(...)}],
    response_format={"type": "json_object"},
    temperature=0.0,
)
deepseek_labels[tid] = json.loads(resp.choices[0].message.content)
```

**Diferença do Bloco 10:** ali, o BERTopic gerencia tudo automaticamente, mas só obtém **um label curto** por tópico. Aqui, você chama o LLM diretamente e pede **estrutura mais rica**:

```json
{
  "rotulo": "Reclamações sobre atraso na entrega",
  "descricao": "Avaliações negativas focadas em prazos não cumpridos..."
}
```

**Truques:**
- `response_format={"type": "json_object"}`: força o LLM a retornar JSON válido. Sem isso, ele às vezes adiciona "Claro, aqui está: { ... }" e quebra o parser.
- `temperature=0.0`: respostas determinísticas. Para rotulagem, queremos consistência, não criatividade.
- **Rotula só os top-20:** percorrer 150 tópicos é caro e desnecessário. Os 20 maiores já cobrem ~80% dos reviews.

---

## Bloco 11 — Análise: tópicos × notas

```python
df["topic"] = topic_model.topics_

ranking = (
    df.loc[df.topic != -1]
      .groupby("topic")
      .agg(media_nota=("overall_rating", "mean"),
           total=("overall_rating", "count"))
      .sort_values("media_nota")
      .head(10)
)
```

**Conceito didático poderoso:** mostra que **topic modeling não termina nos tópicos**. Cruzar tópicos com **metadados estruturados** é onde o valor de negócio aparece.

**O que o código faz:**
1. Adiciona a coluna `topic` ao dataframe original (que já tem `overall_rating`, `product_category`, etc).
2. Agrupa por tópico, calcula a nota média de cada um.
3. Ordena pelos **piores** (menor nota média).

**Resultado:** uma lista tipo:

```
Tópico  17 | nota 1.42 | 230 reviews | Produtos defeituosos / quebrados
Tópico  42 | nota 1.67 | 180 reviews | Atraso extremo na entrega
Tópico   3 | nota 2.10 | 410 reviews | Tamanho diferente do anunciado
```

**Por que é valioso:** uma empresa pode usar isso para priorizar correções. Em pesquisa, é uma forma elegante de combinar análise não-supervisionada com sinais de supervisão fracos (a nota).

**Para a aula:** mostre esse bloco como o "punchline" — topic modeling não é fim em si, é uma ferramenta para descobrir estrutura, e a estrutura ganha valor quando combinada com outros dados.

---

## Recapitulando a jornada

| Bloco | Etapa do pipeline | Saída principal |
|-------|------------------|-----------------|
| 0 | Setup | Chaves de API carregadas |
| 1 | Dados | `textos` (20k reviews) |
| 2 | Embeddings | Vetores 384-D |
| 3 | Redução | Vetores 5-D + 2-D para plot |
| 4 | Clustering | IDs de cluster (~150 + outliers) |
| 5 | Visualização | Gráfico 2D colorido |
| 6 | Topic modeling | Tópicos com keywords |
| 7 | Visualizações | Gráficos interativos |
| 8 | Refinamento semântico | Keywords mais representativas |
| 9 | Diversificação | Keywords menos redundantes |
| 10 | Labels com LLM | Rótulos em linguagem natural |
| 10b | Análise estruturada | Labels + descrições JSON |
| 11 | Cruzamento com metadados | Tópicos negativos identificados |

Se algum bloco específico precisar de mais detalhamento (parâmetros, alternativas, troubleshooting), me diga qual.