# Tarefas Práticas — Aula 03: Semântica Vetorial e Análise de Sentimentos

**Disciplina:** Processamento de Linguagem Natural  
**Aula:** 03 - Semântica Vetorial e Embeddings  
**Dataset:** Dataset de análise de sentimentos (Twitter/Reviews)  
**Objetivo:** Aplicar técnicas de semântica vetorial (Word2Vec, GloVe, FastText) para análise de sentimentos em textos em português.

---

## 📚 Descrição da Tarefa

### Contexto Geral
Esta série de tarefas explora o conceito fundamental de **Semântica Vetorial** - a representação de palavras como vetores de números reais em espaços multidimensionais. Diferentemente da Aula 01 (TF-IDF) que usa vetorização baseada em frequência, a semântica vetorial captura relações semânticas entre palavras. Você aplicará essas técnicas para construir um **classificador de sentimentos** que aprende o significado contextual das palavras.

### Importância de Cada Tarefa

#### **Tarefa 1 - Exploração de Dataset de Sentimentos**
- **Por que é importante:** Compreender a distribuição de sentimentos, desbalanceamento de classes e características do texto é essencial. Datasets desequilibrados afetam o treinamento de modelos.
- **Aplicação prática:** Análise de reviews de produtos, monitoramento de redes sociais, análise de feedback de clientes.
- **Habilidades desenvolvidas:** Análise exploratória de dados sentimentais, tratamento de desbalanceamento, visualização de distribuições.

#### **Tarefa 2 - Pré-processamento Especializado para Sentimentos**
- **Por que é importante:** Sentimentos frequentemente envolvem gírias, emoticons, negações e ênfases que precisam de tratamento especial. Pré-processamento inadequado pode perder informação crucial.
- **Aplicação prática:** Análise de redes sociais, avaliação de comentários de clientes, monitoramento de reputação.
- **Habilidades desenvolvidas:** Tratamento de negações, gírias, emoticons em português; preservação de informação sentimental.

#### **Tarefa 3 - Representações com TF-IDF (Baseline)**
- **Por que é importante:** TF-IDF é a abordagem clássica que serve como baseline. Comparar com métodos mais modernos ajuda a entender seus trade-offs.
- **Aplicação prática:** Quando vetores densos não são viáveis (sparse high-dimensional) ou quando interpretabilidade é crítica.
- **Habilidades desenvolvidas:** Uso de TF-IDF para classificação, compreensão de baseline models.

#### **Tarefa 4 - Word Embeddings: Word2Vec (Skip-Gram)**
- **Por que é importante:** Word2Vec revolucionou PLN ao capturar relações semânticas. Skip-Gram prediz contexto a partir da palavra - ideal para descobrir relacionamentos.
- **Aplicação prática:** Análise de similaridade de palavras, descoberta de padrões semânticos, análise de desvios linguísticos.
- **Habilidades desenvolvidas:** Treinamento de modelos de embedding, análise de espaço vetorial, visualização com t-SNE.

#### **Tarefa 5 - GloVe e FastText: Além de Word2Vec**
- **Por que é importante:** GloVe combina frequência global com contexto local. FastText trata palavras raras melhor (importante em português com muitas flexões).
- **Aplicação prática:** Idiomas com flexões morfológicas (português, espanhol), datasets pequenos, palavras fora do vocabulário.
- **Habilidades desenvolvidas:** Compreensão de diferentes arquiteturas de embedding, escolha baseada em características do corpus.

#### **Tarefa 6 - Classificação com Diferentes Embeddings**
- **Por que é importante:** A qualidade dos embeddings impacta diretamente a performance do classificador. Comparar diferentes abordagens mostra trade-offs reais.
- **Aplicação prática:** Construção de pipelines de PLN em produção, otimização de modelos.
- **Habilidades desenvolvidas:** Machine learning, avaliação de modelos, interpretação de resultados, feature engineering.

#### **Tarefa 7 - Análise Comparativa e Interpretabilidade**
- **Por que é importante:** Não basta ter bom desempenho - é necessário entender POR QUÊ o modelo funciona. Análise de erros e interpretabilidade são críticas.
- **Aplicação prática:** Debug de modelos, detecção de vieses, melhorias contínuas.
- **Habilidades desenvolvidas:** Análise de erros, visualização de embeddings, detecção de padrões linguísticos, relatório técnico.

### Fluxo de Aprendizado

```
Dataset → Pré-proc → TF-IDF → Word2Vec → GloVe/FastText → Classificação → Análise
  (1)       (2)        (3)       (4)          (5)              (6)          (7)
```

**Progressão:** De representações simples (TF-IDF) para embeddings contextuais avançados (GloVe/FastText).

### Conceitos Fundamentais

**Semântica Vetorial:**
- Palavras semelhantes têm vetores próximos no espaço
- Relações entre palavras: `king - man + woman ≈ queen`
- Espaço multidimensional captura múltiplos significados

**Embeddings vs TF-IDF:**
- TF-IDF: Frequência baseada, esparso, interpretável
- Embeddings: Contextual, denso, mais poderoso para ML

**Arquiteturas:**
- **Word2Vec:** Skip-Gram (palavra prediz contexto) ou CBOW (contexto prediz palavra)
- **GloVe:** Combina frequência global + contexto local
- **FastText:** Subword embeddings (importante para português)

### Competências ao Final

Após completar todas as 7 tarefas, você será capaz de:
- ✅ Explorar e preparar datasets de sentimentos
- ✅ Treinare avaliar múltiplos modelos de embedding
- ✅ Construir classificadores de sentimentos com diferentes abordagens
- ✅ Comparar performance de diferentes representações vetoriais
- ✅ Interpretar e visualizar espaços semânticos
- ✅ Comunicar trade-offs entre abordagens
- ✅ Aplicar técnicas de PLN em problemas reais

---

## Tarefa 1: Exploração de Dataset de Sentimentos

### Objetivo
Entender a estrutura, distribuição e características do dataset de análise de sentimentos.

### Dataset Recomendado

**Opção 1: Reviews de Produtos (Português) - Kaggle**
- Dataset: `portuguese-sentiment-tweets` ou `amazon-reviews-pt`
- Características: Reviews reais, sentimentos positivos/negativos/neutros
- Tamanho: ~5000-10000 comentários
- Link: https://www.kaggle.com/datasets/

**Opção 2: Twitter Sentiment (Português)**
- Dataset: `portuguese-twitter-sentiment` 
- Características: Tweets em português com sentimentos
- Tamanho: Grande volume
- Domínio: Redes sociais

**Opção 3: B2W-Reviews01b (Português)**
- Dataset: Reviews de e-commerce brasileiro
- Características: Dados reais de produção
- Tamanho: Dados balanceados

### Código Inicial:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Carregar dataset
df = pd.read_csv('DATA/sentiment_dataset.csv', encoding='utf-8')

# 2. Exploração básica
print("="*70)
print("ANÁLISE EXPLORATÓRIA DO DATASET")
print("="*70)

print(f"\nShape do dataset: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
print(f"\nPrimeiras 5 linhas:")
print(df.head())

# 3. Análise de sentimentos
print(f"\n\nDISTRIBUIÇÃO DE SENTIMENTOS:")
if 'sentiment' in df.columns:
    dist_sentimentos = df['sentiment'].value_counts()
    print(dist_sentimentos)
    print(f"\nProporção (%):")
    print(df['sentiment'].value_counts(normalize=True) * 100)

# 4. Análise de texto
print(f"\n\nESTATÍSTICAS DE TEXTO:")
if 'text' in df.columns or 'comment' in df.columns:
    text_col = 'text' if 'text' in df.columns else 'comment'
    df['text_length'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    
    print(f"Comprimento médio: {df['text_length'].mean():.0f} caracteres")
    print(f"Palavras médias: {df['word_count'].mean():.0f}")
    print(f"Min/Max comprimento: {df['text_length'].min()}/{df['text_length'].max()}")

# 5. Visualizar distribuição
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribuição de sentimentos
if 'sentiment' in df.columns:
    dist_sentimentos.plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='navy')
    axes[0, 0].set_title('Distribuição de Sentimentos', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Sentimento')
    axes[0, 0].set_ylabel('Frequência')

# Distribuição de comprimento
axes[0, 1].hist(df['text_length'], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_title('Distribuição de Comprimento de Texto', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Caracteres')
axes[0, 1].set_ylabel('Frequência')

# Distribuição de palavras
axes[1, 0].hist(df['word_count'], bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Distribuição de Número de Palavras', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Palavras')
axes[1, 0].set_ylabel('Frequência')

# Sentimentos por comprimento (boxplot)
if 'sentiment' in df.columns:
    df.boxplot(column='text_length', by='sentiment', ax=axes[1, 1])
    axes[1, 1].set_title('Comprimento de Texto por Sentimento', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('exploracao_sentimentos.png', dpi=150)
print("\n✓ Gráfico salvo: exploracao_sentimentos.png")
plt.show()

# 6. Análise de desbalanceamento
print(f"\n\nANÁLISE DE DESBALANCEAMENTO:")
if 'sentiment' in df.columns:
    class_distribution = df['sentiment'].value_counts()
    max_class = class_distribution.max()
    min_class = class_distribution.min()
    ratio = max_class / min_class
    print(f"Razão máximo/mínimo: {ratio:.2f}x")
    if ratio > 3:
        print(f"⚠️ Dataset muito desbalanceado! Considere oversampling/undersampling.")
    else:
        print(f"✓ Dataset razoavelmente balanceado.")
```

### Entregáveis:
- Código comentado com análise exploratória
- Gráficos de distribuição
- Relatório sobre características do dataset
- Identificação de desafios (desbalanceamento, valores faltantes, etc.)

---

## Tarefa 2: Pré-processamento Especializado para Análise de Sentimentos

### Objetivo
Preparar textos para análise de sentimentos, preservando informação sentimental crucial.

### Código Inicial:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')

# Stopwords português
stop_words_pt = set(stopwords.words('portuguese'))

# Remover negações de stopwords (importantes para sentimento!)
negacoes = {'não', 'nunca', 'jamais', 'nenhum', 'nada', 'sem'}
stop_words_pt = stop_words_pt - negacoes

# 1. Tratamento de emoticons e gírias sentimentais
emoticons = {
    ':)': 'sorriso',
    ':(': 'triste',
    ':D': 'muito_feliz',
    '😀': 'sorriso',
    '😢': 'triste',
    '❤️': 'amor',
    '👍': 'positivo',
    '👎': 'negativo'
}

gírias_sentimento = {
    'amo': 'positivo',
    'adorei': 'positivo',
    'detesto': 'negativo',
    'odeio': 'negativo',
    'top': 'positivo',
    'ruim': 'negativo',
    'chato': 'negativo',
    'legal': 'positivo'
}

def preprocess_sentimento(texto):
    """Pré-processamento especializado para sentimentos."""
    
    # Converter para minúsculas
    texto = texto.lower()
    
    # Substituir emoticons por palavras-chave
    for emoticon, word in emoticons.items():
        texto = texto.replace(emoticon, f' {word} ')
    
    # Preservar ênfase (múltiplas letras: "muitooo" → "muito_ênfase")
    texto = re.sub(r'([a-z])\1{2,}', r'\1_ênfase', texto)
    
    # Remover URLs
    texto = re.sub(r'http\S+|www\S+', '', texto)
    
    # Remover menções (@usuario)
    texto = re.sub(r'@\w+', '', texto)
    
    # Remover hashtags (manter conteúdo: #incrível → incrível)
    texto = re.sub(r'#(\w+)', r'\1', texto)
    
    # Remover pontuação (manter alguns caracteres de ênfase)
    # Manter ! e ? que indicam emoção
    texto = re.sub(r'[.,;:\'\"`~^]', '', texto)
    
    # Tokenização
    tokens = word_tokenize(texto, language='portuguese')
    
    # Remover stopwords (exceto negações!)
    tokens = [t for t in tokens if t not in stop_words_pt and t.isalpha()]
    
    return ' '.join(tokens)

# 2. Aplicar pré-processamento
print("Pré-processando textos...")
df['texto_processado'] = df['text'].apply(preprocess_sentimento)

# 3. Mostrar exemplos
print("\nEXEMPLOS DE PRÉ-PROCESSAMENTO:")
for i in range(3):
    print(f"\nOriginal {i+1}:")
    print(f"  {df['text'].iloc[i]}")
    print(f"Processado {i+1}:")
    print(f"  {df['texto_processado'].iloc[i]}")

# 4. Análise de vocabulário
print(f"\n\nANÁLISE DE VOCABULÁRIO:")
from collections import Counter

all_tokens = ' '.join(df['texto_processado']).split()
vocab_freq = Counter(all_tokens)

print(f"Vocabulário único: {len(vocab_freq)}")
print(f"Total de tokens: {len(all_tokens)}")
print(f"Tokens médios por documento: {len(all_tokens) / len(df):.1f}")

print(f"\nTop 20 termos:")
for termo, freq in vocab_freq.most_common(20):
    print(f"  {termo:20} : {freq:4d}")

# 5. Análise de sentimentos específicos por termo
print(f"\n\nTERMOS MAIS SENTIMENTAIS:")
if 'sentiment' in df.columns:
    # Encontrar termos associados a sentimentos positivos
    positive_docs = df[df['sentiment'] == 'positive']['texto_processado']
    negative_docs = df[df['sentiment'] == 'negative']['texto_processado']
    
    all_pos_tokens = ' '.join(positive_docs).split()
    all_neg_tokens = ' '.join(negative_docs).split()
    
    pos_freq = Counter(all_pos_tokens)
    neg_freq = Counter(all_neg_tokens)
    
    print(f"\nTermos mais frequentes em POSITIVOS:")
    for termo, freq in pos_freq.most_common(10):
        print(f"  {termo:15} : {freq:4d}")
    
    print(f"\nTermos mais frequentes em NEGATIVOS:")
    for termo, freq in neg_freq.most_common(10):
        print(f"  {termo:15} : {freq:4d}")
```

### Entregáveis:
- Código de pré-processamento especializado
- Exemplos de transformações
- Análise de vocabulário
- Identificação de termos-chave de sentimento

---

## Tarefa 3: Classificação com TF-IDF (Baseline)

### Objetivo
Estabelecer um baseline usando TF-IDF e um classificador simples.

### Código Inicial:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import numpy as np

# 1. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['texto_processado'], 
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# 2. Vetorização TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    norm='l2'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nMatriz TF-IDF shape: {X_train_tfidf.shape}")
print(f"Features: {X_train_tfidf.shape[1]}")

# 3. Treinar Logistic Regression
print("\n" + "="*70)
print("LOGISTIC REGRESSION COM TF-IDF")
print("="*70)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
y_pred_proba_lr = lr_model.predict_proba(X_test_tfidf)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr, average='weighted'):.4f}")

print(f"\n{classification_report(y_test, y_pred_lr)}")

# 4. Treinar Random Forest
print("\n" + "="*70)
print("RANDOM FOREST COM TF-IDF")
print("="*70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

y_pred_rf = rf_model.predict(X_test_tfidf)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")

print(f"\n{classification_report(y_test, y_pred_rf)}")

# 5. Features mais importantes (TF-IDF)
print(f"\n\nFEATURES MAIS IMPORTANTES (TF-IDF):")
feature_names = tfidf.get_feature_names_out()
feature_importance = rf_model.feature_importances_
top_indices = np.argsort(feature_importance)[::-1][:20]

for idx, feature_idx in enumerate(top_indices, 1):
    print(f"  {idx:2d}. {feature_names[feature_idx]:20} (imp: {feature_importance[feature_idx]:.4f})")

# 6. Visualizar resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
axes[0].set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Comparação de modelos
models = ['Logistic Regression', 'Random Forest']
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf)
]
f1_scores = [
    f1_score(y_test, y_pred_lr, average='weighted'),
    f1_score(y_test, y_pred_rf, average='weighted')
]

x = np.arange(len(models))
width = 0.35

axes[1].bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
axes[1].bar(x + width/2, f1_scores, width, label='F1-Score', color='coral')
axes[1].set_ylabel('Score')
axes[1].set_title('Comparação de Modelos com TF-IDF', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].legend()
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('baseline_tfidf.png', dpi=150)
print("\n✓ Gráfico salvo: baseline_tfidf.png")
plt.show()
```

### Entregáveis:
- Baseline com TF-IDF estabelecido
- Comparação de classificadores
- Métricas de performance
- Visualizações

---

## Tarefa 4: Word2Vec (Skip-Gram) para Análise de Sentimentos

### Objetivo
Treinar modelos Word2Vec e usá-los para representar documentos.

### Código Inicial:

```python
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Preparar dados para Word2Vec (tokenizar)
print("Preparando dados para Word2Vec...")
sentences = [texto.split() for texto in df['texto_processado']]

# 2. Treinar Word2Vec Skip-Gram
print("\nTreinando Word2Vec (Skip-Gram)...")
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,          # Dimensionalidade
    window=5,                 # Contexto
    min_count=2,              # Frequência mínima
    sg=1,                     # 1=Skip-Gram, 0=CBOW
    workers=4,
    epochs=10,
    seed=42
)

print(f"Vocabulário: {len(w2v_model.wv)}")
print(f"Dimensão dos vetores: {w2v_model.vector_size}")

# 3. Explorar similaridades
print("\n" + "="*70)
print("SIMILARIDADES DESCOBERTAS")
print("="*70)

termos_teste = ['bom', 'ruim', 'gostei', 'horrível', 'adorei']
for termo in termos_teste:
    if termo in w2v_model.wv:
        print(f"\nPalavras similares a '{termo}':")
        similares = w2v_model.wv.most_similar(termo, topn=5)
        for palavra, score in similares:
            print(f"  {palavra:15} (similaridade: {score:.4f})")

# 4. Criar representação de documento (média dos vetores)
def documento_para_vetor_w2v(texto, modelo, dimensão=100):
    """Converter documento em vetor (média dos word vectors)."""
    palavras = texto.split()
    vetores = []
    
    for palavra in palavras:
        if palavra in modelo.wv:
            vetores.append(modelo.wv[palavra])
    
    if len(vetores) == 0:
        return np.zeros(dimensão)
    
    return np.mean(vetores, axis=0)

# 5. Vetorizar documents
print("\n\nVetorizando documentos com Word2Vec...")
X_train_w2v = np.array([documento_para_vetor_w2v(texto, w2v_model) 
                         for texto in X_train])
X_test_w2v = np.array([documento_para_vetor_w2v(texto, w2v_model) 
                        for texto in X_test])

print(f"X_train_w2v shape: {X_train_w2v.shape}")
print(f"X_test_w2v shape: {X_test_w2v.shape}")

# 6. Escalar
scaler = StandardScaler()
X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
X_test_w2v_scaled = scaler.transform(X_test_w2v)

# 7. Treinar classificadores
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("\n" + "="*70)
print("LOGISTIC REGRESSION COM WORD2VEC")
print("="*70)

lr_w2v = LogisticRegression(max_iter=1000, random_state=42)
lr_w2v.fit(X_train_w2v_scaled, y_train)

y_pred_w2v = lr_w2v.predict(X_test_w2v_scaled)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_w2v):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_w2v, average='weighted'):.4f}")

print(f"\n{classification_report(y_test, y_pred_w2v)}")

# 8. Visualizar embeddings com t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("\nGerando visualização t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(w2v_model.wv.vectors[:500])  # Primeiras 500 palavras

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=30)

# Adicionar labels para termos importantes
termos_label = ['bom', 'ruim', 'gostei', 'adorei', 'horrível', 'legal', 'chato', 'perfeito']
for termo in termos_label:
    if termo in w2v_model.wv:
        idx = list(w2v_model.wv.index_to_key).index(termo)
        if idx < 500:
            plt.annotate(termo, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                        fontsize=10, fontweight='bold')

plt.title('Word2Vec Embeddings (t-SNE)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('word2vec_tsne.png', dpi=150)
print("✓ Gráfico salvo: word2vec_tsne.png")
plt.show()
```

### Entregáveis:
- Modelo Word2Vec treinado
- Análise de similaridades semânticas
- Documentos vetorizados
- Comparação com baseline
- Visualização t-SNE

---

## Tarefa 5: GloVe e FastText - Além de Word2Vec

### Objetivo
Comparar GloVe e FastText como alternativas ao Word2Vec.

### Código Inicial:

```python
from gensim.models.fasttext import FastText
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Preparar sentences
sentences = [texto.split() for texto in df['texto_processado']]

# 1. FASTTEXT
print("="*70)
print("FASTTEXT - Embeddings com Subword")
print("="*70)

ft_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    sg=1,  # Skip-gram
    seed=42
)

print(f"Vocabulário FastText: {len(ft_model.wv)}")

# Demonstrar vantagem de FastText: palavras não vistas
print("\nVANTAGEM DE FASTTEXT - Subword embeddings:")
print("FastText pode gerar vetores para palavras não vistas no treino")
print("Exemplo: 'sensacionalismo' pode ser decomposto em subwords")

# 2. Função para vetorizar com FastText
def documento_para_vetor_ft(texto, modelo, dimensão=100):
    palavras = texto.split()
    vetores = [modelo.wv[palavra] for palavra in palavras if len(palavra) > 0]
    return np.mean(vetores, axis=0) if vetores else np.zeros(dimensão)

# 3. Vetorizar
X_train_ft = np.array([documento_para_vetor_ft(texto, ft_model) for texto in X_train])
X_test_ft = np.array([documento_para_vetor_ft(texto, ft_model) for texto in X_test])

scaler_ft = StandardScaler()
X_train_ft_scaled = scaler_ft.fit_transform(X_train_ft)
X_test_ft_scaled = scaler_ft.transform(X_test_ft)

# 4. Classificar com FastText
lr_ft = LogisticRegression(max_iter=1000, random_state=42)
lr_ft.fit(X_train_ft_scaled, y_train)
y_pred_ft = lr_ft.predict(X_test_ft_scaled)

print(f"\nFastText + Logistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_ft):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_ft, average='weighted'):.4f}")

# 5. COMPARAÇÃO DE TODAS AS ABORDAGENS
print("\n" + "="*70)
print("COMPARAÇÃO DE TODAS AS ABORDAGENS")
print("="*70)

resultados = {
    'TF-IDF (Logistic)': accuracy_score(y_test, y_pred_lr),
    'TF-IDF (Random Forest)': accuracy_score(y_test, y_pred_rf),
    'Word2Vec (Skip-Gram)': accuracy_score(y_test, y_pred_w2v),
    'FastText': accuracy_score(y_test, y_pred_ft)
}

for modelo, acc in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    print(f"{modelo:30} : {acc:.4f}")

# 6. Visualizar comparação
import matplotlib.pyplot as plt

models = list(resultados.keys())
accuracies = list(resultados.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=['steelblue', 'coral', 'lightgreen', 'gold'], 
               edgecolor='navy', linewidth=2)

# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparação de Diferentes Abordagens', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('comparacao_abordagens.png', dpi=150)
print("\n✓ Gráfico salvo: comparacao_abordagens.png")
plt.show()

# 7. Análise de similaridades FastText
print("\n\nSIMILARIDDDES COM FASTTEXT:")
termos_teste = ['bom', 'ruim', 'gostei']
for termo in termos_teste:
    try:
        similares = ft_model.wv.most_similar(termo, topn=3)
        print(f"\n{termo}: {[s[0] for s in similares]}")
    except:
        pass
```

### Entregáveis:
- Modelos FastText treinados
- Comparação GloVe/FastText/Word2Vec
- Análise de trade-offs
- Gráficos comparativos

---

## Tarefa 6: Classificação Avançada com Embeddings

### Objetivo
Usar embeddings em arquiteturas mais sofisticadas (CNN, RNN).

### Código Inicial:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.tokenizer import Tokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 1. Preparar dados para Keras
tokenizer_keras = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer_keras.fit_on_texts(X_train)

X_train_seq = tokenizer_keras.texts_to_sequences(X_train)
X_test_seq = tokenizer_keras.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=100, padding='post')

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"X_train_padded shape: {X_train_padded.shape}")
print(f"Vocabulário: {len(tokenizer_keras.word_index)}")

# 2. Modelo CNN com Embeddings
print("\n" + "="*70)
print("CNN COM EMBEDDINGS")
print("="*70)

model_cnn = Sequential([
    Embedding(input_dim=5001, output_dim=100, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()

history_cnn = model_cnn.fit(
    X_train_padded, tf.keras.utils.to_categorical(y_train_encoded),
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_cnn = np.argmax(model_cnn.predict(X_test_padded), axis=1)
acc_cnn = accuracy_score(y_test_encoded, y_pred_cnn)
print(f"\nCNN Accuracy: {acc_cnn:.4f}")

# 3. Modelo LSTM com Embeddings
print("\n" + "="*70)
print("LSTM COM EMBEDDINGS")
print("="*70)

model_lstm = Sequential([
    Embedding(input_dim=5001, output_dim=100, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_lstm = model_lstm.fit(
    X_train_padded, tf.keras.utils.to_categorical(y_train_encoded),
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_lstm = np.argmax(model_lstm.predict(X_test_padded), axis=1)
acc_lstm = accuracy_score(y_test_encoded, y_pred_lstm)
print(f"\nLSTM Accuracy: {acc_lstm:.4f}")

# 4. Visualizar histórico
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_cnn.history['accuracy'], label='Train CNN')
axes[0].plot(history_cnn.history['val_accuracy'], label='Val CNN')
axes[0].plot(history_lstm.history['accuracy'], label='Train LSTM')
axes[0].plot(history_lstm.history['val_accuracy'], label='Val LSTM')
axes[0].set_title('Accuracy ao longo do tempo')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Comparação final
modelos_final = ['CNN', 'LSTM', 'W2V + LogReg']
accs_final = [acc_cnn, acc_lstm, accuracy_score(y_test, y_pred_w2v)]
axes[1].bar(modelos_final, accs_final, color=['lightblue', 'lightgreen', 'coral'], edgecolor='navy')
axes[1].set_title('Comparação Final de Modelos')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
for i, (modelo, acc) in enumerate(zip(modelos_final, accs_final)):
    axes[1].text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('modelos_avancados.png', dpi=150)
print("\n✓ Gráfico salvo: modelos_avancados.png")
plt.show()
```

### Entregáveis:
- Modelos CNN e LSTM treinados
- Comparação com métodos anteriores
- Análise de performance
- Históricos de treinamento

---

## Tarefa 7: Análise Comparativa e Interpretabilidade

### Objetivo
Analisar todos os modelos, interpretar resultados e comunicar conclusões.

### Código Inicial:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SUMÁRIO COMPARATIVO
print("="*70)
print("SUMÁRIO COMPARATIVO DE TODOS OS MODELOS")
print("="*70)

resultados_completos = {
    'TF-IDF + LogReg': {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr, average='weighted'),
        'model': 'TF-IDF'
    },
    'TF-IDF + RF': {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf, average='weighted'),
        'model': 'TF-IDF'
    },
    'Word2Vec': {
        'accuracy': accuracy_score(y_test, y_pred_w2v),
        'f1': f1_score(y_test, y_pred_w2v, average='weighted'),
        'model': 'Embedding'
    },
    'FastText': {
        'accuracy': accuracy_score(y_test, y_pred_ft),
        'f1': f1_score(y_test, y_pred_ft, average='weighted'),
        'model': 'Embedding'
    },
    'CNN': {
        'accuracy': acc_cnn,
        'f1': f1_score(y_test_encoded, y_pred_cnn, average='weighted'),
        'model': 'Deep Learning'
    },
    'LSTM': {
        'accuracy': acc_lstm,
        'f1': f1_score(y_test_encoded, y_pred_lstm, average='weighted'),
        'model': 'Deep Learning'
    }
}

df_resultados = pd.DataFrame(resultados_completos).T
df_resultados = df_resultados.sort_values('accuracy', ascending=False)

print("\nRESULTADOS ORDENADOS POR ACCURACY:")
print(df_resultados.to_string())

# 2. ANÁLISE DE ERROS
print("\n\n" + "="*70)
print("ANÁLISE DE ERROS - WORD2VEC")
print("="*70)

erros_w2v = y_test[y_pred_w2v != y_test]
indices_erros = np.where(y_pred_w2v != y_test)[0]

print(f"\nTotal de erros: {len(erros_w2v)}")
print(f"Taxa de erro: {len(erros_w2v)/len(y_test)*100:.2f}%")

# Exemplos de erros
print(f"\nExemplos de erros:")
for i in indices_erros[:5]:
    print(f"\nTexto: {X_test.iloc[i][:100]}...")
    print(f"Verdadeiro: {y_test.iloc[i]}, Predito: {y_pred_w2v[i]}")

# 3. ANÁLISE DE INTERPRETABILIDADE
print("\n\n" + "="*70)
print("INTERPRETABILIDADE - FEATURES IMPORTANTES (TF-IDF)")
print("="*70)

feature_names = tfidf.get_feature_names_out()
coef = lr_model.coef_[0]

top_features_positive = np.argsort(coef)[-10:][::-1]
top_features_negative = np.argsort(coef)[:10]

print("\nFeatures MAIS POSITIVAS:")
for idx in top_features_positive:
    print(f"  {feature_names[idx]:20} : {coef[idx]:+.6f}")

print("\nFeatures MAIS NEGATIVAS:")
for idx in top_features_negative:
    print(f"  {feature_names[idx]:20} : {coef[idx]:+.6f}")

# 4. VISUALIZAÇÕES FINAIS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Comparação de Accuracy
models_list = list(resultados_completos.keys())
accuracies_list = [resultados_completos[m]['accuracy'] for m in models_list]
colors = ['steelblue' if resultados_completos[m]['model'] == 'TF-IDF' else 
          'coral' if resultados_completos[m]['model'] == 'Embedding' else 'lightgreen'
          for m in models_list]

axes[0, 0].barh(models_list, accuracies_list, color=colors, edgecolor='navy')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Comparação de Modelos - Accuracy')
axes[0, 0].set_xlim([0, 1])

# F1-Score
f1_list = [resultados_completos[m]['f1'] for m in models_list]
axes[0, 1].barh(models_list, f1_list, color=colors, edgecolor='navy')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('Comparação de Modelos - F1-Score')
axes[0, 1].set_xlim([0, 1])

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_w2v)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Confusion Matrix - Word2Vec')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# Top features
top_k = 10
top_features_idx = np.argsort(np.abs(coef))[-top_k:][::-1]
top_features_names = feature_names[top_features_idx]
top_features_coef = coef[top_features_idx]

axes[1, 1].barh(range(len(top_features_names)), top_features_coef,
                color=['green' if x > 0 else 'red' for x in top_features_coef],
                edgecolor='navy')
axes[1, 1].set_yticks(range(len(top_features_names)))
axes[1, 1].set_yticklabels(top_features_names)
axes[1, 1].set_xlabel('Coefficient')
axes[1, 1].set_title('Top Features - TF-IDF + LogReg')

plt.tight_layout()
plt.savefig('analise_completa.png', dpi=150)
print("\n✓ Gráfico salvo: analise_completa.png")
plt.show()

# 5. SUMÁRIO FINAL
print("\n\n" + "="*70)
print("CONCLUSÕES")
print("="*70)

print(f"""
1. MELHOR MODELO: {df_resultados.index[0]} com accuracy {df_resultados.iloc[0]['accuracy']:.4f}

2. COMPARAÇÃO DE ABORDAGENS:
   - TF-IDF: Simples, rápido, interpretável, baseado em frequência
   - Word Embeddings: Capturam semântica, requerem mais dados
   - Deep Learning: Mais poderoso mas requer mais dados e computação

3. TRADE-OFFS:
   - Performance vs Interpretabilidade: TF-IDF é mais interpretável
   - Performance vs Velocidade: Embeddings simples são rápidos
   - Performance vs Complexidade: Deep Learning é complexo

4. RECOMENDAÇÃO: {df_resultados.index[0]}
""")
```

### Entregáveis:
- Tabela comparativa completa
- Análise de erros
- Interpretabilidade de features
- Visualizações finais
- Relatório com conclusões

---

## Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|-----------|
| Exploração e Preparação | 15% | Qualidade da análise exploratória e pré-processamento |
| Implementação de Modelos | 30% | Corretude e completude dos modelos |
| Comparação e Análise | 25% | Qualidade da análise comparativa |
| Visualizações | 15% | Qualidade dos gráficos e interpretabilidade |
| Documentação e Relatório | 15% | Clareza, organização e conclusões |

---

## Dicas de Implementação

✓ Use dataset equilibrado quando possível  
✓ Sempre normalize features antes de usar embeddings  
✓ Word2Vec/FastText requerem >10k palavras para bons resultados  
✓ Experimental: tente diferentes tamanhos de janela (window)  
✓ Para português: FastText é geralmente melhor que Word2Vec  
✓ Salve modelos treinados para reutilização  
✓ Use visualizações para comunicar resultados  

---

**Data de Entrega:** A definir conforme cronograma  
**Formato:** Relatório PDF + Código Python + Visualizações + Modelos Salvos

Bom trabalho! 🚀

