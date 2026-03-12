# Tarefas Práticas — Aula 01: TF e TF-IDF com Poemas Portugueses

**Disciplina:** Processamento de Linguagem Natural  
**Aula:** 01 - TF e TF-IDF  
**Dataset:** `DATA/portuguese_poems.csv`  
**Objetivo:** Aplicar técnicas de vetorização de texto em um corpus de poemas de língua portuguesa.

---

## 📚 Descrição da Tarefa

### Contexto Geral
Este conjunto de tarefas visa consolidar os conceitos fundamentais de **Processamento de Linguagem Natural (PLN)** através da aplicação prática de técnicas de vetorização de texto. Utilizando um corpus de poemas portugueses (com foco em Fernando Pessoa e seus heterônimos), você aprenderá a transformar texto não-estruturado em representações numéricas que algoritmos de machine learning possam processar.

### Importância de Cada Tarefa

#### **Tarefa 1 - Preparação e Análise Exploratória**
- **Por que é importante:** A análise exploratória é o primeiro passo em qualquer projeto de dados. Compreender a estrutura, distribuição e características do corpus permite identificar padrões, desafios e oportunidades.
- **Aplicação prática:** Em projetos reais, você precisará explorar dados antes de aplicar qualquer modelo. Esta tarefa ensina a fazer perguntas corretas sobre seus dados.
- **Habilidades desenvolvidas:** Manipulação de DataFrames, visualização de dados, pensamento analítico.

#### **Tarefa 2 - Pré-processamento de Texto em Português**
- **Por que é importante:** Textos brutos contêm muito "ruído" (pontuação, números, palavras comuns) que não contribuem para a análise semântica. O pré-processamento garante que apenas informação relevante seja vetorizada.
- **Aplicação prática:** Todo pipeline de PLN começa com pré-processamento. A qualidade desta etapa impacta diretamente nos resultados finais.
- **Habilidades desenvolvidas:** Limpeza de dados, expressões regulares, tratamento de idiomas específicos (português).

#### **Tarefa 3 - Vetorização com TF (CountVectorizer)**
- **Por que é importante:** TF (Term Frequency) é a base para entender a frequência de termos em documentos. É o primeiro passo para transformar texto em números que algoritmos possam processar.
- **Aplicação prática:** TF é usado em motores de busca, classificação de documentos e análise de similaridade. É um conceito fundamental.
- **Habilidades desenvolvidas:** Compreensão de vetorização, análise de frequência, trabalho com matrizes esparsas.

#### **Tarefa 4 - Vetorização com TF-IDF**
- **Por que é importante:** TF-IDF é superior ao TF puro porque considera não apenas a frequência de um termo em um documento, mas também seu rarity (IDF). Identifica palavras verdadeiramente importantes.
- **Aplicação prática:** TF-IDF é usado em sistemas de recomendação, busca por similaridade e análise de documentos. É o padrão da indústria.
- **Habilidades desenvolvidas:** Conceitos de importância relativa, normalização, compreensão matemática de vetores.

#### **Tarefa 5 - Análise Visual com Word Clouds**
- **Por que é importante:** Visualizações ajudam a comunicar insights de forma clara e intuitiva. Word clouds permitem identificar rapidamente os termos mais importantes visualmente.
- **Aplicação prática:** Apresentações para stakeholders, relatórios, comunicação de resultados. Dados são inúteis se não puderem ser comunicados.
- **Habilidades desenvolvidas:** Visualização de dados, comunicação de resultados, pensamento crítico.

#### **Tarefa 6 - Análise de Similaridade entre Poemas (Desafio)**
- **Por que é importante:** Medir similaridade entre textos é fundamental para muitas aplicações (recomendação, clustering, busca). A similaridade de cosseno é a métrica mais comum.
- **Aplicação prática:** Sistemas de recomendação (Netflix, Amazon), busca de documentos similares, agrupamento de textos.
- **Habilidades desenvolvidas:** Cálculo de distâncias, interpretação de métricas, análise de resultados.

#### **Tarefa 7 - Relatório Final**
- **Por que é importante:** A capacidade de sintetizar aprendizados e comunicar resultados é tão importante quanto os resultados em si. Um bom relatório demonstra compreensão profunda.
- **Aplicação prática:** Projetos reais exigem documentação clara. Você deve ser capaz de justificar suas escolhas metodológicas.
- **Habilidades desenvolvidas:** Redação técnica, síntese, apresentação de resultados, reflexão crítica.

### Fluxo de Aprendizado

```
Exploração → Limpeza → TF → TF-IDF → Visualização → Análise → Relatório
   (1)         (2)      (3)    (4)        (5)         (6)        (7)
```

Cada tarefa constrói sobre a anterior, criando uma compreensão progressiva de como transformar texto em insights.

### Competências ao Final

Após completar todas as 7 tarefas, você será capaz de:
- ✅ Explorar e compreender dados de texto
- ✅ Pré-processar textos em português
- ✅ Vetorizar texto usando TF e TF-IDF
- ✅ Analisar e visualizar dados de texto
- ✅ Medir similaridade entre documentos
- ✅ Comunicar resultados de forma clara

---

## Tarefa 1: Preparação e Análise Exploratória

### Objetivo
Entender a estrutura do dataset de poemas e explorar os autores disponíveis.

### Código Inicial:

```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 1. Carregar o dataset
df = pd.read_csv('DATA/portuguese_poems.csv', encoding='utf-8')

# 2. Explorar estrutura
print(f"Shape: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
print(df.head(10))

# 3. Listar todos os autores
autores_unicos = df['author'].unique()
print(f"Total de autores: {len(autores_unicos)}")
print(autores_unicos[:20])

# 4. Contar poemas por autor
autor_counts = df['author'].value_counts()
print(autor_counts.head(15))

# 5. Visualizar distribuição
plt.figure(figsize=(12, 8))
autor_counts.head(15).plot(kind='barh')
plt.xlabel('Número de Poemas')
plt.title('Top 15 Autores com Mais Poemas')
plt.tight_layout()
plt.savefig('autores_distribution.png', dpi=150)
plt.show()

# 6. Filtrar heterônimos de Fernando Pessoa
heteronimos = ['Fernando Pessoa', 'Alberto Caeiro', 'Álvaro de Campos', 'Ricardo Reis']
df_pessoa = df[df['author'].isin(heteronimos)]
print(f"\nPoemas de Fernando Pessoa e heterônimos: {len(df_pessoa)}")
print(df_pessoa['author'].value_counts())
```

### Entregáveis:
- Código comentado com análise exploratória
- Gráfico de distribuição de autores
- Número de poemas por heterônimo

---

## Tarefa 2: Pré-processamento de Texto em Português

### Objetivo  
Limpar e preparar o texto dos poemas para vetorização.

### Código Inicial:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixar recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Carregar stopwords português
stop_words = set(stopwords.words('portuguese'))

# 1. Limpeza de pontuação
df_pessoa['texto_limpo'] = df_pessoa['text'].str.lower()
df_pessoa['texto_limpo'] = df_pessoa['texto_limpo'].str.replace(
    r'[,.:;!?\'"-()\\[\\]{}]', ' ', regex=True)
df_pessoa['texto_limpo'] = df_pessoa['texto_limpo'].str.replace(
    r'\\s+', ' ', regex=True)

# 2. Remover números
df_pessoa['texto_sem_numeros'] = df_pessoa['texto_limpo'].str.replace(
    r'[0-9]+', '', regex=True)

# 3. Tokenização e remoção de stopwords
def remover_stopwords(texto):
    tokens = word_tokenize(texto, language='portuguese')
    return [token for token in tokens if token not in stop_words]

df_pessoa['tokens'] = df_pessoa['texto_sem_numeros'].apply(remover_stopwords)
df_pessoa['num_tokens'] = df_pessoa['tokens'].apply(len)

# 4. Comparação de impacto
tokens_antes = df_pessoa['texto_sem_numeros'].apply(
    lambda x: len(word_tokenize(x, language='portuguese'))).mean()
tokens_depois = df_pessoa['num_tokens'].mean()

print(f"Média de tokens ANTES remover stopwords: {tokens_antes:.2f}")
print(f"Média de tokens APÓS remover stopwords: {tokens_depois:.2f}")
print(f"Redução: {((tokens_antes - tokens_depois) / tokens_antes * 100):.2f}%")

# 5. Visualizar distribuição
plt.figure(figsize=(10, 6))
plt.hist(df_pessoa['num_tokens'], bins=30, edgecolor='black', color='skyblue')
plt.xlabel('Número de Tokens')
plt.ylabel('Frequência')
plt.title('Distribuição de Tokens por Poema (após limpeza)')
plt.savefig('tokens_distribution.png', dpi=150)
plt.show()
```

### Entregáveis:
- Código com funções de limpeza
- DataFrame com colunas de texto em diferentes estágios
- Gráfico de distribuição de tokens

---

## Tarefa 3: Vetorização com TF (CountVectorizer)

### Objetivo
Gerar matriz de frequência de termos usando CountVectorizer.

### Código Inicial:

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 1. Preparar textos limpos como strings
textos = df_pessoa['texto_sem_numeros'].values

# 2. CountVectorizer com unigramas
vectorizer_tf = CountVectorizer(
    lowercase=True,
    stop_words=list(stop_words),
    ngram_range=(1, 1),
    max_features=500
)

# 3. Treinar e transformar
X_tf = vectorizer_tf.fit_transform(textos)
feature_names = vectorizer_tf.get_feature_names_out()

# 4. Calcular frequências
term_freq = np.asarray(X_tf.sum(axis=0)).flatten()
term_freq_sorted = sorted(zip(feature_names, term_freq), 
                         key=lambda x: x[1], reverse=True)

# 5. Top 20 termos mais frequentes
print("=== Top 20 Termos Mais Frequentes (Unigramas) ===")
for i, (termo, freq) in enumerate(term_freq_sorted[:20], 1):
    print(f"{i:2d}. {termo:15s} : {int(freq):4d}")

# 6. Visualizar
top_20 = term_freq_sorted[:20]
termos = [t[0] for t in top_20]
freqs = [t[1] for t in top_20]

plt.figure(figsize=(12, 8))
plt.barh(termos, freqs, color='teal')
plt.xlabel('Frequência')
plt.title('Top 20 Termos Mais Frequentes (TF)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_terms_tf_unigrams.png', dpi=150)
plt.show()

# 7. Bigramas
print("\n=== Bigramas Mais Frequentes ===")
vectorizer_bigram = CountVectorizer(
    lowercase=True,
    stop_words=list(stop_words),
    ngram_range=(2, 2),
    max_features=100
)
X_bigram = vectorizer_bigram.fit_transform(textos)
bigram_names = vectorizer_bigram.get_feature_names_out()
bigram_freq = np.asarray(X_bigram.sum(axis=0)).flatten()
bigram_sorted = sorted(zip(bigram_names, bigram_freq), 
                      key=lambda x: x[1], reverse=True)

for i, (bigrama, freq) in enumerate(bigram_sorted[:10], 1):
    print(f"{i:2d}. {bigrama:25s} : {int(freq):3d}")
```

### Entregáveis:
- Matriz TF (sparse matrix)
- Lista de top 20 unigramas e top 10 bigramas
- Gráficos de frequência

---

## Tarefa 4: Vetorização com TF-IDF

### Objetivo
Identificar termos discriminativos usando TF-IDF.

### Código Inicial:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words=list(stop_words),
    ngram_range=(1, 1),
    max_features=500,
    norm='l2'
)

# 2. Treinar e transformar
X_tfidf = vectorizer_tfidf.fit_transform(textos)
tfidf_names = vectorizer_tfidf.get_feature_names_out()

# 3. Calcular scores TF-IDF globais
tfidf_scores = np.asarray(X_tfidf.sum(axis=0)).flatten()
tfidf_sorted = sorted(zip(tfidf_names, tfidf_scores), 
                     key=lambda x: x[1], reverse=True)

print("=== Top 15 Termos por TF-IDF (Global) ===")
for i, (termo, score) in enumerate(tfidf_sorted[:15], 1):
    print(f"{i:2d}. {termo:15s} : {score:.6f}")

# 4. Comparar TF vs TF-IDF por autor
print("\n=== Comparação TF vs TF-IDF por Autor ===")
for autor in heteronimos:
    textos_autor = df_pessoa[df_pessoa['author'] == autor]['texto_sem_numeros'].values
    
    if len(textos_autor) > 0:
        # TF
        vect_tf_autor = CountVectorizer(
            lowercase=True, 
            stop_words=list(stop_words), 
            max_features=15
        )
        X_tf_autor = vect_tf_autor.fit_transform(textos_autor)
        tf_nomes = vect_tf_autor.get_feature_names_out()
        tf_freq = np.asarray(X_tf_autor.sum(axis=0)).flatten()
        tf_sorted = sorted(zip(tf_nomes, tf_freq), key=lambda x: x[1], reverse=True)
        
        # TF-IDF
        vect_tfidf_autor = TfidfVectorizer(
            lowercase=True, 
            stop_words=list(stop_words), 
            max_features=15,
            norm='l2'
        )
        X_tfidf_autor = vect_tfidf_autor.fit_transform(textos_autor)
        tfidf_nomes = vect_tfidf_autor.get_feature_names_out()
        tfidf_scores_autor = np.asarray(X_tfidf_autor.sum(axis=0)).flatten()
        tfidf_sorted_autor = sorted(zip(tfidf_nomes, tfidf_scores_autor), 
                                   key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*50}")
        print(f"AUTOR: {autor}")
        print(f"{'='*50}")
        print(f"{'Top 10 por TF':<35} | {'Top 10 por TF-IDF':<35}")
        print("-" * 72)
        for j in range(10):
            tf_item = f"{j+1}. {tf_sorted[j][0]}" if j < len(tf_sorted) else ""
            tfidf_item = f"{j+1}. {tfidf_sorted_autor[j][0]}" if j < len(tfidf_sorted_autor) else ""
            print(f"{tf_item:<35} | {tfidf_item:<35}")

# 5. Testar normalizações L1 vs L2
print("\n=== Comparação L1 vs L2 ===")
vect_l1 = TfidfVectorizer(norm='l1', max_features=100, lowercase=True, stop_words=list(stop_words))
vect_l2 = TfidfVectorizer(norm='l2', max_features=100, lowercase=True, stop_words=list(stop_words))

X_l1 = vect_l1.fit_transform(textos)
X_l2 = vect_l2.fit_transform(textos)

print(f"Norma L1 (Manhattan) - Soma de um vetor: {X_l1[0].sum():.6f}")
print(f"Norma L2 (Euclidiana) - Norma de um vetor: {np.sqrt((X_l2[0].data**2).sum()):.6f}")
```

### Entregáveis:
- Matriz TF-IDF
- Tabela comparativa de top-15 termos por autor
- Análise de normalização L1 vs L2

---

## Tarefa 5: Visualização com Word Clouds

### Objetivo
Criar visualizações intuitivas do vocabulário usando word clouds.

### Código Inicial:

```python
from wordcloud import WordCloud

# 1. Word Cloud por Autor
print("Gerando word clouds por autor...")
for autor in heteronimos:
    textos_autor = ' '.join(df_pessoa[df_pessoa['author'] == autor]['texto_sem_numeros'].values)
    
    if len(textos_autor) > 50:  # Apenas se houver texto suficiente
        wc = WordCloud(
            width=1000, 
            height=600, 
            background_color='white',
            colormap='viridis',
            prefer_horizontal=0.7,
            random_state=42
        ).generate(textos_autor)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'Word Cloud - {autor}', fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        filename = f'wordcloud_{autor.replace(" ", "_").lower()}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Salvo: {filename}")
        plt.show()

# 2. Word Cloud TF vs TF-IDF para um autor
autor_escolhido = 'Álvaro de Campos'
textos_escolhido = ' '.join(
    df_pessoa[df_pessoa['author'] == autor_escolhido]['texto_sem_numeros'].values
)

if len(textos_escolhido) > 50:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # TF
    wc_tf = WordCloud(width=800, height=400, background_color='white', 
                      colormap='Blues', random_state=42).generate(textos_escolhido)
    axes[0].imshow(wc_tf, interpolation='bilinear')
    axes[0].set_title(f'TF - {autor_escolhido}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # TF-IDF
    wc_tfidf = WordCloud(width=800, height=400, background_color='white', 
                         colormap='Reds', random_state=42).generate(textos_escolhido)
    axes[1].imshow(wc_tfidf, interpolation='bilinear')
    axes[1].set_title(f'TF-IDF - {autor_escolhido}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('wordcloud_tf_vs_tfidf_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Salvo: wordcloud_tf_vs_tfidf_comparison.png")
    plt.show()
```

### Entregáveis:
- 5-6 imagens de word clouds em alta resolução
- Código para geração automática

---

## Tarefa 6: Análise de Similaridade entre Poemas (Desafio)

### Objetivo
Encontrar poemas similares usando similaridade de cosseno.

### Código Inicial:

```python
from sklearn.metrics.pairwise import cosine_similarity

# 1. Calcular matriz de similaridade
similarity_matrix = cosine_similarity(X_tfidf)

# 2. Selecionar um poema de referência
idx_campos = (df_pessoa['author'] == 'Álvaro de Campos').values
indices_campos = np.where(idx_campos)[0]

if len(indices_campos) > 0:
    idx_referencia = indices_campos[0]
    similares = similarity_matrix[idx_referencia].argsort()[::-1][1:6]
    
    print(f"\n{'='*70}")
    print(f"POEMA DE REFERÊNCIA: {df_pessoa.iloc[idx_referencia]['author']}")
    print(f"{'='*70}")
    print(f"{df_pessoa.iloc[idx_referencia]['text'][:200]}...\n")
    
    print("POEMAS MAIS SIMILARES:")
    for i, idx_similar in enumerate(similares, 1):
        sim_score = similarity_matrix[idx_referencia, idx_similar]
        autor = df_pessoa.iloc[idx_similar]['author']
        texto = df_pessoa.iloc[idx_similar]['text'][:150]
        print(f"\n{i}. [{sim_score:.4f}] {autor}")
        print(f"   {texto}...")

# 3. Comparar com outro autor (Ricardo Reis)
idx_reis = (df_pessoa['author'] == 'Ricardo Reis').values
indices_reis = np.where(idx_reis)[0]

if len(indices_reis) > 0:
    idx_ref_reis = indices_reis[0]
    similares_reis = similarity_matrix[idx_ref_reis].argsort()[::-1][1:6]
    
    print(f"\n{'='*70}")
    print(f"POEMA DE REFERÊNCIA: Ricardo Reis")
    print(f"{'='*70}")
    print(f"{df_pessoa.iloc[idx_ref_reis]['text'][:200]}...\n")
    
    print("POEMAS MAIS SIMILARES:")
    for i, idx_sim in enumerate(similares_reis, 1):
        sim_score = similarity_matrix[idx_ref_reis, idx_sim]
        autor = df_pessoa.iloc[idx_sim]['author']
        print(f"{i}. [{sim_score:.4f}] {autor}")
```

### Entregáveis:
- Código de cálculo de similaridade
- Análise de 2 poemas de referência com vizinhos mais próximos
- Conclusões sobre espaços semânticos por autor

---

## Tarefa 7: Relatório Final

### Estrutura do Relatório:

1. **Introdução (5-10 linhas)**
   - Contexto do dataset
   - Objetivos principais

2. **Metodologia (10-15 linhas)**
   - Técnicas: TF, TF-IDF, normalização
   - Pré-processamento aplicado

3. **Resultados (20-30 linhas)**
   - Estatísticas principais
   - Termos mais relevantes por autor
   - Padrões descobertos

4. **Discussão (15-20 linhas)**
   - Interpretação dos resultados
   - Diferenças entre autores
   - Limitações da abordagem

5. **Conclusão (5-10 linhas)**
   - Aprendizados principais

### Código para Gerar Sumário:

```python
# Sumário dos resultados
print("\n" + "="*70)
print("SUMÁRIO DA ANÁLISE")
print("="*70)

print(f"\n1. DATASET:")
print(f"   • Total de poemas: {len(df_pessoa)}")
print(f"   • Autores analisados: {heteronimos}")
print(f"   • Distribuição: {dict(df_pessoa['author'].value_counts())}")

print(f"\n2. VOCABULÁRIO:")
print(f"   • Termos únicos totais: {len(feature_names)}")
print(f"   • Termo mais frequente: '{term_freq_sorted[0][0]}' ({int(term_freq_sorted[0][1])} ocorrências)")
print(f"   • Comprimento médio de poema: {df_pessoa['num_tokens'].mean():.0f} tokens")

print(f"\n3. TERMOS DISCRIMINATIVOS (TF-IDF):")
print(f"   • Top 3 globais: {', '.join([t[0] for t in tfidf_sorted[:3]])}")

print(f"\n4. BIGRAMAS:")
print(f"   • Bigrama mais frequente: '{bigram_sorted[0][0]}' ({int(bigram_sorted[0][1])} ocorrências)")

print("\n" + "="*70 + "\n")
```

### Entregáveis:
- Relatório em PDF (8-10 páginas)
- Código Python comentado (`analise_poemas_class01.py`)
- Todas as visualizações (gráficos e word clouds)

---

## Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|-----------|
| Exploração e Limpeza | 15% | Qualidade do pré-processamento |
| Implementação TF-IDF | 25% | Corretude e completude |
| Análise e Interpretação | 30% | Profundidade das conclusões |
| Visualizações | 15% | Qualidade dos gráficos e word clouds |
| Documentação e Código | 15% | Clareza e organização |

---

## Dicas de Implementação

✓ Use `encoding='utf-8'` com textos em português  
✓ Customize stopwords conforme necessário  
✓ Salve imagens em alta resolução (dpi=150+)  
✓ Comente seu código explicando cada passo  
✓ Experimente diferentes `max_features` e normalizações  

---

**Data de Entrega:** A definir conforme cronograma  
**Formato:** Relatório PDF + Código Python + Visualizações

Bom trabalho! 🎉

