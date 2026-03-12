# Tarefas Práticas — Aula 02: Processamento com NLTK e spaCy

**Disciplina:** Processamento de Linguagem Natural  
**Aula:** 02 - NLTK e spaCy (Pipelines de PLN)  
**Dataset:** `DATA/portuguese_poems.csv` (Poemas em Português)  
**Objetivo:** Aplicar técnicas de processamento de linguagem natural usando NLTK e spaCy em um corpus de poemas portugueses.

---

## 📚 Descrição da Tarefa

### Contexto Geral
Este conjunto de tarefas explora os **pipelines de processamento de linguagem natural**, focando nas duas principais bibliotecas de PLN em Python: **NLTK** e **spaCy**. Você aprenderá a decompor textos em seus componentes estruturais (tokens, partes do discurso, entidades, dependências) e a comparar as abordagens diferentes dessas ferramentas. O corpus utilizado (poemas portugueses) fornece um contexto real e culturalmente relevante para a aprendizagem.

### Importância de Cada Tarefa

#### **Tarefa 1 - Tokenização e Análise de Estrutura de Texto**
- **Por que é importante:** A tokenização é o primeiro passo em qualquer pipeline de PLN. Diferentes abordagens de tokenização afetam todas as análises posteriores. Compreender as diferenças entre NLTK e spaCy é essencial para escolher a ferramenta certa.
- **Aplicação prática:** Diferentes idiomas e domínios exigem diferentes estratégias de tokenização. Textos poéticos, por exemplo, têm desafios únicos.
- **Habilidades desenvolvidas:** Compreensão de tokenização, conhecimento de duas bibliotecas principais, análise comparativa.

#### **Tarefa 2 - Limpeza e Normalização (Stopwords, Stemming, Lematização)**
- **Por que é importante:** Normalização é crucial para garantir que variações de uma mesma palavra sejam tratadas como equivalentes. Stemming e Lematização são duas abordagens diferentes com trade-offs.
- **Aplicação prática:** A qualidade da normalização impacta a acurácia de classificadores, agrupadores e sistemas de recomendação. Escolher stemming ou lematização depende da aplicação.
- **Habilidades desenvolvidas:** Compreensão de morphologia, trade-offs entre abordagens, otimização para a tarefa.

#### **Tarefa 3 - POS-Tagging (Part-of-Speech)**
- **Por que é importante:** Identificar partes do discurso permite análises sintáticas mais profundas e é base para muitas tarefas de PLN (análise de sentimento, extração de informação, etc.).
- **Aplicação prática:** Sistemas de análise sintática, correção gramatical, análise automática de qualidade textual, geração de resumos.
- **Habilidades desenvolvidas:** Compreensão de estruturas linguísticas, trabalho com modelos pré-treinados, interpretação de tags linguísticas.

#### **Tarefa 4 - Reconhecimento de Entidades Nomeadas (NER)**
- **Por que é importante:** NER é fundamental para extração de informação estruturada a partir de texto. É usado em praticamente todas as aplicações de PLN que precisam entender "quem" e "o quê" em um texto.
- **Aplicação prática:** Análise de notícias, extração de nomes de pessoas/lugares/organizações, análise de redes sociais, sistemas de busca.
- **Habilidades desenvolvidas:** Classificação multilabel, trabalho com entidades, extração de informação, análise de cobertura de modelos.

#### **Tarefa 5 - Análise de Dependências Sintáticas**
- **Por que é importante:** Análise sintática permite entender as relações entre palavras em uma sentença. É essencial para tarefas que exigem compreensão profunda do significado.
- **Aplicação prática:** Análise de sentimentos avançada, tradução automática, geração de texto, resposta a perguntas, entendimento automático de documentos.
- **Habilidades desenvolvidas:** Compreensão de estruturas sintáticas, visualização de dados linguísticos, análise de complexidade linguística.

#### **Tarefa 6 - Pipeline Completo (NLTK vs spaCy) e Análise Comparativa**
- **Por que é importante:** Na prática, você raramente usará apenas um componente. Você precisa entender como construir pipelines completos e ser capaz de escolher entre NLTK (flexibilidade/educacional) e spaCy (performance/produção).
- **Aplicação prática:** Decisões tecnológicas em projetos reais dependem de compreender trade-offs: velocidade vs. flexibilidade, facilidade vs. profundidade.
- **Habilidades desenvolvidas:** Engenharia de software, profiling de performance, tomada de decisão técnica, análise crítica.

#### **Tarefa 7 - Análise Completa e Relatório**
- **Por que é importante:** Consolidar tudo que foi aprendido em uma análise real demonstra compreensão prática e capacidade de comunicação técnica. Mostrar padrões no corpus real é muito mais valioso que código isolado.
- **Aplicação prática:** Projetos de PLN reais envolvem entender padrões no corpus específico. Diferentes autores/domínios têm características diferentes que afetam a escolha de técnicas.
- **Habilidades desenvolvidas:** Análise de dados, síntese de informação, visualização avançada, redação técnica, pensamento crítico.

### Fluxo de Aprendizado

```
Tokenização → Normalização → POS-Tag → NER → Sintaxe → Pipelines → Análise Completa
    (1)            (2)          (3)      (4)     (5)        (6)           (7)
```

Cada tarefa constrói um entendimento mais profundo dos componentes de um pipeline de PLN real.

### Comparação NLTK vs spaCy

| Aspecto | NLTK | spaCy |
|---------|------|-------|
| **Curva de Aprendizado** | Suave | Média |
| **Velocidade** | Lenta | Rápida |
| **Pipeline Integrado** | Não | Sim |
| **Produção** | Não (educacional) | Sim |
| **Flexibilidade** | Alta | Média |
| **Documentação** | Excelente | Muito boa |

Neste curso você aprenderá ambas!

### Competências ao Final

Após completar todas as 7 tarefas, você será capaz de:
- ✅ Tokenizar textos corretamente em português
- ✅ Aplicar stemming e lematização apropriadamente
- ✅ Identificar e extrair partes do discurso
- ✅ Reconhecer entidades nomeadas em textos
- ✅ Analisar estruturas sintáticas
- ✅ Escolher a ferramenta certa para cada tarefa (NLTK vs spaCy)
- ✅ Implementar pipelines completos de PLN
- ✅ Comunicar insights linguísticos de forma clara

---

## Tarefa 1: Tokenização e Análise de Estrutura de Texto

### Objetivo
Aprender a dividir textos em unidades menores (tokens e sentenças) usando NLTK e spaCy.

### Código Inicial:

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Baixar recursos
nltk.download('punkt')
nltk.download('punkt_tab')

# 1. Carregar dataset
df = pd.read_csv('DATA/portuguese_poems.csv', encoding='utf-8')

# 2. Selecionar um poema
poema_exemplo = df.iloc[0]['text']
print("POEMA ORIGINAL:")
print(poema_exemplo[:200] + "...\n")

# 3. Tokenização NLTK - por palavras
tokens_palavras = word_tokenize(poema_exemplo, language='portuguese')
print(f"Total de tokens (palavras): {len(tokens_palavras)}")
print(f"Primeiros 10 tokens: {tokens_palavras[:10]}\n")

# 4. Tokenização NLTK - por sentenças
tokens_sentenças = sent_tokenize(poema_exemplo, language='portuguese')
print(f"Total de sentenças: {len(tokens_sentenças)}")
print(f"Primeira sentença: {tokens_sentenças[0]}\n")

# 5. Tokenização spaCy
nlp = spacy.load('pt_core_news_sm')
doc = nlp(poema_exemplo)

print(f"Tokens spaCy: {len([token for token in doc])}")
print(f"Sentenças spaCy: {len(list(doc.sents))}\n")

# 6. Comparação
print("COMPARAÇÃO NLTK vs spaCy:")
print(f"  NLTK tokens: {len(tokens_palavras)}")
print(f"  spaCy tokens: {len([token for token in doc])}")
print(f"  Diferença: {abs(len(tokens_palavras) - len([token for token in doc]))}")

# 7. Explorar estrutura de tokens spaCy
print("\nESTRUTURA DE TOKENS (spaCy):")
for i, token in enumerate(doc[:10]):
    print(f"  {i+1}. '{token.text}' - tipo: {token.pos_}")
```

### Entregáveis:
- Código comentado com análise de tokenização
- Comparação NLTK vs spaCy
- Visualização de estrutura de tokens

---

## Tarefa 2: Limpeza e Normalização com Stopwords, Stemming e Lematização

### Objetivo
Aprender técnicas de normalização: remover stopwords, aplicar stemming e lematização.

### Código Inicial:

```python
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('stopwords')
nltk.download('rslp')

# 1. Carregar stopwords português
stop_words = set(stopwords.words('portuguese'))

# 2. Remover stopwords NLTK
tokens_limpos = [token.lower() for token in tokens_palavras 
                 if token.lower() not in stop_words and token.isalpha()]

print(f"Tokens originais: {len(tokens_palavras)}")
print(f"Tokens sem stopwords: {len(tokens_limpos)}")
print(f"Redução: {((len(tokens_palavras) - len(tokens_limpos)) / len(tokens_palavras) * 100):.1f}%\n")

print(f"Primeiros 20 tokens limpos: {tokens_limpos[:20]}\n")

# 3. Stemming - RSLP (Portuguese Stemmer)
stemmer = RSLPStemmer()
tokens_stemmed = [stemmer.stem(token) for token in tokens_limpos]

print("STEMMING - Exemplos de redução:")
exemplos = [
    'computador', 'computadores', 'computação',
    'correr', 'correndo', 'corrida',
    'belo', 'beleza', 'belíssimo'
]
for palavra in exemplos:
    stem = stemmer.stem(palavra)
    print(f"  {palavra:15} → {stem}")

# 4. Lematização spaCy
doc = nlp(poema_exemplo)
tokens_lematizados = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

print("\nLEMATIZAÇÃO spaCy - Exemplos:")
exemplos_doc = nlp("computador computadores computação correr correndo corrida belo beleza")
for token in exemplos_doc:
    if token.is_alpha:
        print(f"  {token.text:15} → {token.lemma_}")

# 5. Comparação Stemming vs Lematização
print("\nCOMPARAÇÃO STEMMING vs LEMATIZAÇÃO:")
palavras_teste = ['correndo', 'correria', 'correr', 'corrida']
for palavra in palavras_teste:
    stem = stemmer.stem(palavra)
    token_doc = nlp(palavra)[0]
    lemma = token_doc.lemma_
    print(f"  {palavra:12} | Stem: {stem:10} | Lemma: {lemma}")

# 6. Frequência dos tokens processados
from collections import Counter
freq_stems = Counter(tokens_stemmed)
print("\nTop 20 STEMS mais frequentes:")
for stem, freq in freq_stems.most_common(20):
    print(f"  {stem:15} : {freq:3d}")
```

### Entregáveis:
- Código com stemming e lematização
- Comparação entre técnicas
- Análise de frequência
- Gráfico de impacto de normalização

---

## Tarefa 3: POS-Tagging (Part-of-Speech) e Análise Sintática

### Objetivo
Identificar partes do discurso (substantivos, verbos, adjetivos, etc.) usando NLTK e spaCy.

### Código Inicial:

```python
import matplotlib.pyplot as plt
from collections import Counter

# 1. POS-Tagging NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

pos_tags_nltk = nltk.pos_tag(tokens_limpos, lang='pt')

print("POS-TAGGING NLTK - Primeiros 20 tokens:")
for i, (token, pos) in enumerate(pos_tags_nltk[:20]):
    print(f"  {i+1:2d}. {token:12} → {pos}")

# 2. POS-Tagging spaCy
doc = nlp(poema_exemplo)
print("\nPOS-TAGGING spaCy - Primeiros 20 tokens:")
for i, token in enumerate(list(doc)[:20]):
    if token.is_alpha:
        print(f"  {i+1:2d}. {token.text:12} → {token.pos_:5} ({token.tag_})")

# 3. Distribuição de POS tags
pos_dist_nltk = Counter([pos for token, pos in pos_tags_nltk])
print("\nDISTRIBUIÇÃO DE POS TAGS (NLTK):")
for pos, count in pos_dist_nltk.most_common(10):
    print(f"  {pos:10} : {count:3d}")

# 4. Distribuição de POS tags spaCy
pos_dist_spacy = Counter([token.pos_ for token in doc if token.is_alpha])
print("\nDISTRIBUIÇÃO DE POS TAGS (spaCy):")
for pos, count in pos_dist_spacy.most_common(10):
    print(f"  {pos:10} : {count:3d}")

# 5. Extrair palavras por tipo
print("\nPALAVRAS POR TIPO (spaCy):")
substantivos = [token.text for token in doc if token.pos_ == 'NOUN']
verbos = [token.text for token in doc if token.pos_ == 'VERB']
adjetivos = [token.text for token in doc if token.pos_ == 'ADJ']

print(f"  Substantivos ({len(substantivos)}): {substantivos[:10]}")
print(f"  Verbos ({len(verbos)}): {verbos[:10]}")
print(f"  Adjetivos ({len(adjetivos)}): {adjetivos[:10]}")

# 6. Visualizar distribuição
plt.figure(figsize=(12, 6))
pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
plt.bar(pos_counts.keys(), pos_counts.values(), color='skyblue', edgecolor='navy')
plt.xlabel('POS Tag')
plt.ylabel('Frequência')
plt.title('Distribuição de Partes do Discurso (POS Tags)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pos_distribution.png', dpi=150)
plt.show()
```

### Entregáveis:
- Código com POS-tagging NLTK e spaCy
- Comparação entre ferramentas
- Extração de palavras por tipo
- Gráfico de distribuição de POS tags

---

## Tarefa 4: Reconhecimento de Entidades Nomeadas (NER)

### Objetivo
Identificar e classificar entidades (pessoas, lugares, organizações, etc.) nos textos.

### Código Inicial:

```python
# 1. NER com spaCy
doc = nlp(poema_exemplo)

print("ENTIDADES IDENTIFICADAS (spaCy NER):")
if len(doc.ents) > 0:
    for i, ent in enumerate(doc.ents, 1):
        print(f"  {i}. '{ent.text}' → Tipo: {ent.label_} (Confiança: {ent.start_char}-{ent.end_char})")
else:
    print("  Nenhuma entidade nomeada encontrada neste poema.")

# 2. Distribuição de tipos de entidades
from collections import Counter
ent_types = Counter([ent.label_ for ent in doc.ents])
print(f"\nDISTRIBUIÇÃO DE ENTIDADES:")
for ent_type, count in ent_types.most_common():
    print(f"  {ent_type:10} : {count:2d}")

# 3. Extrair entidades por tipo
print("\nENTIDADES POR TIPO:")
for ent_type in ent_types.keys():
    entidades = [ent.text for ent in doc.ents if ent.label_ == ent_type]
    print(f"  {ent_type}: {entidades}")

# 4. NER em múltiplos poemas
print("\nNER EM MÚLTIPLOS POEMAS:")
heteronimos = ['Fernando Pessoa', 'Alberto Caeiro', 'Álvaro de Campos', 'Ricardo Reis']
df_pessoa = pd.read_csv('DATA/portuguese_poems.csv', encoding='utf-8')
df_pessoa = df_pessoa[df_pessoa['author'].isin(heteronimos)]

ner_por_autor = {}
for autor in heteronimos:
    textos = df_pessoa[df_pessoa['author'] == autor]['text'].values
    ents_autor = []
    
    for texto in textos[:5]:  # Primeiros 5 poemas por autor
        doc = nlp(texto)
        for ent in doc.ents:
            ents_autor.append(ent.text)
    
    ner_por_autor[autor] = ents_autor
    print(f"\n{autor}:")
    if ents_autor:
        unique_ents = list(set(ents_autor))[:10]
        print(f"  Entidades encontradas: {unique_ents}")
    else:
        print(f"  Sem entidades nomeadas.")

# 5. Visualizar entidades com displacy (opcional)
from spacy import displacy

# Selecionar um texto com entidades
textos_com_ents = [texto for texto in df_pessoa['text'].values 
                   if len(nlp(texto).ents) > 0]
if textos_com_ents:
    doc_vis = nlp(textos_com_ents[0][:500])
    html = displacy.render(doc_vis, style='ent', page=True)
    with open('ner_visualization.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("\nVisualizacao salva em: ner_visualization.html")
```

### Entregáveis:
- Código com NER (Named Entity Recognition)
- Análise de entidades por tipo
- Comparação entre autores
- Visualização HTML de entidades (opcional)

---

## Tarefa 5: Análise de Dependências Sintáticas

### Objetivo
Entender a estrutura sintática das frases e as relações entre palavras.

### Código Inicial:

```python
# 1. Dependências sintáticas
doc = nlp(poema_exemplo)

print("ANÁLISE DE DEPENDÊNCIAS SINTÁTICAS:")
print("Relações entre palavras (Token → Dependência → Head):\n")

# Mostrar primeiras 20 relações
for i, token in enumerate(list(doc)[:20]):
    if token.head != token:  # Se não for a raiz
        print(f"  {token.text:12} →({token.dep_:8})→ {token.head.text}")

# 2. Árvore sintática - extração
print("\nÁRVORE SINTÁTICA (root):")
roots = [token for token in doc if token.head == token]
for root in roots[:3]:
    print(f"\n  Raiz: '{root.text}'")
    children = list(root.children)
    for child in children[:5]:
        print(f"    └─ {child.text} ({child.dep_})")

# 3. Extrair sintagmas nominais
print("\nSINTAGMAS NOMINAIS (Noun Chunks):")
noun_chunks = list(doc.noun_chunks)
for i, chunk in enumerate(noun_chunks[:10], 1):
    print(f"  {i}. '{chunk.text}' (head: {chunk.root.text})")

# 4. Análise de verbos e seus complementos
print("\nVERBOS E SEUS COMPLEMENTOS:")
for token in doc:
    if token.pos_ == "VERB":
        # Encontrar os complementos diretos
        complements = [child.text for child in token.children 
                      if child.dep_ in ['dobj', 'attr']]
        if complements:
            print(f"  Verbo: '{token.text}' → Complementos: {complements}")

# 5. Visualizar árvore sintática
from spacy import displacy

# Criar visualização
doc_sample = nlp(poema_exemplo[:300])  # Limitar tamanho para visualização
html = displacy.render(doc_sample, style='dep', page=True, manual=False)
with open('syntax_tree.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("\nVisualizacao salva em: syntax_tree.html")

# 6. Métricas de complexidade sintática
print("\nCOMPLEXIDADE SINTÁTICA:")
num_tokens = len([token for token in doc if not token.is_punct])
num_deps = len(set([token.dep_ for token in doc]))
print(f"  Total de tokens (sem pontuação): {num_tokens}")
print(f"  Tipos diferentes de dependências: {num_deps}")
print(f"  Profundidade média: {sum([len(list(token.ancestors)) for token in doc]) / num_tokens:.2f}")
```

### Entregáveis:
- Código com análise de dependências
- Extração de sintagmas nominais
- Visualização de árvore sintática (HTML)
- Análise de complexidade sintática

---

## Tarefa 6: Pipeline Completo de PLN com Análise Comparativa

### Objetivo
Aplicar pipeline completo (NLTK vs spaCy) e comparar eficiência e resultados.

### Código Inicial:

```python
import time

# 1. PIPELINE NLTK COMPLETO
print("="*70)
print("PIPELINE NLTK")
print("="*70)

start = time.time()

# Tokenização
tokens = word_tokenize(poema_exemplo, language='portuguese')

# Limpeza
stop_words = set(stopwords.words('portuguese'))
tokens_limpos = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]

# Stemming
stemmer = RSLPStemmer()
tokens_stemmed = [stemmer.stem(t) for t in tokens_limpos]

# POS Tagging
pos_tags = nltk.pos_tag(tokens_limpos, lang='pt')

nltk_time = time.time() - start

print(f"\nResultados NLTK:")
print(f"  Tokens originais: {len(tokens)}")
print(f"  Tokens limpos: {len(tokens_limpos)}")
print(f"  POS tags únicos: {len(set([p[1] for p in pos_tags]))}")
print(f"  Tempo de processamento: {nltk_time:.4f}s")

# 2. PIPELINE spaCy COMPLETO
print("\n" + "="*70)
print("PIPELINE spaCy")
print("="*70)

start = time.time()

doc = nlp(poema_exemplo)

# Extrair informações
tokens_spacy = [token.text for token in doc if not token.is_punct]
tokens_spacy_limpos = [token.text for token in doc if token.is_alpha and not token.is_stop]
lemmas = [token.lemma_ for token in doc if token.is_alpha]
pos_tags_spacy = set([token.pos_ for token in doc])
ents = len(doc.ents)

spacy_time = time.time() - start

print(f"\nResultados spaCy:")
print(f"  Tokens: {len(tokens_spacy)}")
print(f"  Tokens limpos: {len(tokens_spacy_limpos)}")
print(f"  Lemmas únicos: {len(set(lemmas))}")
print(f"  POS tags únicos: {len(pos_tags_spacy)}")
print(f"  Entidades encontradas: {ents}")
print(f"  Tempo de processamento: {spacy_time:.4f}s")

# 3. COMPARAÇÃO
print("\n" + "="*70)
print("COMPARAÇÃO")
print("="*70)
print(f"\nspaCy é {nltk_time/spacy_time:.1f}x mais rápido" if spacy_time < nltk_time 
      else f"\nNLTK é {spacy_time/nltk_time:.1f}x mais rápido")

print("\nVANTAGENS E DESVANTAGENS:")
print("""
  NLTK:
    ✓ Simples e educacional
    ✓ Mais flexível para customização
    ✗ Mais lento
    ✗ Menos integrado

  spaCy:
    ✓ Mais rápido e eficiente
    ✓ Pipeline completo integrado
    ✓ Melhor para produção
    ✗ Menos flexível
    ✗ Requer modelos pré-treinados
""")

# 4. Aplicar em múltiplos textos
print("\nAPLICANDO PIPELINE EM MÚLTIPLOS POEMAS:")
df_pessoa = pd.read_csv('DATA/portuguese_poems.csv', encoding='utf-8')

resultados = []
for idx, row in df_pessoa.head(10).iterrows():
    doc = nlp(row['text'])
    resultados.append({
        'author': row['author'],
        'num_tokens': len([t for t in doc if not t.is_punct]),
        'num_lemmas': len(set([t.lemma_ for t in doc if t.is_alpha])),
        'num_pos_types': len(set([t.pos_ for t in doc])),
        'num_entities': len(doc.ents)
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
```

### Entregáveis:
- Código com pipeline NLTK completo
- Código com pipeline spaCy completo
- Comparação de eficiência
- Análise de múltiplos textos
- Tabela de resultados

---

## Tarefa 7: Análise Completa e Relatório

### Objetivo
Sintetizar todos os aprendizados em uma análise completa dos poemas.

### Código Inicial:

```python
# ANÁLISE COMPLETA - POEMAS PORTUGUESES

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 1. Carregar dados
df = pd.read_csv('DATA/portuguese_poems.csv', encoding='utf-8')
heteronimos = ['Fernando Pessoa', 'Alberto Caeiro', 'Álvaro de Campos', 'Ricardo Reis']
df_pessoa = df[df['author'].isin(heteronimos)]

# 2. Análise por autor
print("="*70)
print("ANÁLISE COMPLETA: POEMAS DE FERNANDO PESSOA")
print("="*70)

analise_autores = []
for autor in heteronimos:
    textos = df_pessoa[df_pessoa['author'] == autor]['text'].values
    
    # Processar todos os textos
    todos_tokens = []
    todos_lemmas = []
    todos_pos = []
    todas_ents = []
    
    for texto in textos:
        doc = nlp(texto)
        todos_tokens.extend([t.text for t in doc if not t.is_punct])
        todos_lemmas.extend([t.lemma_ for t in doc if t.is_alpha])
        todos_pos.extend([t.pos_ for t in doc])
        todas_ents.extend([ent.label_ for ent in doc.ents])
    
    analise_autores.append({
        'Autor': autor,
        'Num Poemas': len(textos),
        'Total Tokens': len(todos_tokens),
        'Vocabulário': len(set(todos_lemmas)),
        'POS Tipos': len(set(todos_pos)),
        'Entidades': len(todas_ents),
        'Tokens/Poema': len(todos_tokens) // len(textos) if len(textos) > 0 else 0
    })

df_analise = pd.DataFrame(analise_autores)
print("\nRESUMO POR AUTOR:")
print(df_analise.to_string(index=False))

# 3. Termos mais comuns por autor
print("\n" + "="*70)
print("TERMOS MAIS COMUNS POR AUTOR")
print("="*70)

for autor in heteronimos:
    textos = df_pessoa[df_pessoa['author'] == autor]['text'].values
    todos_lemmas = []
    
    for texto in textos:
        doc = nlp(texto)
        todos_lemmas.extend([t.lemma_ for t in doc 
                           if t.is_alpha and not t.is_stop])
    
    freq = Counter(todos_lemmas)
    print(f"\n{autor} - Top 10 lemmas:")
    for lemma, count in freq.most_common(10):
        print(f"  {lemma:15} : {count:3d}")

# 4. Análise POS por autor
print("\n" + "="*70)
print("DISTRIBUIÇÃO DE POS TAGS POR AUTOR")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, autor in enumerate(heteronimos):
    textos = df_pessoa[df_pessoa['author'] == autor]['text'].values
    todos_pos = []
    
    for texto in textos:
        doc = nlp(texto)
        todos_pos.extend([t.pos_ for t in doc if t.is_alpha])
    
    pos_dist = Counter(todos_pos)
    
    axes[idx].bar(pos_dist.keys(), pos_dist.values(), color='steelblue', edgecolor='navy')
    axes[idx].set_title(f'{autor}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('POS Tag')
    axes[idx].set_ylabel('Frequência')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('pos_by_author.png', dpi=150)
print("Gráfico salvo: pos_by_author.png")
plt.show()

# 5. Análise de complexidade
print("\n" + "="*70)
print("COMPLEXIDADE SINTÁTICA")
print("="*70)

complexidade = []
for autor in heteronimos:
    textos = df_pessoa[df_pessoa['author'] == autor]['text'].values
    
    prof_media = 0
    num_textos = 0
    
    for texto in textos:
        doc = nlp(texto)
        if len(doc) > 0:
            prof = sum([len(list(token.ancestors)) for token in doc]) / len(doc)
            prof_media += prof
            num_textos += 1
    
    if num_textos > 0:
        prof_media /= num_textos
    
    complexidade.append({
        'Autor': autor,
        'Profundidade Media': f"{prof_media:.2f}"
    })

df_complexidade = pd.DataFrame(complexidade)
print(df_complexidade.to_string(index=False))

# 6. Sumário estatístico
print("\n" + "="*70)
print("SUMÁRIO FINAL")
print("="*70)

print(f"""
Total de poemas analisados: {len(df_pessoa)}
Autores: {len(heteronimos)}
Distribuição:
  - Fernando Pessoa: {len(df_pessoa[df_pessoa['author'] == 'Fernando Pessoa'])} poemas
  - Alberto Caeiro: {len(df_pessoa[df_pessoa['author'] == 'Alberto Caeiro'])} poemas
  - Álvaro de Campos: {len(df_pessoa[df_pessoa['author'] == 'Álvaro de Campos'])} poemas
  - Ricardo Reis: {len(df_pessoa[df_pessoa['author'] == 'Ricardo Reis'])} poemas

Ferramentas utilizadas:
  ✓ NLTK: Tokenização, Stemming, POS-Tagging
  ✓ spaCy: Pipeline completo, NER, análise sintática

Análises realizadas:
  ✓ Tokenização e análise estrutural
  ✓ Normalização e stemming
  ✓ POS-Tagging e análise sintática
  ✓ NER (Reconhecimento de Entidades)
  ✓ Dependências sintáticas
  ✓ Análise comparativa NLTK vs spaCy
  ✓ Análise completa por autor
""")
```

### Entregáveis:
- Análise completa de todos os poemas
- Tabelas comparativas por autor
- Múltiplos gráficos de análise
- Relatório final em PDF (8-10 páginas)
- Código Python comentado

---

## Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|-----------|
| Implementação NLTK | 20% | Corretude e completude |
| Implementação spaCy | 25% | Uso correto do pipeline |
| Análise Sintática | 20% | Profundidade da análise |
| Comparação e Insights | 20% | Qualidade das conclusões |
| Documentação e Código | 15% | Clareza e organização |

---

## Dicas de Implementação

✓ Baixe recursos NLTK e modelo spaCy antes de começar  
✓ Use `encoding='utf-8'` para textos em português  
✓ spaCy é mais eficiente para textos grandes  
✓ NLTK oferece mais flexibilidade para customização  
✓ Salve visualizações em alta resolução (dpi=150+)  
✓ Comente seu código explicando cada passo  

---

**Data de Entrega:** A definir conforme cronograma  
**Formato:** Relatório PDF + Código Python + Visualizações (HTML/PNG)

Bom trabalho! 🚀

