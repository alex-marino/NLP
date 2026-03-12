# Aula 01 — TF e TF-IDF (Class_01)

## Objetivo da aula
Gerar vetores de caracteristicas de texto para um conjunto de documentos, usando:
- TF (frequencia do termo)
- TF-IDF (frequencia do termo e frequencia inversa do documento)

Ao final, o objetivo e aplicar essas representacoes nos datasets da pasta `DATA`.

## Preparacao do ambiente

### Bibliotecas
- pandas
- nltk (stopwords, tokenizacao)
- scikit-learn (CountVectorizer, TfidfTransformer)
- matplotlib e wordcloud (visualizacao)

### Downloads NLTK
E necessario baixar os recursos:
- `stopwords`
- `punkt`
- `punkt_tab`

## Importacao do dataset

- Carregamento do arquivo `../DATA/data.csv` com `pandas.read_csv`.
- Visualizacao inicial do dataframe.

## Pre-processamento

### 1) Remocao de pontuacao
Duas etapas foram usadas:
- Substituir pontuacao basica por espaco.
- Remover simbolos adicionais (ex.: operadores, parenteses, simbolos especiais).

Exemplo de uso:
- `df['clean_punct']`
- `df['clean_punct2']`

### 2) Remocao de numeros
Numeros sao removidos e a limpeza de pontuacao e reaplicada.

### 3) Stopwords
- Carregar stopwords do ingles com `stopwords.words('english')`.

### 4) Tokenizacao
- Exemplo de tokenizacao com `nltk.word_tokenize`.

## Vetorizacao

### CountVectorizer (TF)
- `CountVectorizer` com:
  - `strip_accents='ascii'`
  - `lowercase=True`
  - `stop_words=stop_words`
- Ajuste com um subconjunto do corpus e geracao da matriz termo-documento.

Resultado esperado:
- Lista de termos com `get_feature_names_out()`
- Matriz de contagens com `toarray()`

### TF-IDF
- `TfidfTransformer(use_idf=True, norm='l2')`
- Ajuste usando a matriz de frequencia (TF)
- Transformacao para matriz TF-IDF

## Atividades propostas

1) Gerar grafico de termos mais frequentes considerando os ngramas:
   - ngram(1,1)
   - ngram(1,2)
   - ngram(2,2)
   - ngram(2,3)

2) Criar wordclouds para as configuracoes de ngram acima.

3) Investigar o efeito da remocao de stopwords no tamanho do dicionario final.

4) Investigar as tecnicas de normalizacao L1 e L2.

## Sugestoes de extensao

- Comparar resultados entre TF e TF-IDF para os mesmos ngramas.
- Avaliar impacto de remover numeros vs manter numeros.
- Repetir o processo para diferentes datasets em `DATA/`.

## Materiais relacionados

- Notebook: `notebooks/Class_01.ipynb`
- Dataset: `DATA/data.csv`

