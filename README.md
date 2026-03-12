# Projeto NLP - Processamento de Linguagem Natural

Este repositório contém materiais, notebooks e exemplos práticos de Processamento de Linguagem Natural (PLN) usando Python.

## 📋 Conteúdo

### Notebooks

- **Class_01**: Introdução ao NLP
- **Class_02**: Pipelines com NLTK e spaCy
- **Class_03**: Word Embeddings
- **Class_04**: Skip-Gram e Word2Vec
- **Class_05**: Clustering, Sistemas de Recomendação e Busca Semântica
- **Class_06**: Question Answering, Sentiment Analysis e Text Generation
- **Class_07**: Summarization (Extractive e Abstractive)
- **Ngram_Models_Tutorial**: Tutorial sobre modelos N-gram
- **Trabalho_Final_baseline**: Baseline do trabalho final

### Dados

- `DATA/SUMMARY/`: Datasets para sumarização
- `DATA/TWITTER/`: Dados do Twitter para análise

### Documentação

- `DOC/`: Documentação sobre modelos N-gram e exemplos
  - Class_NGRAM_MODELS.md
  - Exemplo_2GRAM_01.md
  - Exemplo_3GRAM_01.md
  - Exemplo_perplexidade_2GRAM.md

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.8 ou superior
- Conda (recomendado) ou pip
- WSL (se estiver no Windows)

### Instalação Automática

```bash
# Tornar o script executável
chmod +x install.sh

# Executar instalação
./install.sh
```

### Instalação Manual

1. **Criar ambiente conda:**
```bash
conda create -n NLP python=3.10
conda activate NLP
```

2. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

3. **Baixar recursos NLTK:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

4. **Baixar modelo spaCy:**
```bash
python -m spacy download pt_core_news_sm
```

### Verificar Instalação

```bash
python check_env.py
```

## 📚 Bibliotecas Principais

### Core
- NumPy: Computação numérica
- Pandas: Manipulação de dados
- SciPy: Algoritmos científicos

### Machine Learning
- Scikit-learn: ML clássico (clustering, classificação, regressão)

### NLP
- **NLTK**: Natural Language Toolkit
- **spaCy**: Processamento moderno de linguagem natural
- **Gensim**: Word embeddings e modelagem de tópicos
- **Transformers**: Modelos de deep learning (BERT, GPT, etc.)

### Visualização
- Matplotlib: Visualização básica
- Seaborn: Visualização estatística
- Plotly: Gráficos interativos
- WordCloud: Nuvens de palavras

## 📖 Tópicos Abordados

### 1. Fundamentos
- Tokenização
- Stopwords
- Stemming e Lematização
- POS-Tagging
- NER (Named Entity Recognition)

### 2. Modelos de Linguagem
- N-gramas (Unigram, Bigram, Trigram)
- Perplexidade
- Smoothing

### 3. Word Embeddings
- Word2Vec (Skip-Gram e CBOW)
- GloVe
- Visualização de embeddings

### 4. Aplicações
- Análise de Sentimentos
- Classificação de Texto
- Clustering de Documentos
- Sistemas de Recomendação
- Busca Semântica
- Question Answering
- Geração de Texto
- Sumarização (Extractive e Abstractive)

### 5. Deep Learning
- Transformers
- BERT e modelos derivados
- Fine-tuning de modelos
- Transfer Learning

## 🗂️ Estrutura do Projeto

```
NLP/
├── requirements.txt          # Dependências Python
├── INSTALL.md               # Instruções detalhadas de instalação
├── README.md                # Este arquivo
├── install.sh               # Script de instalação automática
├── check_env.py             # Script de verificação do ambiente
├── .gitignore              # Arquivos ignorados pelo Git
│
├── DATA/                    # Datasets (ignorado no Git)
│   ├── SUMMARY/
│   │   ├── test.csv
│   │   └── val.csv
│   └── TWITTER/
│       ├── data.csv
│       ├── Twitter_Data.csv
│       ├── twitter_training.csv
│       └── twitter_validation.csv
│
├── DOC/                     # Documentação
│   ├── Class_NGRAM_MODELS.md
│   ├── Exemplo_2GRAM_01.md
│   ├── Exemplo_3GRAM_01.md
│   └── Exemplo_perplexidade_2GRAM.md
│
├── Models/                  # Modelos treinados (ignorado no Git)
│   ├── teste.txt
│   └── word2vec_skipgram.model
│
├── notebooks/               # Jupyter notebooks
│   ├── Class_01.ipynb
│   ├── Class_01_task.ipynb
│   ├── Class_02_nltk_exemplos.ipynb
│   ├── Class_02_nltk_pipeline.ipynb
│   ├── Class_02_spacy_exemplos.ipynb
│   ├── Class_02_spacy_pipeline.ipynb
│   ├── Class_03_wordEmbedings.ipynb
│   ├── Class_04_skipGram.ipynb
│   ├── Class_05_cluster.ipynb
│   ├── Class_05_recomendSystem.ipynb
│   ├── Class_05_semanticSearch.ipynb
│   ├── Class_05_sentiment.ipynb
│   ├── Class_06_questionAnswering.ipynb
│   ├── Class_06_sentimentAnalysis.ipynb
│   ├── Class_06_textGeneration.ipynb
│   ├── Class_07_Abstractive.ipynb
│   ├── Class_07_Extrative.ipynb
│   ├── Ngram_Models_Tutorial.ipynb
│   ├── Trabalho_Final_baseline.ipynb
│   └── Aula_Ngramas_Embeddings.ipynb
│
└── Slides/                  # Material de aula (PDFs)
    ├── PLN_2024_2_Aula_01.pdf
    ├── PLN_2024_2_Aula_02.pdf
    ├── PLN_2024_2_Aula_03.pdf
    ├── PLN_2024_2_Aula_04.pdf
    ├── PLN_2024_2_Aula_06.pdf
    ├── PLN_2024_2_Aula_07.pdf
    └── PLN_2024_2_Trabalho_Final.pdf
```

## 🎯 Como Usar

### Iniciar Jupyter Lab
```bash
jupyter lab
```

### Iniciar Jupyter Notebook
```bash
jupyter notebook
```

### Executar um notebook específico
```bash
jupyter nbconvert --execute --to notebook notebooks/Class_01.ipynb
```

## 🛠️ Troubleshooting

### Erro com PyTorch no Windows
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Erro ao baixar modelo spaCy
```bash
pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
```

### Problemas de memória no Jupyter
```bash
jupyter notebook --NotebookApp.max_buffer_size=1000000000
```

## 📝 Notas

- Os datasets e modelos grandes não estão versionados (veja `.gitignore`)
- Certifique-se de ter pelo menos 5GB de espaço livre em disco
- Algumas operações podem consumir bastante memória RAM
- Para GPU support com PyTorch, consulte a documentação oficial

## 🤝 Contribuindo

Este é um projeto educacional. Sugestões e melhorias são bem-vindas!

## 📄 Licença

Este projeto é para fins educacionais.

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Última atualização:** 2024

