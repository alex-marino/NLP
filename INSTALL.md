# Instalação do Ambiente NLP

Este documento descreve como configurar o ambiente Python para o projeto NLP.

## Requisitos

- Python 3.8 ou superior
- Conda (recomendado) ou pip
- WSL (Windows Subsystem for Linux) se estiver no Windows

## Instalação com Conda (Recomendado)

### 1. Criar o ambiente

```bash
conda create -n NLP python=3.10
conda activate NLP
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Download de recursos adicionais do NLTK

Após instalar o NLTK, você precisará baixar alguns recursos adicionais:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

### 4. Download do modelo do spaCy para português

```bash
python -m spacy download pt_core_news_sm
```

## Instalação com pip (Alternativa)

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# No Linux/WSL:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Download de recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"

# Download modelo spaCy
python -m spacy download pt_core_news_sm
```

## Verificação da Instalação

Para verificar se tudo foi instalado corretamente:

```python
# Testar imports principais
import numpy as np
import pandas as pd
import sklearn
import nltk
import spacy
import gensim
import transformers
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Todas as bibliotecas foram importadas com sucesso!")
```

## Bibliotecas Principais

### Core
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **SciPy**: Algoritmos científicos

### Machine Learning
- **Scikit-learn**: Algoritmos de ML clássicos

### NLP
- **NLTK**: Natural Language Toolkit
- **spaCy**: Processamento de linguagem natural moderno
- **Gensim**: Modelagem de tópicos e word embeddings
- **Transformers**: Modelos de deep learning da HuggingFace

### Visualização
- **Matplotlib**: Visualização básica
- **Seaborn**: Visualização estatística
- **Plotly**: Gráficos interativos
- **WordCloud**: Nuvens de palavras

### Jupyter
- **JupyterLab**: Ambiente de desenvolvimento
- **Notebook**: Interface notebook
- **IPython**: Shell interativo

## Troubleshooting

### Erro com PyTorch no Windows

Se tiver problemas com PyTorch no Windows, instale a versão CPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Erro com spaCy

Se o modelo português não instalar:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl
```

### Problemas de Memória

Para notebooks grandes, aumente o limite de memória do Jupyter:

```bash
jupyter notebook --NotebookApp.max_buffer_size=1000000000
```

## Recursos Adicionais

### Dados NLTK

Os seguintes corpora e modelos do NLTK são utilizados no projeto:
- `punkt`: Tokenizador
- `stopwords`: Palavras de parada
- `wordnet`: Base lexical
- `averaged_perceptron_tagger`: POS tagger
- `omw-1.4`: Open Multilingual WordNet

### Modelos spaCy

Para português, utilizamos:
- `pt_core_news_sm`: Modelo small para português

## Estrutura do Projeto

```
NLP/
├── requirements.txt          # Dependências Python
├── INSTALL.md               # Este arquivo
├── DATA/                    # Datasets
│   ├── SUMMARY/
│   └── TWITTER/
├── notebooks/               # Jupyter notebooks
├── Models/                  # Modelos treinados
└── DOC/                    # Documentação
```

## Notas

- O arquivo `.gitignore` está configurado para ignorar a pasta `/Models/*` e `/DATA/*`
- Modelos grandes devem ser baixados separadamente
- Certifique-se de ter espaço suficiente em disco (pelo menos 5GB)

## Suporte

Para problemas ou dúvidas:
1. Verifique os logs de instalação
2. Consulte a documentação oficial das bibliotecas
3. Verifique as versões do Python e pip

## Licença

Este projeto é para fins educacionais.

