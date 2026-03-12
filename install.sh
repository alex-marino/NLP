#!/bin/bash

# Script de instalação automática do ambiente NLP
# Uso: bash install.sh

set -e  # Parar em caso de erro

echo "========================================"
echo "Instalação do Ambiente NLP"
echo "========================================"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Função para imprimir mensagens coloridas
print_green() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_red() {
    echo -e "${RED}✗ $1${NC}"
}

# Verificar se está em um ambiente conda
if ! command -v conda &> /dev/null; then
    print_yellow "Conda não encontrado. Usando pip."
    USE_CONDA=false
else
    print_green "Conda encontrado."
    USE_CONDA=true
fi

# Verificar versão do Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_green "Python versão: $PYTHON_VERSION"

# Instalar dependências
echo ""
echo "========================================"
echo "Instalando dependências..."
echo "========================================"

pip install -r requirements.txt

print_green "Dependências instaladas com sucesso!"

# Download de recursos NLTK
echo ""
echo "========================================"
echo "Baixando recursos NLTK..."
echo "========================================"

python -c "
import nltk
print('Baixando recursos NLTK...')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('✓ Recursos NLTK baixados com sucesso!')
"

print_green "Recursos NLTK baixados!"

# Download modelo spaCy
echo ""
echo "========================================"
echo "Baixando modelo spaCy para português..."
echo "========================================"

python -m spacy download pt_core_news_sm

print_green "Modelo spaCy instalado!"

# Verificação
echo ""
echo "========================================"
echo "Verificando instalação..."
echo "========================================"

python << EOF
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False

all_ok = True
print("\nBibliotecas principais:")
all_ok &= check_import('numpy', 'NumPy')
all_ok &= check_import('pandas', 'Pandas')
all_ok &= check_import('sklearn', 'Scikit-learn')
all_ok &= check_import('nltk', 'NLTK')
all_ok &= check_import('spacy', 'spaCy')
all_ok &= check_import('gensim', 'Gensim')
all_ok &= check_import('transformers', 'Transformers')

print("\nVisualization:")
all_ok &= check_import('matplotlib', 'Matplotlib')
all_ok &= check_import('seaborn', 'Seaborn')
all_ok &= check_import('plotly', 'Plotly')
all_ok &= check_import('wordcloud', 'WordCloud')

print("\nJupyter:")
all_ok &= check_import('jupyter', 'Jupyter')
all_ok &= check_import('jupyterlab', 'JupyterLab')
all_ok &= check_import('IPython', 'IPython')

if not all_ok:
    print("\n❌ Algumas bibliotecas falharam na importação.")
    sys.exit(1)
else:
    print("\n✅ Todas as bibliotecas foram importadas com sucesso!")
EOF

if [ $? -eq 0 ]; then
    print_green "Instalação concluída com sucesso!"
else
    print_red "Houve problemas na instalação. Verifique os erros acima."
    exit 1
fi

echo ""
echo "========================================"
echo "Próximos passos:"
echo "========================================"
echo "1. Ative o ambiente (se usando conda):"
echo "   conda activate NLP"
echo ""
echo "2. Inicie o Jupyter Lab:"
echo "   jupyter lab"
echo ""
echo "3. Ou Jupyter Notebook:"
echo "   jupyter notebook"
echo "========================================"

