#!/usr/bin/env python
"""
Script para verificar e exibir informações sobre as bibliotecas instaladas
"""

import sys
import importlib

def get_version(module_name):
    """Tenta obter a versão de um módulo"""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, '__version__', 'N/A')
    except ImportError:
        return 'NOT INSTALLED'

def check_module(module_name, display_name=None):
    """Verifica se um módulo pode ser importado e retorna sua versão"""
    if display_name is None:
        display_name = module_name

    version = get_version(module_name)
    status = '✓' if version != 'NOT INSTALLED' else '✗'

    return {
        'name': display_name,
        'module': module_name,
        'version': version,
        'status': status,
        'installed': version != 'NOT INSTALLED'
    }

def main():
    print("=" * 60)
    print("Verificação do Ambiente NLP")
    print("=" * 60)
    print()

    # Python
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

    # Definir módulos a verificar
    modules = [
        # Core
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),

        # Machine Learning
        ('sklearn', 'Scikit-learn'),

        # NLP
        ('nltk', 'NLTK'),
        ('spacy', 'spaCy'),
        ('gensim', 'Gensim'),
        ('transformers', 'Transformers'),
        ('tokenizers', 'Tokenizers'),

        # Deep Learning
        ('torch', 'PyTorch'),

        # Visualization
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('plotly', 'Plotly'),
        ('wordcloud', 'WordCloud'),

        # Jupyter
        ('jupyter', 'Jupyter'),
        ('jupyterlab', 'JupyterLab'),
        ('IPython', 'IPython'),
        ('ipykernel', 'IPyKernel'),
        ('notebook', 'Notebook'),

        # Utilities
        ('yaml', 'PyYAML'),
        ('requests', 'Requests'),
        ('tqdm', 'tqdm'),
        ('regex', 'regex'),
    ]

    results = []
    for module_info in modules:
        if isinstance(module_info, tuple):
            module_name, display_name = module_info
        else:
            module_name = display_name = module_info

        result = check_module(module_name, display_name)
        results.append(result)

    # Agrupar por categoria
    categories = {
        'Core': results[0:3],
        'Machine Learning': results[3:4],
        'NLP': results[4:9],
        'Deep Learning': results[9:10],
        'Visualization': results[10:14],
        'Jupyter': results[14:19],
        'Utilities': results[19:],
    }

    # Exibir resultados
    all_installed = True
    for category, items in categories.items():
        print(f"\n{category}:")
        print("-" * 60)
        for item in items:
            print(f"{item['status']} {item['name']:<20} {item['version']}")
            if not item['installed']:
                all_installed = False

    # Verificações adicionais
    print("\n" + "=" * 60)
    print("Verificações Adicionais:")
    print("=" * 60)

    # NLTK data
    try:
        import nltk
        nltk_data_path = nltk.data.path
        print(f"✓ NLTK data path: {nltk_data_path[0] if nltk_data_path else 'N/A'}")

        # Verificar recursos específicos
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"  ✓ {resource}")
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{resource}')
                    print(f"  ✓ {resource}")
                except LookupError:
                    print(f"  ✗ {resource} (não encontrado)")
    except ImportError:
        print("✗ NLTK não está instalado")

    # spaCy models
    try:
        import spacy
        models = ['pt_core_news_sm']
        for model in models:
            try:
                spacy.load(model)
                print(f"✓ spaCy model: {model}")
            except OSError:
                print(f"✗ spaCy model: {model} (não encontrado)")
    except ImportError:
        print("✗ spaCy não está instalado")

    print("\n" + "=" * 60)
    if all_installed:
        print("✅ Todas as bibliotecas essenciais estão instaladas!")
    else:
        print("⚠️  Algumas bibliotecas não estão instaladas.")
        print("Execute: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == '__main__':
    main()

