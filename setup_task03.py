#!/usr/bin/env python
"""
Script de Setup para TASK_03
Baixa dataset do Kaggle e prepara ambiente para análise de sentimentos
"""

import os
import subprocess
import pandas as pd
import sys

def check_kaggle_api():
    """Verifica se Kaggle API está instalada."""
    try:
        import kaggle
        print("✅ Kaggle API instalada")
        return True
    except ImportError:
        print("❌ Kaggle API não instalada")
        print("Execute: pip install kaggle")
        return False

def check_kaggle_credentials():
    """Verifica se credenciais do Kaggle estão configuradas."""
    cred_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(cred_path):
        print("✅ Credenciais do Kaggle encontradas")
        return True
    else:
        print("❌ Credenciais do Kaggle não encontradas")
        print(f"Salve seu token em: {cred_path}")
        return False

def download_dataset(dataset_name, output_path='DATA'):
    """Baixa dataset do Kaggle."""
    print(f"\n📥 Baixando dataset: {dataset_name}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        cmd = f"kaggle datasets download -d {dataset_name} -p {output_path} --unzip"
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Dataset baixado em {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao baixar dataset: {e}")
        return False

def check_data_structure(data_path='DATA'):
    """Verifica estrutura do dataset."""
    print(f"\n📊 Analisando estrutura do dataset...")

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado")
        return False

    print(f"✅ Arquivos CSV encontrados: {csv_files}")

    # Analisar primeiro CSV
    csv_file = os.path.join(data_path, csv_files[0])
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"\n📋 Dataset: {csv_files[0]}")
        print(f"   Shape: {df.shape}")
        print(f"   Colunas: {df.columns.tolist()}")
        print(f"   Primeiras 3 linhas:")
        print(df.head(3).to_string())
        return True
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        return False

def main():
    print("="*70)
    print("SETUP PARA TASK_03 - ANÁLISE DE SENTIMENTOS")
    print("="*70)

    # 1. Verificar Kaggle
    print("\n🔍 Verificando Kaggle API...")
    if not check_kaggle_api():
        print("⚠️ Instale Kaggle: pip install kaggle")
        return False

    if not check_kaggle_credentials():
        print("\n📝 Para obter credenciais:")
        print("1. Acesse https://www.kaggle.com/settings/account")
        print("2. Clique em 'Create New Token'")
        print("3. Salve em ~/.kaggle/kaggle.json")
        return False

    # 2. Opcoes de dataset
    print("\n" + "="*70)
    print("ESCOLHA UM DATASET")
    print("="*70)

    datasets = {
        '1': ('rpscampos/portuguese-sentiment-tweets', 'Portuguese Sentiment Tweets (13K, Recomendado)'),
        '2': ('felipec76/mercado-libre-reviews-portuguese', 'Amazon Reviews Portuguese (10K+)'),
        '3': ('kazanova/sentiment140', 'Sentiment140 - 1.6M tweets (Multi-idioma)'),
        '4': ('mrtkmiller/hotel-reviews-from-brazil', 'Hotel Reviews Brazil (80K)'),
    }

    for key, (dataset_id, desc) in datasets.items():
        print(f"{key}. {desc}")

    escolha = input("\nEscolha (1-4) [1]: ").strip() or '1'

    if escolha not in datasets:
        print("❌ Opção inválida")
        return False

    dataset_id, dataset_name = datasets[escolha]

    # 3. Baixar dataset
    print(f"\n📥 Você escolheu: {dataset_name}")
    resposta = input("Deseja baixar? (S/N) [S]: ").strip().lower() or 's'

    if resposta == 's':
        if not download_dataset(dataset_id):
            return False

    # 4. Verificar estrutura
    if os.path.exists('DATA'):
        check_data_structure()

    print("\n" + "="*70)
    print("✅ SETUP CONCLUÍDO!")
    print("="*70)
    print("\n📝 Próximos passos:")
    print("1. Abra: tasks/TASK_03.md")
    print("2. Comece pela Tarefa 1: Exploração de Dataset")
    print("3. Use os exemplos de código fornecidos")
    print("\n💡 Dica: Comece com a Tarefa 1 para entender seu dataset!")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

