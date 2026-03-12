#!/usr/bin/env python
"""
Script para copiar TASK_01.md de DOC para tasks
Execute este script para copiar o arquivo de tarefas para a pasta tasks
"""

import shutil
import os
from pathlib import Path

def main():
    # Definir caminhos
    source = Path('DOC/TASK_01.md')
    destination = Path('tasks/TASK_01.md')

    # Verificar se o arquivo de origem existe
    if not source.exists():
        print(f"❌ Erro: Arquivo {source} não encontrado!")
        return

    # Criar pasta tasks se não existir
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Copiar arquivo
    try:
        shutil.copy2(source, destination)
        print(f"✅ Arquivo copiado com sucesso!")
        print(f"   De: {source.absolute()}")
        print(f"   Para: {destination.absolute()}")
    except Exception as e:
        print(f"❌ Erro ao copiar arquivo: {e}")

if __name__ == '__main__':
    main()

