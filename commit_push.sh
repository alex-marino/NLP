#!/bin/bash
# Script para fazer commit e push de todas as alterações

echo "=========================================="
echo "GIT COMMIT E PUSH - NLP PROJECT"
echo "=========================================="

# Navegar para o diretório
cd /d/MyWorkspaces/PyWorkspace/NLP

# Adicionar todos os arquivos
echo "📝 Adicionando arquivos..."
git add -A

# Verificar status
echo ""
echo "📊 Status atual:"
git status

# Fazer commit
echo ""
echo "💾 Fazendo commit..."
git commit -m "feat: Add TASK_01, TASK_02, TASK_03 with complete practical exercises

- TASK_01: TF-IDF and Sentiment Analysis with Portuguese Poems (573 lines)
- TASK_02: NLTK and spaCy Pipeline with Comparative Analysis (762 lines)
- TASK_03: Semantic Vectorization and Sentiment Analysis (1200+ lines)
  - Includes Word2Vec, FastText, CNN, LSTM implementations
  - 5 Kaggle dataset recommendations
  - Automated setup script (setup_task03.py)
- Added 'Descrição da Tarefa' section to all task files
- Complete pedagogical progression with 7 tasks each
- Code examples for all tasks
- Evaluation criteria and learning outcomes"

# Verificar última versão
echo ""
echo "✅ Últimos commits:"
git log --oneline -5

# Fazer push
echo ""
echo "🚀 Fazendo push..."
if git push origin main 2>&1; then
    echo "✅ Push realizado com sucesso para main!"
elif git push origin master 2>&1; then
    echo "✅ Push realizado com sucesso para master!"
else
    echo "⚠️ Verifique se o repositório remoto está configurado"
    echo "Repositórios remotos:"
    git remote -v
fi

echo ""
echo "=========================================="
echo "✅ COMMIT E PUSH CONCLUÍDO!"
echo "=========================================="

