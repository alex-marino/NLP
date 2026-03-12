# Instruções e Agenda de Entrega - Tarefas Práticas de PLN

**Disciplina:** Processamento de Linguagem Natural  
**Semestre:** 2025.1  
**Data de Atualização:** 12 de março de 2025

---

## 📋 Informações Gerais sobre Entrega e Apresentação

### Composição de Grupos

**OBRIGATÓRIO:**
- ✅ **Grupos de 4 integrantes** - Modalidade padrão
- ⚠️ **Grupos de 3 integrantes** - APENAS como exceção se não for possível formar grupos de 4

**PROIBIDO:**
- ❌ **Grupos de 5 integrantes** - Nunca será aceito
- ❌ **Trabalhos individuais** - Não será aceito
- ❌ **Grupos maiores que 5** - Não será aceito

**Comunicação de Exceções:**
Se seu grupo tiver apenas 3 integrantes (exceção), isso deve ser comunicado ao professor **antes** da data de entrega. Grupos com 3 integrantes podem ter requisitos ajustados a critério do professor.

### Apresentação

**Apresentador:**
- Um único aluno apresentará o trabalho do grupo
- O apresentador será **escolhido pelo professor no dia da apresentação**
- Não é permitido ao grupo indicar qual aluno apresentará
- O apresentador deve estar preparado para responder perguntas sobre todo o trabalho

**Tempo de Apresentação:**
- Será definido de acordo com a complexidade da tarefa
- Mínimo: 10 minutos
- Máximo: 20 minutos (varia por tarefa)

**Críterios de Apresentação:**
- Clareza na comunicação
- Domínio do conteúdo
- Capacidade de responder perguntas
- Qualidade das visualizações
- Justificativa das decisões técnicas

### Entregáveis

**Cada grupo deve entregar:**
1. ✅ **Código Python comentado** (arquivos `.py` ou notebook `.ipynb`)
2. ✅ **Relatório em PDF** (conforme especificado em cada tarefa)
3. ✅ **Visualizações** (gráficos, tabelas, word clouds em alta resolução)
4. ✅ **Modelos treinados** (se aplicável - salvar em formato `.pkl` ou `.h5`)
5. ✅ **Arquivo README.md** explicando como reproduzir os resultados

**Formato de Entrega:**
- Criar pasta: `TASK_XX_Grupo_NomeGrupo`
- Conteúdo organizado internamente
- Fazer upload no repositório ou enviar via email (conforme indicado)

---

## 📅 Cronograma de Entrega

### TASK_01 + TASK_02 - Entrega: **26 de março de 2025**

#### TASK_01: TF-IDF e Análise de Sentimentos com Poemas Portugueses
- **Conceitos:** Vetorização TF-IDF, análise de similaridade, word clouds
- **Dataset:** `DATA/portuguese_poems.csv` (Fernando Pessoa e heterônimos)
- **Entregáveis:**
  - ✅ Código Python comentado (6-8 arquivos)
  - ✅ Relatório PDF (8-10 páginas)
  - ✅ Mínimo 5 visualizações em alta resolução
  - ✅ Análise comparativa (TF vs TF-IDF)

#### TASK_02: Processamento com NLTK e spaCy
- **Conceitos:** Tokenização, POS-tagging, NER, análise sintática, comparação de ferramentas
- **Dataset:** `DATA/portuguese_poems.csv` (mesmo dos poemas)
- **Entregáveis:**
  - ✅ Código Python comentado (7-9 arquivos)
  - ✅ Relatório PDF (10-12 páginas)
  - ✅ Visualizações HTML (árvores sintáticas)
  - ✅ Tabela comparativa NLTK vs spaCy
  - ✅ Mínimo 6 gráficos

### TASK_03 - Entrega: **2 de abril de 2025**

#### TASK_03: Semântica Vetorial e Análise de Sentimentos
- **Conceitos:** Word2Vec, FastText, embeddings, deep learning (CNN/LSTM), análise comparativa
- **Dataset:** Dataset de análise de sentimentos do Kaggle (à escolha do grupo - recomendado Portuguese Sentiment Tweets)
- **Entregáveis:**
  - ✅ Código Python comentado (10-12 arquivos)
  - ✅ Relatório PDF (12-15 páginas)
  - ✅ Modelos treinados (Word2Vec, FastText, CNN, LSTM)
  - ✅ Visualizações t-SNE de embeddings
  - ✅ Mínimo 8 gráficos comparativos
  - ✅ Tabela de resultados (6+ modelos)

---

## 📊 Estrutura do Relatório

### Seções Obrigatórias para Todo Relatório:

1. **Capa** (1 página)
   - Título da tarefa
   - Nomes dos 4 integrantes
   - Data de entrega
   - Instituição

2. **Índice** (1 página)

3. **Introdução** (1-2 páginas)
   - Contexto do problema
   - Objetivos
   - Motivação

4. **Metodologia** (2-3 páginas)
   - Técnicas utilizadas
   - Ferramentas e bibliotecas
   - Pré-processamento de dados
   - Arquitetura de modelos (se aplicável)

5. **Resultados** (3-4 páginas)
   - Tabelas de performance
   - Gráficos comparativos
   - Análises específicas
   - Insights descobertos

6. **Discussão** (2-3 páginas)
   - Interpretação dos resultados
   - Trade-offs entre abordagens
   - Limitações encontradas
   - Trabalhos futuros

7. **Conclusões** (1 página)
   - Resumo dos aprendizados
   - Contribuições principais
   - Recomendações

8. **Referências** (1 página)
   - Artigos, documentação, sites consultados

9. **Apêndice** (opcional)
   - Código adicional
   - Tabelas extensas
   - Gráficos adicionais

---

## 📝 Critérios de Avaliação

### Por Tarefa (distribuição de pontos):

#### TASK_01 e TASK_02 (25 pontos cada)
- **Código (7 pontos)**
  - Qualidade e organização
  - Comentários e documentação
  - Funcionamento correto

- **Análise e Resultados (8 pontos)**
  - Exploração de dados
  - Comparações realizadas
  - Insights gerados

- **Visualizações (5 pontos)**
  - Qualidade dos gráficos
  - Clareza das representações
  - Resoluções e formato

- **Relatório (3 pontos)**
  - Clareza e organização
  - Coerência entre análise e conclusões
  - Profissionalismo

- **Apresentação (2 pontos)**
  - Clareza na comunicação
  - Resposta a perguntas
  - Domínio do conteúdo

#### TASK_03 (30 pontos)
- **Código (9 pontos)**
  - Múltiplos modelos implementados
  - Qualidade da implementação
  - Documentação

- **Análise Comparativa (10 pontos)**
  - Comparação entre 6+ abordagens
  - Análise de trade-offs
  - Conclusões fundamentadas

- **Visualizações (5 pontos)**
  - Embeddings (t-SNE, etc)
  - Gráficos comparativos
  - Qualidade geral

- **Relatório (4 pontos)**
  - Profundidade da análise
  - Clareza das conclusões
  - Estrutura e organização

- **Apresentação (2 pontos)**
  - Domínio técnico completo
  - Resposta aprofundada a perguntas
  - Profissionalismo

### Pontuação Total: 80 pontos

**Conversão para Nota:**
- 80 pontos = 10.0
- Proporção linear até 0 pontos = 0.0

---

## 🚨 Penalidades

### Por Atraso
- Até 24 horas após prazo: -2 pontos
- Até 48 horas após prazo: -5 pontos
- Após 48 horas: Não será aceito (0 pontos)

### Por Formatação Incorreta
- Grupo com 5 integrantes: Trabalho não aceito (-0 pontos)
- Grupo com número incorreto de integrantes (após exceção comunicada): -5 pontos
- Falta de pelo menos um integrante apresentado: -3 pontos

### Por Qualidade Insuficiente
- Código sem comentários: -2 pontos
- Gráficos em baixa resolução: -1 ponto por gráfico
- Relatório desorganizado: -3 pontos
- Análise superficial: -5 pontos

---

## ✅ Checklist de Entrega

Antes de submeter, verifique:

### Documentação
- [ ] Pasta com nome correto: `TASK_XX_Grupo_NomeGrupo`
- [ ] README.md explicando como executar
- [ ] Relatório PDF com 8-15 páginas (conforme tarefa)
- [ ] Nomes dos 4 integrantes no relatório
- [ ] Data de entrega no relatório

### Código
- [ ] Comentários em português explicando lógica
- [ ] Imports organizados
- [ ] Sem hardcoding de paths (usar paths relativos)
- [ ] Reproduz os resultados do relatório
- [ ] Trata exceções e erros apropriadamente

### Dados e Modelos
- [ ] Dataset em pasta `DATA/`
- [ ] Modelos salvos (se aplicável)
- [ ] Visualizações em PNG/PDF alta resolução (dpi=150+)
- [ ] Resultados podem ser reproduzidos

### Apresentação
- [ ] 1 aluno designado para apresentar (será sorteado pelo professor)
- [ ] Todos os 4 integrantes presentes na data
- [ ] Todos entendem o trabalho (para possíveis perguntas)

---

## 📞 Dúvidas Frequentes

**P: Posso apresentar com 5 integrantes?**  
R: Não. A política é rigorosa: máximo 4 integrantes. Não há exceção para 5.

**P: Se tivermos apenas 3 pessoas, o que fazer?**  
R: Comunique o professor imediatamente. Trabalhos com 3 integrantes podem ser aceitos como exceção, mas devem ser comunicados antecipadamente.

**P: Quem vai apresentar?**  
R: O professor escolherá um aluno do grupo NO DIA da apresentação. Todos devem estar preparados.

**P: Quantos slides devo preparar?**  
R: Não há requisito de slides - foque no conteúdo. Se usar slides, prepare 20-30 slides bem estruturados.

**P: Posso usar dataset diferente?**  
R: Para TASK_01 e TASK_02: não, use o dataset fornecido. Para TASK_03: sim, escolha um dataset de sentimentos do Kaggle (preferencialmente português).

**P: Qual é o prazo para comunicar grupo de 3 integrantes?**  
R: **Mínimo 1 semana antes da entrega da tarefa.**

---

## 📅 Datas Importantes

| Evento | Data | Observações |
|--------|------|-------------|
| Liberação TASK_01 + TASK_02 | 19 de março | Material disponível em `tasks/` |
| Entrega TASK_01 + TASK_02 | **26 de março** | Apresentações no mesmo dia |
| Apresentações TASK_01 + TASK_02 | **26 de março** | 2 grupos por aula |
| Liberação TASK_03 | 27 de março | Após feedback das tarefas anteriores |
| Entrega TASK_03 | **2 de abril** | Apresentações agendadas |
| Apresentações TASK_03 | **2 a 9 de abril** | Cronograma será divulgado |
| **RESULTADO FINAL** | **16 de abril** | Notas publicadas |

---

## 🎯 Resumo das Regras

✅ **Obrigatório:**
- 4 integrantes por grupo (padrão)
- 3 integrantes apenas como exceção (comunicar com antecedência)
- 1 apresentador escolhido pelo professor
- Código comentado em português
- Relatório PDF conforme especificado
- Visualizações em alta resolução

❌ **Proibido:**
- Grupos com 5 integrantes
- Trabalhos individuais
- Apresentação por email
- Código sem comentários
- Gráficos em baixa qualidade

⚠️ **Penalidades:**
- Atraso: -2 a -5 pontos
- 5 integrantes: trabalho não aceito
- Problemas de formatação: -3 a -5 pontos

---

## 📚 Materiais de Referência

- **Tarefas:** `/tasks/TASK_01.md`, `/tasks/TASK_02.md`, `/tasks/TASK_03.md`
- **Conceitos:** `/DOC/Class_01.md`, `/DOC/Class_02.md`
- **Exemplos:** `/notebooks/Class_0X.ipynb`
- **Setup TASK_03:** `python setup_task03.py`

---

**Dúvidas? Consulte o professor ou revise este documento.**

**Bom trabalho! 🎓🚀**

