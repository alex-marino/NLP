# Modelo de N-gram

Um **modelo de N-gram** é um tipo de modelo probabilístico usado em **Processamento de Linguagem Natural (PLN)** para estimar a probabilidade de uma palavra aparecer em uma sequência, dado o contexto das palavras anteriores.

---

## 🔎 Definição básica
- Um **N-grama** é uma sequência de **N palavras consecutivas** em um texto.  
  - **Unigrama (N=1):** considera cada palavra isolada.  
  - **Bigrama (N=2):** considera pares de palavras consecutivas.  
  - **Trigrama (N=3):** considera sequências de três palavras consecutivas.  

**Exemplo:** frase: *"O gato preto dorme"*  
- Unigramas: `["O", "gato", "preto", "dorme"]`  
- Bigramas: `["O gato", "gato preto", "preto dorme"]`  
- Trigramas: `["O gato preto", "gato preto dorme"]`

---

## 📊 Modelo de linguagem com N-gramas
Um modelo de linguagem busca atribuir probabilidade a uma frase.  
A ideia central do modelo N-gram é **aproximar a probabilidade de uma palavra considerando apenas as N-1 palavras anteriores**, em vez de toda a história.

$$
P(w_1, w_2, \dots, w_m) \approx \prod_{i=1}^m P(w_i \mid w_{i-(N-1)}, \dots, w_{i-1})
$$

**Exemplo com bigrama (N=2):**  
$$
P(O\ gato\ preto\ dorme) \approx P(O) \times P(gato|O) \times P(preto|gato) \times P(dorme|preto)
$$

---

## ✅ Vantagens
- Simples de treinar a partir de um corpus de texto.
- Boa performance em tarefas básicas (correção ortográfica, predição de próxima palavra).
- Base para algoritmos mais sofisticados.

---

## ⚠️ Limitações
- **Dependência curta:** só considera N-1 palavras anteriores → ignora contextos longos.
- **Esparsidade:** muitas sequências possíveis nunca aparecem no corpus.
- **Memória:** para N grandes, o número de combinações cresce exponencialmente.

---

## 📌 Aplicações
- Corretores ortográficos e sistemas de sugestão de próxima palavra (teclados de celular).
- Tradução automática estatística.
- Modelagem básica de linguagem em chatbots simples.
