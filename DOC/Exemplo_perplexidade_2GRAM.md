# Perplexidade em Modelos de Linguagem (Exemplo com Bigramas)

A **perplexidade** mede o quão bem um modelo de linguagem prevê uma sequência de palavras.  
- Quanto **menor** a perplexidade, **melhor** o modelo.  
- Intuitivamente, é o "grau de surpresa" do modelo frente ao texto.  

---

## 1. Fórmula geral

Dado um modelo de linguagem que estima a probabilidade de uma sequência de palavras $ w_1, w_2, \dots, w_N $:

$$
PP(W) = P(w_1, w_2, \dots, w_N)^{-\frac{1}{N}}
$$

ou, equivalentemente:

$$
PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2 P(w_i \mid contexto)}
$$

---

## 2. Exemplo com Bigramas (2-gram)

Considere o corpus de treino (mesmo usado antes):

1. "o gato dorme"  
2. "o cachorro dorme"  
3. "o gato dorme"  
4. "o gato dorme"  

### Probabilidades de bigramas
- $ P(gato \mid o) = \frac{3}{4} = 0.75 $  
- $ P(cachorro \mid o) = \frac{1}{4} = 0.25 $  
- $ P(dorme \mid gato) = \frac{3}{3} = 1.0 $  
- $ P(dorme \mid cachorro) = \frac{1}{1} = 1.0 $  

---

## 3. Frase de teste: "o gato dorme"

Cálculo da probabilidade da sequência usando bigramas:

$$
P(o\ gato\ dorme) = P(o) \times P(gato \mid o) \times P(dorme \mid gato)
$$

Assumindo $ P(o) = 0.5 $:

$$
P(o\ gato\ dorme) = 0.5 \times 0.75 \times 1.0 = 0.375
$$

---

## 4. Cálculo da Perplexidade

A frase tem $N=3$ palavras.

$$
PP(W) = P(W)^{-\frac{1}{N}}
$$

$$
PP(o\ gato\ dorme) = 0.375^{-\frac{1}{3}}
$$

$$
PP(o\ gato\ dorme) \approx 1.39
$$

---

## 5. Interpretação

- A perplexidade **1.39** significa que o modelo está relativamente "confortável" ao prever essa frase.  
- Para comparação:
  - Se testássemos "o cachorro dorme", a probabilidade seria menor ($0.125$), logo a perplexidade seria maior ($\approx 2.0$), indicando que o modelo a considera **menos provável**.  
  - Se usássemos uma frase nunca vista ("o gato cachorro"), a probabilidade seria **0**, e a perplexidade tenderia ao infinito.

---

## 6. Intuição final

- **Baixa perplexidade** → o modelo prevê bem a frase (texto coerente com o treino).  
- **Alta perplexidade** → o modelo está "surpreso" (texto improvável ou incoerente com o treino).  
- É uma métrica muito usada para comparar modelos de linguagem (bigramas, trigramas, redes neurais, Transformers etc.).
