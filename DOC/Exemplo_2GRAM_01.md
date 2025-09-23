# Exemplo Matemático de Modelo N-gram (Vocabulário Maior)

Considere o vocabulário:  
**{o, gato, cachorro, dorme}**

E um **corpus de treino** formado pelas frases:

1. "o gato dorme"  
2. "o cachorro dorme"  
3. "o gato dorme"  
4. "o gato dorme"  

---

## 1. Frequência de bigramas
Extraímos todos os bigramas (pares de palavras consecutivas):

- "o gato" → 3 vezes  
- "o cachorro" → 1 vez  
- "gato dorme" → 3 vezes  
- "cachorro dorme" → 1 vez  

---

## 2. Cálculo das probabilidades condicionais
Para cada palavra inicial, calculamos a probabilidade da seguinte:

$$
P(w_i \mid w_{i-1}) = \frac{\text{Contagem}(w_{i-1}, w_i)}{\sum_{w} \text{Contagem}(w_{i-1}, w)}
$$

### Exemplo 1: depois de "o"
- Total após "o": $3 + 1 = 4$  
- $ P(\text{gato} \mid o) = \frac{3}{4} = 0.75 $  
- $ P(\text{cachorro} \mid o) = \frac{1}{4} = 0.25 $  

### Exemplo 2: depois de "gato"
- Total após "gato": $3$  
- $ P(\text{dorme} \mid gato) = \frac{3}{3} = 1.0 $  

### Exemplo 3: depois de "cachorro"
- Total após "cachorro": $1$  
- $ P(\text{dorme} \mid cachorro) = \frac{1}{1} = 1.0 $  

---

## 3. Probabilidade de frases

### Frase A: "o gato dorme"
$$
P(o\ gato\ dorme) = P(o) \times P(gato \mid o) \times P(dorme \mid gato)
$$

Assumindo $ P(o) = 0.5 $ (metade das frases do corpus começam com "o"):

$$
P(o\ gato\ dorme) = 0.5 \times 0.75 \times 1.0 = 0.375
$$

---

### Frase B: "o cachorro dorme"
$$
P(o\ cachorro\ dorme) = P(o) \times P(cachorro \mid o) \times P(dorme \mid cachorro)
$$

$$
P(o\ cachorro\ dorme) = 0.5 \times 0.25 \times 1.0 = 0.125
$$

---

### Frase C: "o gato gato"
Não aparece no corpus → **probabilidade zero** (a não ser que usemos suavização).

---

## 4. Interpretação
- O modelo considera "o gato dorme" **três vezes mais provável** do que "o cachorro dorme".  
- Sequências nunca vistas ("o gato gato") recebem probabilidade 0 sem técnicas de suavização.  
- Esse problema da esparsidade motiva o uso de técnicas mais modernas (Laplace smoothing, embeddings, Transformers etc.).
