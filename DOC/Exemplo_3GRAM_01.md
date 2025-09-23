# Exemplo Matemático de Modelo Trigrama

Vocabulário:  
**{o, gato, cachorro, dorme}**

Corpus de treino:  
1. "o gato dorme"  
2. "o cachorro dorme"  
3. "o gato dorme"  
4. "o gato dorme"  

---

## 1. Extraindo trigramas
De cada frase, pegamos **sequências de 3 palavras consecutivas**:

- "o gato dorme" → aparece 3 vezes  
- "o cachorro dorme" → aparece 1 vez  

Total de trigramas: 4

---

## 2. Fórmula da probabilidade trigram
Um **modelo de trigramas** aproxima a probabilidade de uma frase como:

$$
P(w_1, w_2, w_3, \dots, w_m) \approx \prod_{i=1}^m P(w_i \mid w_{i-2}, w_{i-1})
$$

Ou seja, cada palavra depende das **duas anteriores**.

---

## 3. Cálculo de probabilidades condicionais

### Exemplo 1: P(dorme | o, gato)
$$
P(\text{dorme} \mid o, gato) = \frac{\text{Contagem("o gato dorme")}}{\text{Contagem("o gato ...")}}
$$

- "o gato dorme" → 3  
- "o gato ..." → 3  
$$
P(\text{dorme} \mid o, gato) = \frac{3}{3} = 1.0
$$

---

### Exemplo 2: P(dorme | o, cachorro)
$$
P(\text{dorme} \mid o, cachorro) = \frac{\text{Contagem("o cachorro dorme")}}{\text{Contagem("o cachorro ...")}}
$$

- "o cachorro dorme" → 1  
- "o cachorro ..." → 1  
$$
P(\text{dorme} \mid o, cachorro) = \frac{1}{1} = 1.0
$$

---

## 4. Probabilidade de frases

### Frase A: "o gato dorme"
$$
P(o\ gato\ dorme) = P(o) \times P(gato \mid o) \times P(dorme \mid o, gato)
$$

Do modelo de bigramas vimos que:
- $ P(o) = 0.5 $ (metade das frases começam com "o")  
- $ P(gato \mid o) = 0.75 $  
- $ P(dorme \mid o, gato) = 1.0 $  

$$
P(o\ gato\ dorme) = 0.5 \times 0.75 \times 1.0 = 0.375
$$

---

### Frase B: "o cachorro dorme"
$$
P(o\ cachorro\ dorme) = P(o) \times P(cachorro \mid o) \times P(dorme \mid o, cachorro)
$$

- $ P(o) = 0.5 $  
- $ P(cachorro \mid o) = 0.25 $  
- $ P(dorme \mid o, cachorro) = 1.0 $  

$$
P(o\ cachorro\ dorme) = 0.5 \times 0.25 \times 1.0 = 0.125
$$

---

### Frase C: "o gato cachorro"
Esse trigram **não aparece no corpus** → probabilidade **0** sem suavização.

---

## 5. Interpretação
- Com trigramas, o modelo diferencia contextos mais longos.  
- "o gato dorme" e "o cachorro dorme" têm probabilidades distintas, reforçando padrões frequentes no corpus.  
- Sequências não vistas ficam com probabilidade 0 (problema da **esparsidade**), que pode ser corrigido com **suavização (Laplace, Kneser-Ney, etc.)**.
