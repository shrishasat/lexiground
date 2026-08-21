# lexiground

**Lexical Sensorimotor and Iconicity Rating Estimation from Language**

LexiGround is an open-source Python package for retrieving and estimating
lexical semantic ratings from words.

The package combines existing human lexical norm databases with
pretrained **Sentence-BERT (SBERT) + Ridge regression models** to estimate
ratings for words that are absent from the available human norms.

---

## Features

LexiGround currently provides models for:

### Perceptual dimensions

- Auditory
- Gustatory
- Haptic
- Interoceptive
- Olfactory
- Visual

### Motor dimensions

- Foot/leg
- Hand/arm
- Head
- Mouth
- Torso

### Composite sensorimotor dimensions

- Minkowski3 perceptual
- Minkowski3 action
- Minkowski3 sensorimotor

### Form–meaning correspondence

- Iconicity

For each feature, LexiGround can return either:

- a **human rating**, when the word is present in the relevant norm
  database; or
- an **estimated rating**, generated using a pretrained SBERT + Ridge
  regression model.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/shrishasat/lexiground.git
cd lexiground





# References

## Lexical norm datasets

### Lancaster Sensorimotor Norms

Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020).
The Lancaster Sensorimotor Norms for 40,000 English words.
*Behavior Research Methods, 52*, 1271–1291.

### Iconicity ratings

Winter, B., Lupyan, G., Perry, L.K., Dingemanse, M., Perlman, M. (2024).
Iconicity ratings for 14,000+ English words.
*Behavior Research Methods, 56*, 1640-1655.


## Computational methods

### Sentence-BERT

Reimers, N., & Gurevych, I. (2019).
Sentence-BERT: Sentence embeddings using Siamese BERT-networks.
*Proceedings of EMNLP-IJCNLP*.

### Ridge regression

LexiGround uses Ridge regression to map SBERT representations
onto human lexical ratings.


