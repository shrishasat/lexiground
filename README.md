Yes. Since **0.1.0 is already published**, I would make the README reflect what actually works now, avoid promising a problematic `[sbert]` install command, and clearly separate **human norms** from **model-based estimates**.

One important correction: your validation showed **Iconicity n = 2,955**, while the Lancaster dimensions have **n = 7,942** in your held-out validation. I would present those validation results explicitly rather than implying all features were validated on the same number of words.

# LexiGround

**LexiGround** is an open-source Python package for retrieving and estimating lexical **sensorimotor and iconicity ratings** from language models.

It combines publicly available human lexical norms with pretrained **SBERT + feature-specific Ridge regression models** to provide ratings for words that may not be present in the original human-rated datasets.

[![PyPI version](https://img.shields.io/pypi/v/lexiground.svg)](https://pypi.org/project/lexiground/)
[![Python versions](https://img.shields.io/pypi/pyversions/lexiground.svg)](https://pypi.org/project/lexiground/)
[![License](https://img.shields.io/github/license/shrishasat/lexiground)](https://github.com/shrishasat/lexiground/blob/main/LICENSE)

---

## Features

LexiGround currently provides the following lexical dimensions:

### Sensorimotor dimensions

* Auditory
* Gustatory
* Haptic
* Interoceptive
* Olfactory
* Visual
* Foot/leg
* Hand/arm
* Head
* Mouth
* Torso

### Composite dimensions

* Minkowski perceptual
* Minkowski action
* Minkowski sensorimotor

### Other lexical dimension

* Iconicity

When a human rating is available for a word, LexiGround returns the original human rating.

When a human rating is unavailable, LexiGround can estimate the corresponding value using a pretrained SBERT + Ridge regression model.

---

## Installation

Install LexiGround directly from PyPI:

```bash
pip install lexiground
```

The package itself does not require users to manually download or specify paths to the underlying lexical norm CSV files.

Required public datasets are downloaded automatically when needed and stored in the user's local cache.

### SBERT estimation

SBERT-based estimation requires `sentence-transformers` and PyTorch.

For CPU-only environments, install a CPU version of PyTorch first and then install the SBERT dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

Then use LexiGround normally:

```python
from lexiground import LexiGround

lex = LexiGround()
```

> **Note:** GPU/CUDA installation is not required for LexiGround's pretrained models. CPU inference is supported.

---

## Quick Start

```python
from lexiground import LexiGround

lex = LexiGround()

result = lex.get("neuroscience")

for feature, value in result["features"].items():
    print(
        feature,
        "→",
        value["value"],
        "|",
        value["source"]
    )
```

Example output has the general form:

```text
Auditory →  ...
Visual →  ...
Haptic →  ...
Iconicity →  ...
```

The `source` field indicates whether the value came from an original human norm or was estimated by a pretrained model.

---

## How It Works

LexiGround follows a simple lookup-and-estimation workflow:

```text
                 Input word
                     │
                     ▼
             ┌───────────────┐
             │ LexiGround    │
             │ word lookup   │
             └───────┬───────┘
                     │
            ┌────────┴────────┐
            │                 │
     Human rating        No human rating
       available            available
            │                 │
            ▼                 ▼
    Return original      SBERT embedding
    human rating              │
                              ▼
                       Feature-specific
                       Ridge regression
                              │
                              ▼
                       Estimated rating
```

This allows users to work with lexical items that are not necessarily present in the original human-rated datasets.

---

## Human Ratings vs. Estimated Ratings

LexiGround prioritizes human ratings whenever they are available.

For example:

```python
result = lex.get("neuroscience")

for feature, value in result["features"].items():
    print(
        feature,
        value["value"],
        value["source"]
    )
```

The returned structure contains the rating and its source:

```python
{
    "value": ...,
    "source": ...
}
```

This distinction allows downstream analyses to determine whether a value originated from a human normative dataset or from model-based estimation.

---

## Looking Up Multiple Words

The same LexiGround object can be reused for multiple words:

```python
from lexiground import LexiGround

lex = LexiGround()

words = [
    "neuroscience",
    "thunder",
    "beautiful",
    "running",
]

for word in words:

    result = lex.get(word)

    print(f"\n{word}")

    for feature, value in result["features"].items():

        print(
            feature,
            "→",
            value["value"],
            "|",
            value["source"]
        )
```

---

## Available Features

You can inspect the supported dimensions programmatically:

```python
from lexiground import LexiGround

lex = LexiGround()

print(lex.available_features())
```

You can also inspect the loaded pretrained models:

```python
print(lex.models.keys())
```

---

## Data Sources and Attribution

LexiGround uses publicly available lexical norm datasets, including:

### Lancaster Sensorimotor Norms

Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020).

*The Lancaster Sensorimotor Norms: multidimensional measures of perceptual and action strength for 40,000 English words.*

**Behavior Research Methods, 52**, 1271–1291.

### English Iconicity Ratings

Winter, B., Lupyan, G., Perry, L. K., Dingemanse, M., & Perlman, M. (2024).

*Iconicity ratings for 14,000+ English words.*

**Behavior Research Methods, 56**, 1640–1655.

The underlying datasets are obtained from their original public sources when required and are **not redistributed as part of the LexiGround package**.

Users should refer to the original sources for the applicable attribution and licensing terms.

LexiGround does not claim ownership of the underlying datasets.

---

## Automatic Data Download and Caching

Users do **not** need to download the Lancaster or iconicity CSV files manually or provide local file paths.

For example:

```python
from lexiground import LexiGround

lex = LexiGround()
```

When a required dataset is not already present locally, LexiGround downloads it from its configured public source and stores it in the user's local cache.

The cached files are stored on the user's own machine.

LexiGround therefore does not require access to the developer's filesystem, server, or research-computing environment.

---

## Pretrained Models

LexiGround includes pretrained feature-specific Ridge regression models using SBERT representations.

The models were trained to estimate the lexical dimensions listed above from contextualized sentence embeddings.

The released models are distributed with the LexiGround package and are loaded automatically by the package.

Users do not need to train the models themselves to use LexiGround.

---

## Validation

The pretrained models were evaluated using held-out test data.

The validation procedure evaluates both linear and rank-order agreement between predicted and human ratings.

### Held-out validation results

| Feature                |     n | Pearson r | Spearman ρ |    R² |  RMSE |   MAE |
| ---------------------- | ----: | --------: | ---------: | ----: | ----: | ----: |
| Auditory               | 7,942 |     0.707 |      0.693 | 0.500 | 0.701 | 0.545 |
| Gustatory              | 7,942 |     0.761 |      0.338 | 0.579 | 0.454 | 0.294 |
| Haptic                 | 7,942 |     0.742 |      0.687 | 0.551 | 0.623 | 0.476 |
| Interoceptive          | 7,942 |     0.755 |      0.712 | 0.571 | 0.580 | 0.438 |
| Olfactory              | 7,942 |     0.701 |      0.449 | 0.491 | 0.452 | 0.303 |
| Visual                 | 7,942 |     0.674 |      0.680 | 0.454 | 0.664 | 0.524 |
| Foot/leg               | 7,942 |     0.686 |      0.588 | 0.470 | 0.553 | 0.412 |
| Hand/arm               | 7,942 |     0.697 |      0.657 | 0.486 | 0.651 | 0.509 |
| Head                   | 7,942 |     0.598 |      0.577 | 0.358 | 0.583 | 0.464 |
| Mouth                  | 7,942 |     0.693 |      0.652 | 0.481 | 0.648 | 0.496 |
| Torso                  | 7,942 |     0.634 |      0.559 | 0.402 | 0.512 | 0.384 |
| Minkowski perceptual   | 7,942 |     0.656 |      0.646 | 0.431 | 0.652 | 0.518 |
| Minkowski action       | 7,942 |     0.586 |      0.556 | 0.343 | 0.675 | 0.535 |
| Minkowski sensorimotor | 7,942 |     0.678 |      0.664 | 0.460 | 0.663 | 0.525 |
| Iconicity              | 2,955 |     0.586 |      0.547 | 0.342 | 0.721 | 0.576 |

The complete validation results are available in:

```text
results/sbert_validation.csv
```

---

## Example

A complete usage example is available at:

```text
examples/example.py
```

Run it with:

```bash
python examples/example.py
```

---

## Citation

If you use LexiGround in research, please cite the software:

> Sathishkumar, S. (2026). *LexiGround: Estimating Lexical Sensorimotor and Iconicity Ratings from Language Models* (Version 0.1.0) [Computer software]. [https://github.com/shrishasat/lexiground](https://github.com/shrishasat/lexiground)

Please also cite the original lexical norm datasets used by LexiGround:

**Lancaster Sensorimotor Norms**

> Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020). The Lancaster Sensorimotor Norms: multidimensional measures of perceptual and action strength for 40,000 English words. *Behavior Research Methods, 52*, 1271–1291.

**Iconicity Ratings**

> Winter, B., Lupyan, G., Perry, L. K., Dingemanse, M., & Perlman, M. (2024). Iconicity ratings for 14,000+ English words. *Behavior Research Methods, 56*, 1640–1655.

---

## License

LexiGround software is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

The MIT License applies to the LexiGround software itself and does not replace or supersede the original licensing or attribution requirements of the underlying lexical norm datasets.

---

## Repository

Source code, pretrained models, validation results, and documentation:

[https://github.com/shrishas/lexiground](https://github.com/shrishas/lexiground)

## PyPI

LexiGround is available on PyPI:

[https://pypi.org/project/lexiground/](https://pypi.org/project/lexiground/)

## Version

Current release:

**0.1.0 — August 2026**

---

```
```
## Acknowledgements

LexiGround was developed as part of research conducted at the Centre for
Human Brain Health, University of Birmingham.

I thank **Dr. Hyojin Park** and **Dr. Marcus Perlman** for
research supervision and support during the development of the project.
