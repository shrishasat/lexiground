```python
"""
LexiGround: Basic Usage Example
================================

This example demonstrates how to use LexiGround to retrieve lexical
sensorimotor and iconicity ratings.

LexiGround:
1. Automatically downloads the required public lexical norm datasets
   when they are first needed.
2. Returns human ratings when a word is present in the original
   normative datasets.
3. Uses pretrained SBERT + Ridge regression models to estimate ratings
   when human ratings are unavailable.

Install:
    pip install lexiground

For SBERT-based estimation:
    pip install "lexiground[sbert]"
"""

from lexiground import LexiGround


# ---------------------------------------------------------------------
# 1. Create a LexiGround object
# ---------------------------------------------------------------------
#
# No CSV paths are required.
#
# LexiGround automatically handles the required datasets and stores
# downloaded data in the user's local cache.
#
# The first run may take longer because the datasets and/or SBERT
# model need to be downloaded.
#
lex = LexiGround()


# ---------------------------------------------------------------------
# 2. See which lexical features are available
# ---------------------------------------------------------------------

print("Available features:")
print(lex.available_features())


# ---------------------------------------------------------------------
# 3. Look up a single word
# ---------------------------------------------------------------------
#
# Here we look up "neuroscience".
#
# For each feature, LexiGround reports:
#   - value  : the rating
#   - source : whether the value comes from a human norm or an
#              SBERT-based estimate
#
word = "neuroscience"

result = lex.get(word)

print(f"\nResults for: {word}")
print("-" * 60)

for feature, value in result["features"].items():
    print(
        f"{feature:25s} "
        f"→ {value['value']:.3f} "
        f"({value['source']})"
    )


# ---------------------------------------------------------------------
# 4. Understanding the "source" field
# ---------------------------------------------------------------------
#
# If a word exists in the original human-rated dataset, LexiGround
# returns the human rating.
#
# If the word is not available in the human norms, LexiGround can
# estimate the rating using the pretrained SBERT + Ridge model.
#
# This means users do NOT need to know whether a word is present in
# the underlying datasets before calling lex.get().
#
print("\nExample source labels:")
print("  human     = rating obtained from the original human norms")
print("  estimated = rating predicted by the pretrained SBERT model")


# ---------------------------------------------------------------------
# 5. Look up another word
# ---------------------------------------------------------------------

word = "thunder"

result = lex.get(word)

print(f"\nResults for: {word}")
print("-" * 60)

for feature, value in result["features"].items():
    print(
        f"{feature:25s} "
        f"→ {value['value']:.3f} "
        f"({value['source']})"
    )


# ---------------------------------------------------------------------
# 6. Looking up several words
# ---------------------------------------------------------------------
#
# You can use the same LexiGround object repeatedly.
#

words = [
    "neuroscience",
    "thunder",
    "beautiful",
    "running",
]

print("\nMultiple-word example")
print("=" * 60)

for word in words:
    result = lex.get(word)

    print(f"\n{word}")

    for feature, value in result["features"].items():
        print(
            f"  {feature:23s} "
            f"{value['value']:.3f} "
            f"[{value['source']}]"
        )


# ---------------------------------------------------------------------
# 7. Inspecting one particular feature
# ---------------------------------------------------------------------
#
# If you only care about one dimension, you can extract it from the
# returned dictionary.
#

word = "thunder"
result = lex.get(word)

iconicity = result["features"]["Iconicity"]

print("\nIconicity example")
print("-" * 60)
print(f"Word:   {word}")
print(f"Value:  {iconicity['value']:.3f}")
print(f"Source: {iconicity['source']}")


# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------
#
# LexiGround does not require users to manually download or specify
# paths to the underlying lexical norm CSV files.
#
# The required public datasets are downloaded automatically and cached
# locally on the user's machine.
#
# Human ratings are preferred whenever available. Otherwise,
# LexiGround uses the corresponding pretrained SBERT + Ridge model
# to estimate the missing rating.
#
# See the README for installation instructions, supported features,
# validation results, citations, and data-source attribution.
# ---------------------------------------------------------------------
```
