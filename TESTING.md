# TESTING.md — Manual Verification (1 DGA + 1 Legit Domain)

This guide walks you through a **hands-on test** of the H2O AutoML domain classifier using one DGA-like domain and one legitimate domain. You will:

1) Load the previously saved H2O model (`./models/best_dga_model`).  
2) Compute the two features the model expects (`length`, `entropy`) for two test domains.  
3) Run predictions and check pass/fail criteria.  

> **Assumptions**
> - Your model was saved via `h2o.save_model(..., path="./models", ...)` and (optionally) renamed to `./models/best_dga_model`.
> - Training columns: `domain`, `length`, `entropy`, `class`.
> - Features used: `length`, `entropy`. Target: `class` (factor).

---

## 0) Prerequisites

- **Python** ≥ 3.9+ (3.11)
- **Packages**:
  ```bash
  pip install -U h2o pandas numpy matplotlib
  ```
  pip install shap
  ```

- **Files present**:
  - `./models/best_dga_model`  ← H2O binary model path to load

---

## 1) Choose Test Domains

Use any pair you like. Suggested examples:

- **DGA-like**: `xj3k9qplz.ru`
- **Legit**: `wikipedia.org`

We will compute `length` and `entropy` from the raw domain string (dots removed). If your training pipeline used a different normalization (e.g., stripping TLDs, punycode handling), apply the same here for consistency.

**Feature definitions used here**
- `length`: number of characters in the domain with dots removed, e.g., `"wikipediaorg"` → 12.
- `entropy` (Shannon, bits/char): computed on the character distribution of the dotless domain string.

---

## 2) Run the Quick Manual Test

Create and run the following Python script [`manual_test.py`) or paste it into a REPL:](4_generate_prescriptive_playbook.py)

```python
import os
from math import log2
from collections import Counter
import pandas as pd
import h2o

# ---------- Config ----------
MODEL_PATH = "./models/best_dga_model"  # binary H2O model
TEST_DOMAINS = [
    ("xj3k9qplz.ru",  "DGA_EXPECTED"),   # DGA-like
    ("wikipedia.org",  "LEGIT_EXPECTED")  # Legit
]



