# The Prescriptive DGA Detector

## Objective

The labs in the "AutoML & Model Interpretability" lecture demonstrated a powerful workflow:

- Rapidly build a high-performance model using AutoML.
- Understand its decisions using Explainable AI (XAI).
- Translate those explanations into actionable instructions using Generative AI.

Integrates building a model using AutoML, explainable AI and translating explanations into actionable instructions using Generative AI into a single, end-to-end Python application. The tool classifies a domain as legitimate or DGA, explains why it made that decision, and automatically generates a specific, context-aware incident response playbook.



---

## Setup

**Prerequisites**
- Python 3.11
- Java 8+ JRE (required by H2O backend)
- OpenAI API key for prescriptive playbooks

**Install**
- Create a virtual env and activate it.
- Run: `pip install -U pip wheel h2o pandas numpy matplotlib shap streamlit`
- (Optional) If you also train with PyCaret: `pip install pycaret`

**Data layout (expected)**
- `./data/dga_dataset_train.csv` with columns: `domain,length,entropy,class`
- `./models/best_dga_model` saved via `h2o.save_model(...)`

---

## Project Structure 
- `1_generate_dga_data.py`
- `2_run_automl.py` — trains AutoML and saves best model
- `3_explain_model.py` 
- `4_generate_prescriptive_playbook.py` - uses your OpenAI key to create a playbook with GenAI
- `models/` — saved H2O model (binary)
- `dga_dataset_train.csv`
- `TESTING.md` — manual verification guide
- `README.md`


---

## Quickstart
1) **Create the data csv containing DGA and normal URLs**: run `1_generate_dga_data.py` (creates dga_dataset_train.csv)
2) **Train with AutoML**: `python 2_run_automl.py` (saves model to `./models/`).
3) **Generate explanations**: run your SHAP script to create images in `./explain/` (creates shap_force.png and shap_summary.png).
4) **Prescriptive playbook**: implement or run `genai_prescriptions.py` to turn outputs into an incident response plan.



---

## Configuration
- Model path: `./models/best_dga_model`
- Data path: `./data/dga_dataset_train.csv`
- Threshold: default `0.5` for DGA vs legit; tune via validation
- LLM provider: set `OPENAI_API_KEY`

---

## Troubleshooting
- "can't multiply sequence by non-int of type 'float'": ensure `length` and `entropy` are numeric, not strings.
- MOJO vs Binary: load the **binary** model in Python; MOJO zip is for Java scoring.
- Probability column names vary (`p0/p1` vs `pBENIGN/pMALICIOUS`); select dynamically.
- `predict_contributions` works for tree models; ensembles may not support it directly.
- Prevent shutdown prompts with `h2o.shutdown(prompt=False)`.

## Alternative Solution
- **Execute the SEAS8414_Kushmerick_HW9_Steps1-3.ipynb
- **Execute the 4_generate_prescriptive_playbook.py