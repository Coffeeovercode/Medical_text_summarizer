#  Medical Text Summarizer — T5 

## TL;DR

This repo contains a lightweight **medical text summarizer** implemented with a fine-tuned **T5** model (default: `t5-small`). It includes an executable CLI (`main.py`) to:

(1) batch-summarize clinical notes from a CSV file and 
(2) run single-note Q\&A. 

The code is intentionally compact and readable for simpler implementation.
---

## Contents of this README

1. Repo overview & quick demo
2. Project structure 
3. Setup & run commands (exact commands for this repo)
4. Input/Output format examples
5. `config.py` explained (parameters used)
8. Limitations & ethical notes

---

## Repo structure 

```
MEDICAL_TEXT_SUMMARIZER/
├─ .gitignore
├─ config.py            # model & generation settings (provided)
├─ main.py              # CLI + MedicalNLP class (provided)
├─ README.md            # (this file)
├─ requirements.txt
├─ data/                # put input CSVs here (not tracked)
└─ output/              # outputs will be written here
```

---

## Quickstart — install & run 

```bash
# 1) clone repo (if not already)
git clone https://github.com/yourusername/medical-text-summarizer.git
cd medical-text-summarizer

# 2) create virtualenv and install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) run summarization (uses config.DEFAULT paths by default)
python main.py summarize

# or specify files explicitly
python main.py summarize \
  --input_file data/Synthetic\ Medical\ Notes.csv \
  --output_file output/summarized_medical_notes.csv

# 4) run QA on a single note
python main.py qa \
  --note "65-year-old male with chest pain, troponin elevated." \
  --question "What is the suspected diagnosis?"
```

---

## `requirements.txt` 

Make sure `requirements.txt` includes at least:

```
transformers>=4.0.0
torch>=1.7.0
pandas
tqdm
```

(Adjust versions to match your runtime environment / CUDA.)

---

## Input CSV format — exact expectation

`main.py` expects a CSV with a column named exactly: `clinical_note`.

Example `data/Synthetic Medical Notes.csv` (CSV header + one row):

```csv
id,clinical_note
1,"Patient is a 45-year-old female with 3 days of cough and fever. CXR shows right lower lobe consolidation. Started on IV antibiotics. Plan: observe, follow labs."
```

Output will be the same CSV with a new `summary` column:

```csv
id,clinical_note,summary
1,"Patient is a 45-year-old female ...","45F with RLL consolidation; started IV abx; observe & follow labs."
```

---

## `config.py` — what each parameter does (tailored to repo)

```py
# MODEL_NAME = "t5-small"
# DEFAULT_INPUT_PATH = "data/Synthetic Medical Notes.csv"
# DEFAULT_OUTPUT_PATH = "output/summarized_medical_notes.csv"
# MAX_INPUT_LENGTH = 512
# MAX_OUTPUT_LENGTH = 150
```

* `MODEL_NAME`: Hugging Face model identifier used by `MedicalNLP`. Default `t5-small`. Change to `t5-base`/`t5-large` if you have more compute.
* `DEFAULT_INPUT_PATH` / `DEFAULT_OUTPUT_PATH`: CLI defaults for summarize command.
* `MAX_INPUT_LENGTH`: Tokenization + truncation length for inputs.
* `MAX_OUTPUT_LENGTH`: Maximum length for generated summaries.

> Tip: If you want dynamic override without editing `config.py`, add a CLI arg later.

---

## `main.py` Walkthrough -

**Key class & functions**

* `class MedicalNLP`

  * `__init__(model_name)`: loads tokenizer & model, sets `device` (cpu/cuda).
  * `summarize(text)`: constructs `"summarize: {text}"` prompt, tokenizes (truncates), generates with settings from `config.py`, decodes result.
  * `ask_question(note, question)`: constructs `"question: {question} context: {note}"` prompt and generates an answer.
* CLI handling (argparse):

  * `summarize` subcommand: expects `--input_file` and `--output_file` (defaults from `config.py`). Applies `processor.summarize` to each row in `clinical_note` column and writes `summary` column.
  * `qa` subcommand: runs `processor.ask_question` on one note/question pair.

**What to highlight:**

* Simple, reproducible, and easily extensible architecture (single class + CLI).
* Clear input contract (`clinical_note` column) — look at how errors are handled (`FileNotFoundError`, missing column).
* Where to add preprocessing / PHI removal (before `df['clinical_note'].apply(...)`).

---

## Future ideas for improvement

* Add **preprocessing** pipeline: PHI removal, abbreviation normalization, numeric normalization.
* Add **sliding window** + aggregation to handle very long notes.
* Add **constrained decoding** or entity-copy mechanism to reduce hallucinations.
* Track experiments with `wandb` / `tensorboard`.
* Improve CLI to allow overriding `MODEL_NAME`, `MAX_INPUT_LENGTH`, etc., without editing `config.py`.
* Save and load fine-tuned checkpoints and add a `--model` CLI param.
* Unit tests for data I/O and a small sample integration test (fast).

---

## Limitations & ethical considerations

* **PHI**: Do **not** run on PHI without approvals & de-identification. This repo contains no de-id code by default.
* **Clinical risk**: Outputs are assistive and must be verified by clinicians before any deployment into care.
* **Bias**: Dataset biases will influence outputs. 

---

## Contact

Author: **Supreet Sarita Das** — `esupreetsaritadas@gmail.com` • GitHub: `@Coffeeovercode`

---
