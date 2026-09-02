# Assignment 3: Model the Patient Journey

## Overview

You are the same analytics engineer from Assignments 1 and 2. Same clinical
scenario, the same 320-patient synthetic EHR dataset, the same chronic-pain
binary label, the same personal seed derived from your NetID. **What changes is
the shape of the data.**

Each patient is no longer a single row. A companion file,
`data/patient_visits.csv`, gives every patient between **1 and 6 clinical
visits** (variable length, like real EHR data). Each visit carries four
per-visit measurements:

| column | meaning |
|---|---|
| `days_since_first_visit` | 0 at the first visit, increasing after that |
| `pain_score_at_visit` | 0.0–10.0, a **noisy** reading of that day's pain |
| `medications_at_visit` | prescriptions written at the visit |
| `visit_type_code` | 1 routine · 2 urgent · 3 specialist · 4 procedure · 5 telehealth |

A fixed-size input layer cannot consume a variable-length history directly. This
assignment works through the standard fix:

1. **Pad / truncate** every patient's history to one common length.
2. **Mask** — build a boolean array marking real visits vs. padding.
3. **Split patients** (never individual visits) into train / validation.
4. **Masked pooling** — collapse a per-time-step sequence to one vector per
   patient, ignoring padded steps.
5. **Build (don't train)** a PyTorch recurrent classifier (LSTM or GRU); the UI
   compares it against a Conv1D baseline — the "architecture must match the data
   structure" lesson, applied to patient visit histories.

| Part | What it is |
|---|---|
| UI Explorer (hosted, not part of this repo) | Explore each patient's visit timeline, experiment with padding length, and run a live *Recurrent vs. Convolutional* arena against your personal oracle |
| `pipeline/` | Python stubs you implement in this repo to reproduce those results programmatically |

**Contents:** [Setup](#setup) · [Part 1 — UI Explorer](#part-1--ui-explorer-start-here) · [Part 2 — pipeline.py](#part-2--implement-pipelinepy) · [Running tests locally](#running-tests-locally) · [Submit your work](#submit-your-work) · [Folder structure](#folder-structure)

---

## Setup

Requires **Python 3.10 – 3.13** (PyTorch wheels are not yet published for 3.14).

```bash
python -m venv myenv
# macOS / Linux
source myenv/bin/activate
# Windows
myenv\Scripts\activate

pip install -r requirements.txt
```

---

## Part 1 — UI Explorer (start here)

The UI Explorer is an interactive browser app where you explore the visit
timelines, see the padding/masking trade-off, and tune a recurrent sequence
classifier for your personal seed. Nothing to install for the hosted version.

**UI Explorer:** https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-3/

### What to do in the UI Explorer

1. **Overview & Concepts tab** — open all 8 concept cards.
2. **Patient Timeline Explorer tab** — click through a few patients with
   different visit counts and read the sequence-length histogram.
3. **Sequence Explorer tab** — pick a `max_seq_len` with the **1–6 buttons** and
   watch the *padding fraction* and per-patient preview update; check the worked
   2-patient / 3-visit mini-example against your own understanding.
4. **Recurrent Arena tab** — with your `max_seq_len` chosen, tune `cell_type`,
   `hidden_units`, and `bidirectional` until the green **"Optimal Performance
   Reached!"** banner appears, comparing against the always-visible Conv1D
   baseline. Copy the winning `max_seq_len`, `hidden_units`, and `cell_type`.

---

## Part 2 — Implement `pipeline.py`

### What you implement

Open `pipeline/pipeline.py` and implement the 6 stub functions.

**NumPy / pandas — padding, masking, splitting, pooling**

| # | Function | What it does |
|---|---|---|
| 1 | `pad_sequence` | One patient's `[T, F]` history → fixed `(max_len, F)`: pad zero rows at the **end**, or keep the most-recent `max_len` visits |
| 2 | `build_patient_sequences` | Long visits table → `(N, max_len, F)` array + the length-`N` array of true visit counts |
| 3 | `create_sequence_mask` | `(N, max_len)` boolean, `True` where `t < length` (fully vectorised) |
| 4 | `patient_level_split` | Class-balanced train/val split at the **patient** level (Assignment 1's stratified split, moved from rows to patients) |
| 5 | `masked_mean_pooling` | `(N, T, H)` sequence output → `(N, H)`, averaging over real time steps only |

**PyTorch — build, do NOT train**

| # | Function | What it does |
|---|---|---|
| 6 | `build_sequence_classifier` | An `nn.Module`: `nn.LSTM`/`nn.GRU` (`batch_first=True`, optionally `bidirectional`) → masked mean-pool over non-padding time steps → `nn.Linear(·, 1)` → sigmoid. No optimiser/loss — the caller trains it. |

`load_patient_visits` is **provided — do not modify it**.
`pipeline/metrics_utils.py` (`confusion_counts`, `precision_recall_f1`,
`roc_auc`) is carried over unchanged from Assignments 1 and 2 — **import it, do
not reimplement it**.

### Declare your UI Explorer parameters

Near the top of `pipeline/pipeline.py`, fill in `get_sandbox_params()`:

```python
def get_sandbox_params() -> dict:
    return {
        "student_id":   "",     # ← your NetID, e.g. "jdoe"
        "max_seq_len":  0,      # ← from the UI Explorer's Sequence Explorer tab
        "hidden_units": 0,      # ← from the UI Explorer's Recurrent Arena tab
        "cell_type":    "",     # ← "lstm" or "gru", from the Recurrent Arena tab
        "val_fraction": 0.20,   # ← must stay 0.20
    }
```

`val_fraction` must stay exactly `0.20` — `get_sandbox_params()` raises a
`ValueError` otherwise, since the oracle always uses a 20% validation split.

**These four values are read by the hidden autograder, not just by you.** The
grader hashes your `student_id` to your personal seed, builds a train/validation
split and padded sequences at your `max_seq_len`, builds
`build_sequence_classifier` with your `hidden_units` / `cell_type`, trains it
with a fixed deterministic loop, and checks that its validation **AUC and F1**
land within tolerance of your seed's oracle (20 points). The local unit tests
only check that `max_seq_len` is 1–6, `hidden_units > 0`, and
`cell_type ∈ {"lstm", "gru"}` — so a wrong-but-valid value passes locally and
still loses those 20 points on the autograder. Copy the exact values the UI's
green **"Optimal Performance Reached!"** banner shows for your seed.

### Verify locally

```bash
python pipeline/pipeline.py
```

Example output:

```
  Student : jdoe   Seed: 254
  Params  : max_seq_len=6, hidden_units=24, cell_type='lstm', val_fraction=0.2

  Padded sequence shape : (320, 6, 4)
  Padding fraction      : 41.8% of visit slots

  Metric    Value   (mean-pain probe, NOT a trained model)
  --------------------
  F1        0.759
  Accuracy  0.781
  AUC       0.825

  Sequence classifier
  -------------------
  modules     : LSTM -> Linear -> Sigmoid
  parameters  : 2905
  forward     : zeros(4, 6, 4) -> (4, 1) in [0.53, 0.53]
```

This local output is informational only and does not reveal hidden test code.

---

## Running tests locally

```bash
pytest pipeline/test_pipeline.py -v
```

`conftest.py` runs first and aborts the whole run with an error if you modified
any provided helper (`load_patient_visits`, `PAIN_KEYWORDS`,
`_VISIT_FEATURE_COLS`) — revert the change if that happens.

Every stub raises `NotImplementedError` until you implement it; its test then
fails with a plain `... is not implemented yet` message (no traceback) —
expected for a stub, not a bug. Implement the function and re-run. 

Run a single test while you work on one function:

```bash
pytest pipeline/test_pipeline.py::test_pad_sequence_truncates_keeping_the_tail -v
```

---

## Submit your work

```bash
git add pipeline/pipeline.py
git commit -m "Implement pipeline.py"
git push
```

Pushing triggers the `Autograding` GitHub Actions workflow, which clones the
**hidden grader** and runs it against your `pipeline/pipeline.py`. Check the
**Actions** tab on your repo for the score.

### How it is graded (100 points)

| Component | Points | What it checks |
|---|---:|---|
| Pipeline unit tests | 70 | `pad_sequence`, `build_patient_sequences`, `create_sequence_mask`, `patient_level_split`, `masked_mean_pooling` on hand-computed toy inputs, plus the structure of `build_sequence_classifier` |
| Sequence model reproduces your oracle | 20 | The grader builds sequences at **your** `max_seq_len`, builds the model with **your** `hidden_units` / `cell_type`, trains it deterministically, and requires its validation **AUC** and **F1** to be within tolerance of your personal-seed oracle |
| Code quality | 10 | Every function keeps its docstring; no function is left as an unimplemented stub |

The grader ships its own copy of `data/`, `metrics_utils.py` and the reference
solution; it never uses anything you submit except `pipeline/pipeline.py`
(and, from it, only your `get_sandbox_params()` values and the 6 functions).

---

## Folder structure

```
neural-network-assignment-3/
├── README.md                  This file
├── requirements.txt           pip install -r requirements.txt
├── .github/workflows/
│   └── autograding.yml        Runs the hidden grader on push (do not edit)
├── data/
│   ├── patient_features.csv   320 synthetic patients (do not edit)
│   └── patient_visits.csv     1,117 visit rows, 1–6 per patient (do not edit)
└── pipeline/
    ├── pipeline.py            The only file you edit
    ├── metrics_utils.py       Assignment 1 metrics, imported (do not edit)
    ├── test_pipeline.py       14 unit tests (do not edit)
    └── conftest.py            Guards the provided helpers (do not edit)
```
