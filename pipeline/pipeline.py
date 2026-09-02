"""
pipeline.py — Assignment 3: Model the Patient Journey

Same 320 patients and the same chronic-pain label as Assignments 1 and 2 — but
each patient is no longer a single row. ``data/patient_visits.csv`` gives every
patient between 1 and 6 clinical visits (variable length, like real EHR data),
each visit carrying four per-visit measurements:

    days_since_first_visit   0 at the first visit, increasing after that
    pain_score_at_visit      0.0 - 10.0, a noisy reading of that day's pain
    medications_at_visit     count of prescriptions written at the visit
    visit_type_code          1 routine · 2 urgent · 3 specialist · 4 procedure
                             · 5 telehealth   (never 0)

A fixed-size input layer cannot consume a variable-length history directly.
This assignment works through the standard fix: pad every patient's history to a
common length, build a boolean mask that says which time steps are real, split
*patients* (never individual visits) into train/validation, pool a per-time-step
sequence down to one vector per patient while ignoring padded steps, and finally
build — but do NOT train — a PyTorch recurrent classifier (LSTM or GRU); the UI
Explorer compares it against a Conv1D baseline on the same task.

─────────────────────────────────────────────────────────────────────────────
YOU IMPLEMENT 6 FUNCTIONS
─────────────────────────────────────────────────────────────────────────────

NumPy / pandas — padding, masking, splitting, pooling
  1. pad_sequence                 One patient's [T, F] history -> fixed
                                  (max_len, F): pad with zero rows at the END,
                                  or keep the most-recent max_len visits.
  2. build_patient_sequences      Long visits table -> (N, max_len, F) array
                                  plus the length-N array of true visit counts.
  3. create_sequence_mask         (N, max_len) boolean: True where t < length.
  4. patient_level_split          Class-balanced train/val split at the PATIENT
                                  level (Assignment 1's stratified split, moved
                                  from rows to patients).
  5. masked_mean_pooling          (N, T, H) sequence output -> (N, H), averaging
                                  only over real time steps.

PyTorch — build, do NOT train
  6. build_sequence_classifier    An nn.Module: LSTM/GRU (optionally
                                  bidirectional) -> masked mean-pool over real
                                  time steps -> Linear(., 1) -> sigmoid.

Every stub raises ``NotImplementedError`` until you implement it — the unit
tests report which functions still need work.

─────────────────────────────────────────────────────────────────────────────
PROVIDED — do NOT modify
─────────────────────────────────────────────────────────────────────────────
  load_patient_visits, PAIN_KEYWORDS, _VISIT_FEATURE_COLS.
  confusion_counts / precision_recall_f1 / roc_auc are imported from
  metrics_utils.py, carried over unchanged from Assignments 1 and 2.

─────────────────────────────────────────────────────────────────────────────
WORKFLOW
─────────────────────────────────────────────────────────────────────────────
  1. In the UI Explorer, open all 8 concept cards, pick a max_seq_len with the
     1-6 buttons in the Sequence Explorer, then tune the Recurrent Arena
     (cell_type, hidden_units, bidirectional) until the green "Optimal
     Performance Reached!" banner shows.
  2. Fill in get_sandbox_params() with your NetID and those values.
  3. Implement the 6 functions below.
  4. python pipeline/pipeline.py          (informational sanity check)
  5. pytest pipeline/test_pipeline.py -v  (unit tests)
  6. Commit pipeline/pipeline.py and push. The autograder then runs the hidden
     tests -- including a check that a model built + trained from your
     get_sandbox_params() values reproduces your personal oracle's AUC/F1.

Only pipeline/pipeline.py may be edited.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:  # PyTorch is only needed for function 6.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - environments without torch installed
    torch = None
    nn = None

from metrics_utils import confusion_counts, precision_recall_f1, roc_auc


# =============================================================================
# Assignment-3 UI parameters  ← fill in from the UI Explorer
# =============================================================================


def get_sandbox_params() -> dict:
    """Return your NetID and the sequence-model settings that produced the green
    "Optimal Performance Reached!" banner for your personal seed.

    :return: dict with keys 'student_id' (str), 'max_seq_len' (int 1-6),
             'hidden_units' (int), 'cell_type' (str, "lstm" or "gru"),
             'val_fraction' (float, must be 0.20).
    """
    params = {
        "student_id":   "",      # ← your NetID, e.g. "jdoe"
        "max_seq_len":  0,       # ← from the UI Explorer's Sequence Explorer tab
        "hidden_units": 0,       # ← from the UI Explorer's Recurrent Arena tab
        "cell_type":    "",      # ← "lstm" or "gru", from the Recurrent Arena tab
        "val_fraction": 0.20,    # ← must stay 0.20
    }

    if params["val_fraction"] != 0.20:
        raise ValueError(
            "val_fraction must be exactly 0.20 -- the oracle always uses a "
            "20% validation split."
        )
    return params


# ---------------------------------------------------------------------
# Do not change these. Same keyword set as Assignments 1-2; the four
# numeric per-visit feature columns, in the order the models expect them.
# ---------------------------------------------------------------------

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]

_VISIT_FEATURE_COLS = [
    "days_since_first_visit",
    "pain_score_at_visit",
    "medications_at_visit",
    "visit_type_code",
]


# =============================================================================
# PROVIDED HELPER — do not modify
# =============================================================================


def load_patient_visits(
    features_path, visits_path
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """PROVIDED. Load both CSVs and return ``(visits_df, patient_ids, labels)``:

    * ``visits_df`` — the long-format visit table, with the four
      ``_VISIT_FEATURE_COLS`` cast to ``float`` and ``patient_id`` cast to
      ``str``. One row per visit, 1-6 rows per patient.
    * ``patient_ids`` — length-320 array of patient ids (``"P0000"`` ...) in
      ``patient_features.csv`` order. This is the canonical patient order for
      the whole pipeline.
    * ``labels`` — length-320 int array of 0/1, the Assignment 1 chronic-pain
      label rebuilt from ``condition_text`` (index-aligned to ``patient_ids``).
    """
    features_df = pd.read_csv(features_path)
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in PAIN_KEYWORDS) + r")\b"
    compiled = re.compile(pattern)
    labels = np.array(
        [1 if compiled.search(text.lower()) else 0
         for text in features_df["condition_text"].tolist()],
        dtype=int,
    )
    patient_ids = features_df["id"].astype(str).to_numpy()

    visits_df = pd.read_csv(visits_path)
    visits_df["patient_id"] = visits_df["patient_id"].astype(str)
    for col in _VISIT_FEATURE_COLS:
        visits_df[col] = visits_df[col].astype(float)
    return visits_df, patient_ids, labels


# =============================================================================
# FUNCTIONS YOU MUST IMPLEMENT (6) — every stub raises NotImplementedError
# =============================================================================


def pad_sequence(visits: np.ndarray, max_len: int) -> np.ndarray:
    """Pad or truncate one patient's visit history to a fixed length.

    Each patient may have a different number of visits, but the model needs
    every patient sequence to have the same shape: ``(max_len, F)``.

    ``visits`` has shape ``(T, F)``:
      - ``T`` = number of real visits for this patient
      - ``F`` = number of features per visit
      - rows are already sorted from oldest visit to most recent visit

    This function handles three cases:
      * If ``T < max_len``:
        Keep all real visits and add zero rows after them until the sequence
        has ``max_len`` rows.

      * If ``T > max_len``:
        Keep only the most recent ``max_len`` visits by dropping older visits
        from the front.

      * If ``T == max_len``:
        Return the visits unchanged in value, but as a new copied array.

    Example:
        If a patient has 2 visits and ``max_len = 4``, the result keeps the
        2 real visits and adds 2 rows of zeros.

        If a patient has 6 visits and ``max_len = 4``, the result keeps visits
        3, 4, 5, and 6 because they are the most recent visits.

    Hint: ~6-8 lines. You can use ``np.vstack`` with a zero block, or create a
    ``np.zeros`` array and copy the real visits into it. Make sure padding is added
    after the real visits, and when truncating, keep the most recent visits from
    the end of the array.

    :param visits: ``(T, F)`` array of one patient's visits, oldest first.
    :param max_len: Target number of time steps.
    :return: ``(max_len, F)`` array.
    """
    raise NotImplementedError("TODO: implement pad_sequence")


def build_patient_sequences(
    visits_df: pd.DataFrame, patient_ids: List[str], max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the visit-level table into padded patient sequences.

    The input table has one row per patient visit. This function groups those
    rows by patient and creates one fixed-length sequence for each patient, in
    the same order as ``patient_ids``.

    For each patient id in ``patient_ids``:
      1. Select that patient's visits from ``visits_df``.
      2. Sort the visits by ``visit_number`` so they are in time order.
      3. Extract the visit feature columns listed in ``_VISIT_FEATURE_COLS``.
      4. Record the patient's true number of visits before padding/truncation.
      5. Pad or truncate the visit array using ``pad_sequence(..., max_len)``.

    The final output contains:
      - ``sequences``: a float array of shape ``(N, max_len, F)``, where
        ``N`` is the number of patients and ``F`` is the number of visit features.
      - ``lengths``: an integer array of shape ``(N,)`` containing each patient's
        true visit count before padding or truncation.
    
    Hint: ~10-14 lines. A simple loop over ``patient_ids`` is expected here. 
    The returned ``lengths`` array is later used to build the sequence mask.

    :param visits_df: Long-format visit table (see ``load_patient_visits``).
    :param patient_ids: Patient ids, in the order rows should appear.
    :param max_len: Fixed sequence length to pad / truncate to.
    :return: ``(sequences, lengths)`` — ``(N, max_len, F)`` float array and a
             length-``N`` int array of true (pre-padding) visit counts.
    """
    raise NotImplementedError("TODO: implement build_patient_sequences")


def create_sequence_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """Build the boolean ``(N, max_len)`` array that marks real visits.

    Each patient may have a different number of visits, but the model expects
    every patient to have the same sequence length, `max_len`. This function
    uses the true visit counts in `lengths` to build a mask of shape
    `(N, max_len)`, where `N` is the number of patients.

    For each patient `i`, the first `lengths[i]` positions are marked as
    `True` because they are real visits. Any remaining positions are marked as
    `False` because they are padding.

    Entry ``(i, t)`` is ``True`` exactly when ``t < lengths[i]`` — the first
    ``lengths[i]`` positions of row ``i`` are real visits, the rest is padding.

    For efficient coding: Use `np.arange(max_len)` and compare it with
    `lengths[:, None]`.

    Hint: ~2-6 lines.    

    :param lengths: Length-``N`` int array of true visit counts.
    :param max_len: Number of columns in the mask.
    :return: ``(N, max_len)`` boolean array, ``True`` at real-visit positions.
    """
    raise NotImplementedError("TODO: implement create_sequence_mask")


def patient_level_split(
    patient_ids: np.ndarray, labels: np.ndarray, val_fraction: float, seed: int
) -> np.ndarray:
    """Create a class-balanced train/validation split at the patient level.

    Each patient is assigned to either the training set or the validation set.
    A patient should never appear in both sets. This is important for sequence
    data because all visits from the same patient must stay together.

    The split is stratified by class label, similar to Assignment 1:
      1. Create a random number generator with ``np.random.default_rng(seed)``.
      2. Process the sorted unique labels in ascending order, so class 0 comes
         before class 1.
      3. For each class, find the patient indices for that class in their
         original order.
      4. Shuffle those indices using ``rng.permutation(...)``.
      5. Mark the first ``round(val_fraction * class_count)`` shuffled indices
         as validation patients.

    ``patient_ids`` is included to match the rest of the pipeline signature.
    The actual split is based on ``labels`` and their positions.

    :param patient_ids: Length-``N`` array of patient ids (canonical order).
    :param labels: Length-``N`` int array of 0/1 patient labels.
    :param val_fraction: Fraction of each class to route to validation.
    :param seed: Seed for ``np.random.default_rng``.
    :return: Length-``N`` boolean array; ``True`` marks a validation patient.
    """
    raise NotImplementedError("TODO: implement patient_level_split")


def masked_mean_pooling(
    sequence_output: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Convert the sequence output for each patient into one fixed-size vector by
    averaging only the real visits and ignoring padded visits.

    `sequence_output` has shape `(N, T, H)`:
        N = number of patients
        T = number of visit positions after padding
        H = number of hidden features produced by the model

    `mask` has shape `(N, T)`:
        True  = this visit position is real
        False = this visit position is padding

    For each patient, average the hidden vectors only at the visit positions
    where the mask is True.

    Example:
        If patient 1 has 3 real visits and 1 padded visit, only the first
        3 visit vectors are averaged. The padded visit is ignored.

    Implementation idea:
        1. Expand the mask from `(N, T)` to `(N, T, 1)` so it can match
           `sequence_output`.
        2. Multiply `sequence_output` by the expanded mask. This keeps real
           visit vectors and turns padded visit vectors into zeros.
        3. Sum across the visit dimension, `axis=1`.
        4. Count how many real visits each patient has.
        5. Divide the summed vectors by the real-visit counts.

    If a patient has zero real visits, use a count of 1 to avoid division by
    zero.

    Hint: ~6-8 lines.

    :param sequence_output: ``(N, T, H)`` float array.
    :param mask: ``(N, T)`` boolean array.
    :return: ``(N, H)`` float array of per-patient pooled vectors.
    """
    raise NotImplementedError("TODO: implement masked_mean_pooling")

def build_sequence_classifier( 
    input_shape: tuple, cell_type: str, hidden_units: int, bidirectional: bool 
) -> "nn.Module": 
    """Build, but do not train, a recurrent classifier for patient visit sequences.

    This model reads padded patient visit histories and returns one probability
    per patient. It is the same type of model used in the UI Explorer's
    Recurrent Arena.

    The input to ``forward(x)`` should have shape ``(batch, max_seq_len, F)``:
      - ``batch`` = number of patients
      - ``max_seq_len`` = fixed number of visit positions
      - ``F`` = number of features per visit

    The output should have shape ``(batch, 1)`` and contain probabilities.

    Model structure:
      1. A recurrent layer:
         - Use ``nn.LSTM`` if ``cell_type == "lstm"``.
         - Use ``nn.GRU`` if ``cell_type == "gru"``.
         - Raise ``ValueError`` for any other ``cell_type``.
         - Use ``batch_first=True``.
         - Use ``input_size=input_shape[1]``.
         - Use ``hidden_size=hidden_units``.
         - Use ``bidirectional=bidirectional``.

      2. In ``forward``:
         - Run the recurrent layer on ``x``.
         - This gives one output vector for each visit position.
         - Build a mask where a visit is real if its input row is not all zeros:
           ``x.abs().sum(dim=-1) > 0``.
         - Use the mask to average only the real visit outputs.
         - Clamp the count of real visits to at least 1 to avoid division by zero.

      3. A final prediction head:
         - Use ``nn.Linear(hidden_units * D, 1)``, where
           ``D = 2 if bidirectional else 1``.
         - Apply sigmoid so the final output is a probability between 0 and 1.

    Do not create an optimizer or loss function here. The caller will train the
    model separately.

    Hint: ~15-20 lines. Write a small ``nn.Module`` subclass with the recurrent
    layer and the linear head created in ``__init__`` and the pooling done in
    ``forward``.

    :param input_shape: ``(max_seq_len, F)`` — only ``F`` is needed here.
    :param cell_type: ``"lstm"`` or ``"gru"``.
    :param hidden_units: Hidden size of the recurrent layer.
    :param bidirectional: Whether the recurrent layer is bidirectional.
    :return: An untrained ``nn.Module``.
    """ 
    raise NotImplementedError("TODO: implement build_sequence_classifier")


# =============================================================================
#
# Local verification  ─  run:  python pipeline/pipeline.py
# Do not modify anything below this line.
#
# =============================================================================


def _seed_from_id(student_id: str) -> int:
    h_val = int(hashlib.sha256(student_id.lower().strip().encode()).hexdigest(), 16)
    return h_val % 900 + 100


def _run_sequence_metrics(features_path, visits_path, seed, max_seq_len, val_fraction):
    """Informational only. Pools the RAW padded visit features (no trained
    model) and scores patients by their mean pain reading, just to exercise
    functions 1-5 end to end. Returns None if a function is still a stub."""
    try:
        visits_df, patient_ids, labels = load_patient_visits(features_path, visits_path)
        sequences, lengths = build_patient_sequences(
            visits_df, list(patient_ids), max_seq_len
        )
        mask = create_sequence_mask(lengths, max_seq_len)
        is_val = patient_level_split(patient_ids, labels, val_fraction, seed)

        pooled = masked_mean_pooling(sequences, mask)          # (N, F)
        pain_idx = _VISIT_FEATURE_COLS.index("pain_score_at_visit")
        scores = pooled[:, pain_idx]

        s_val, y_val = scores[is_val], labels[is_val]
        thresh = float(np.median(scores[~is_val]))
        y_pred = (s_val >= thresh).astype(int)
        tp, fp, tn, fn = confusion_counts(y_val, y_pred)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        pad_frac = 1.0 - float(mask.sum()) / float(mask.size)
        return {
            "shape": sequences.shape,
            "pad_frac": pad_frac,
            "auc": roc_auc(y_val, s_val),
            "accuracy": (tp + tn) / len(y_val),
            "f1": f1,
        }
    except NotImplementedError:
        return None


def _verify() -> None:
    """Local verification — called by __main__. Not part of the graded API."""
    root = Path(__file__).parent.parent
    features_path = root / "data" / "patient_features.csv"
    visits_path = root / "data" / "patient_visits.csv"

    p = get_sandbox_params()
    student_id = p.get("student_id", "").strip()
    max_seq_len = p["max_seq_len"]
    hidden_units = p["hidden_units"]
    cell_type = p["cell_type"]
    val_fraction = p["val_fraction"]

    if not student_id:
        print("\n  Fill in 'student_id' in get_sandbox_params() with your NetID.\n")
        raise SystemExit(1)
    if max_seq_len == 0 or hidden_units == 0 or not cell_type:
        print("\n  Fill in 'max_seq_len', 'hidden_units' and 'cell_type' in\n"
              "  get_sandbox_params() with the values from the UI Explorer.\n")
        raise SystemExit(1)

    seed = _seed_from_id(student_id)
    print(f"\n  Student : {student_id}   Seed: {seed}")
    print(f"  Params  : max_seq_len={max_seq_len}, hidden_units={hidden_units}, "
          f"cell_type={cell_type!r}, val_fraction={val_fraction}\n")

    m = _run_sequence_metrics(features_path, visits_path, seed, max_seq_len, val_fraction)
    if m is None:
        print("  Sequence summary skipped — implement functions 1-5 first.\n")
    else:
        print(f"  Padded sequence shape : {m['shape']}")
        print(f"  Padding fraction      : {m['pad_frac'] * 100:.1f}% of visit slots")
        print()
        print("  Metric    Value   (mean-pain probe, NOT a trained model)")
        print("  --------------------")
        print(f"  {'F1':<9} {m['f1']:.3f}")
        print(f"  {'Accuracy':<9} {m['accuracy']:.3f}")
        print(f"  {'AUC':<9} {m['auc']:.3f}")
        print()

    _verify_model(max_seq_len, cell_type, hidden_units)

    print("  Note: informational only. It does not reveal the hidden grading logic.")
    print("  Run 'pytest pipeline/test_pipeline.py -v' to check your implementation.")


def _verify_model(max_seq_len, cell_type, hidden_units) -> None:
    """Informational summary of build_sequence_classifier (function 6)."""
    if torch is None:
        print("  Model summary skipped — PyTorch is not installed.\n")
        return
    try:
        model = build_sequence_classifier(
            (max_seq_len, len(_VISIT_FEATURE_COLS)), cell_type, hidden_units, False
        )
    except NotImplementedError:
        print("  Model summary skipped — implement build_sequence_classifier first.\n")
        return
    except ValueError as exc:
        print(f"  build_sequence_classifier raised ValueError: {exc}\n")
        return

    module_names = [type(m).__name__ for m in model.modules() if not isinstance(m, nn.Sequential)][1:]
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.no_grad():
        out = model(torch.zeros(4, max_seq_len, len(_VISIT_FEATURE_COLS)))
    print("  Sequence classifier")
    print("  -------------------")
    print(f"  modules     : {' -> '.join(module_names)}")
    print(f"  parameters  : {n_params}")
    print(f"  forward     : zeros(4, {max_seq_len}, {len(_VISIT_FEATURE_COLS)}) -> "
          f"{tuple(out.shape)} in [{float(out.min()):.2f}, {float(out.max()):.2f}]")
    print()


if __name__ == "__main__":
    _verify()
