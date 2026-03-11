# ADMET-AI Context

## Overview
ADMET-AI is the **Filtration & Scoring** engine of the Drug Discovery Pipeline. It predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity properties of molecules to filter out non-viable candidates.

## Core Logic
*   **Library**: `admet_ai` (Python package).
*   **Models**: Uses Chemprop-RDKit models trained on TDC datasets.
*   **Entry Points**:
    *   `admet_predict.py`: Main prediction logic.
    *   `admet_model.py`: Model wrapper.

## Role in Pipeline
Receives molecules (SMILES) from generation steps (via `chem-spark` or `stoned-selfies`) and annotates them with ADMET profiles. High-toxicity or low-bioavailability compounds are discarded.
