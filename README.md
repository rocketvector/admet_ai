> [!IMPORTANT]
> This repository is superseded. Canonical development moved to
> [`rocketvector/drug/workflows/molecule/components/admet`](https://github.com/rocketvector/drug/tree/main/workflows/molecule/components/admet).
> Do not open new code changes here; archival is tracked in
> [`rocketvector/drug#1286`](https://github.com/rocketvector/drug/issues/1286).
> Signposted: `2026-08-15`.
>
> Pre-signpost source: `f5ac718d9fd14bb770a5dff258ee003929772c1a`
> (tree `91d00d5d16fe55ddf291c842801287b8b1a8c8ef`). Drug imported it in
> `044f14533df937a9fb9387cdbcf33232de7a4e59`, moved it in
> `57c3fc9dea9aa4ce6641eb889705b91b60446a41`, and released terminal source
> `1123a6a56e930ba5963ea2e1604ddcdda2f934f3`. The `admet_ai` compatibility
> import/distribution remains available from Drug; publication is root-owned.
> Recovery bundle:
> `gs://rocketvector-drug-code/repository-mirrors/20260815-restructure-7c9c05e/pre-signpost/bundles/admet_ai-pre-signpost-7c9c05e.bundle`
> (`sha256:764948c8909de0ec7fceb0dc290c88fafeaf2c0fe79d1c1dde13308cdb7674fb`).
> This repository remains writable only for the required 14-day no-write soak,
> then will be archived, never deleted.

# ADMET-AI

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/admet_ai)](https://badge.fury.io/py/admet_ai)
[![PyPI version](https://badge.fury.io/py/admet_ai.svg)](https://badge.fury.io/py/admet_ai)
[![Downloads](https://pepy.tech/badge/admet_ai)](https://pepy.tech/project/admet_ai)
[![license](https://img.shields.io/github/license/swansonk14/admet_ai.svg)](https://github.com/swansonk14/admet_ai/blob/main/LICENSE.txt)

This git repo contains the code for ADMET-AI, an ADMET prediction platform that
uses [Chemprop-RDKit]((https://github.com/chemprop/chemprop)) models trained on ADMET datasets from the Therapeutics
Data Commons ([TDC](https://tdcommons.ai/)). ADMET-AI can be used to make ADMET predictions on new molecules via the
command line, via the Python API, or via a web server. A live web server hosting ADMET-AI is
at [admet.ai.greenstonebio.com](https://admet.ai.greenstonebio.com)

Please see the following paper and [this blog post](https://portal.valencelabs.com/blogs/post/admet-ai-a-machine-learning-admet-platform-for-evaluation-of-large-scale-QPEa0j5OTYYHTaA) for more
details, and please cite us if ADMET-AI is useful in your work. Instructions to reproduce the results in our paper are in [docs/reproduce.md](docs/reproduce.md).

[ADMET-AI: A machine learning ADMET platform for evaluation of large-scale chemical libraries](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btae416/7698030?utm_source=authortollfreelink&utm_campaign=bioinformatics&utm_medium=email&guestAccessKey=f4fca1d2-49ec-4b10-b476-5aea3bf37045)


## Table of Contents

- [Installation](#installation)
- [Predicting ADMET properties](#predicting-admet-properties)
    * [Command line tool](#command-line-tool)
    * [Python module](#python-module)
    * [Web server](#web-server)

## Installation

ADMET-AI can be installed in a few minutes on any operating system using pip (optionally within a conda environment). If
a GPU is available, it will be used by default, but the code can also run on CPUs only.

Optionally, create a conda environment.

```bash
conda create -y -n admet_ai python=3.10
conda activate admet_ai
```

Install ADMET-AI via pip.

```bash
pip install admet-ai
```

Alternatively, clone the repo and install ADMET-AI locally.

```bash
git clone https://github.com/swansonk14/admet_ai.git
cd admet_ai
pip install -e .
```

By default, the pip installation only includes dependencies required for making ADMET predictions, either via the
command line or via the Python API. To install dependencies required for processing TDC data or plotting TDC results,
run `pip install admet-ai[tdc]`. To install dependencies required for hosting the ADMET-AI web server,
run `pip install admet-ai[web]`.

If there are version issues with the required packages, create a conda environment with specific working versions of the
packages as follows.

```bash
pip install -r requirements.txt
pip install -e .
```

Note: If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`,
run `conda install -c conda-forge xorg-libxrender`.

## Predicting ADMET properties

ADMET-AI can be used to make ADMET predictions in three ways: (1) as a command line tool, (2) as a Python module, or (3)
as a web server.

### Command line tool

ADMET predictions can be made on the command line with the `admet_predict` command, as illustrated below.

```bash
admet_predict \
    --data_path data.csv \
    --save_path preds.csv \
    --smiles_column smiles
```

This command assumes that there exists a file called `data.csv` with SMILES strings in the column `smiles`. The
predictions will be saved to a file called `preds.csv`.

### Python module

ADMET predictions can be made using the `predict` function in the `admet_ai` Python module, as illustrated below.

```python
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")
```

If a SMILES string is provided, then `preds` is a dictionary mapping property names to values. If a list of SMILES
strings is provided, then `preds` is a Pandas DataFrame where the index is the SMILES and the columns are the
properties.

### Web server

ADMET predictions can be made using the ADMET-AI web server, as illustrated below. Note: Running the following command
requires additional web dependencies (i.e., `pip install admet-ai[web]`).

```bash
admet_web
```

Then navigate to http://127.0.0.1:5000 to view the website.

### Analysis plots

The DrugBank reference plot and radial plots displayed on the ADMET-AI website can be generated locally using the
`scripts/plot_drugbank_reference.py` and `scripts/plot_radial_summaries.py` scripts, respectively. Both scripts
take as input a CSV file with ADMET-AI predictions along with other parameters.
