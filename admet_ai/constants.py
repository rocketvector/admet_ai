"""Contains constants used throughout ADMET-AI."""

import atexit
from contextlib import ExitStack
from importlib import resources

RESOURCE_STACK = ExitStack()
atexit.register(RESOURCE_STACK.close)
RESOURCE_ROOT = resources.files("admet_ai").joinpath("resources")

# Paths to data and models
DEFAULT_ADMET_PATH = RESOURCE_STACK.enter_context(
    resources.as_file(RESOURCE_ROOT.joinpath("data", "admet.csv"))
)
DEFAULT_DRUGBANK_PATH = RESOURCE_STACK.enter_context(
    resources.as_file(RESOURCE_ROOT.joinpath("data", "drugbank_approved.csv"))
)
DEFAULT_MODELS_DIR = RESOURCE_STACK.enter_context(
    resources.as_file(RESOURCE_ROOT.joinpath("models"))
)

# DrugBank columns
DRUGBANK_ID_COLUMN = "id"
DRUGBANK_NAME_COLUMN = "name"
DRUGBANK_SMILES_COLUMN = "smiles"
DRUGBANK_ATC_PREFIX = "atc"
DRUGBANK_ATC_NAME_PREFIX = "atc_name"
DRUGBANK_ATC_CODE_COLUMN = DRUGBANK_ATC_PREFIX
DRUGBANK_DELIMITER = ";"
