import csv
import json
import logging
import os
import re
from typing import List, Tuple, Union

import numpy as np
import torch

from admet_ai.chemprop.data import AtomBondScaler, StandardScaler
from admet_ai.chemprop.model import MoleculeModel, TrainArgs
from admet_ai.chemprop.rdkit import make_mol


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def preprocess_smiles_columns(
    path: str,
    smiles_columns: Union[str, List[str]] = None,
    number_of_molecules: int = 1,
) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None] * number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError(
                    "Length of smiles_columns must match number_of_molecules."
                )
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError(
                    "Provided smiles_columns do not match the header of data file."
                )

    return smiles_columns


def get_mixed_task_names(
    path: str,
    smiles_columns: Union[str, List[str]] = None,
    target_columns: List[str] = None,
    ignore_columns: List[str] = None,
    keep_h: bool = None,
    add_h: bool = None,
    keep_atom_map: bool = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Gets the task names for atomic, bond, and molecule targets separately from a data CSV file.

    If :code:`target_columns` is provided, returned lists based off `target_columns`.
    Otherwise, returned lists based off all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: A tuple containing the task names of atomic, bond, and molecule properties separately.
    """
    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(
            path=path, smiles_columns=smiles_columns
        )

    ignore_columns = set(
        smiles_columns + ([] if ignore_columns is None else ignore_columns)
    )

    if target_columns is not None:
        target_names = target_columns
    else:
        target_names = [column for column in columns if column not in ignore_columns]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_target_names, bond_target_names, molecule_target_names = [], [], []
            smiles = [row[c] for c in smiles_columns]
            mol = make_mol(smiles[0], keep_h, add_h, keep_atom_map)
            for column in target_names:
                value = row[column]
                value = value.replace("None", "null")
                target = np.array(json.loads(value))

                is_atom_target, is_bond_target, is_molecule_target = False, False, False
                if len(target.shape) == 0:
                    is_molecule_target = True
                elif len(target.shape) == 1:
                    if len(mol.GetAtoms()) == len(mol.GetBonds()):
                        break
                    elif len(target) == len(
                        mol.GetAtoms()
                    ):  # Atom targets saved as 1D list
                        is_atom_target = True
                    elif len(target) == len(
                        mol.GetBonds()
                    ):  # Bond targets saved as 1D list
                        is_bond_target = True
                elif len(target.shape) == 2:  # Bond targets saved as 2D list
                    is_bond_target = True
                else:
                    raise ValueError(
                        "Unrecognized targets of column {column} in {path}."
                    )

                if is_atom_target:
                    atom_target_names.append(column)
                elif is_bond_target:
                    bond_target_names.append(column)
                elif is_molecule_target:
                    molecule_target_names.append(column)
            if len(atom_target_names) + len(bond_target_names) + len(
                molecule_target_names
            ) == len(target_names):
                break

    return atom_target_names, bond_target_names, molecule_target_names


def load_args(path: str) -> TrainArgs:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The :class:`~chemprop.args.TrainArgs` object that the model was trained with.
    """
    args = TrainArgs()
    args.from_dict(
        vars(torch.load(path, map_location=lambda storage, loc: storage)["args"]),
        skip_unsettable=True,
    )

    return args


def load_checkpoint(
    path: str, device: torch.device = None, logger: logging.Logger = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if (
            re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name)
            and not args.reaction_solvent
        ):
            param_name = loaded_param_name.replace(
                "encoder.encoder", "encoder.encoder.0"
            )
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif (
            model_state_dict[param_name].shape
            != loaded_state_dict[loaded_param_name].shape
        ):
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    return model


def load_scalers(
    path: str,
) -> Tuple[
    StandardScaler, StandardScaler, StandardScaler, StandardScaler, List[StandardScaler]
]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    if state["data_scaler"] is not None:
        scaler = StandardScaler(
            state["data_scaler"]["means"], state["data_scaler"]["stds"]
        )
    else:
        scaler = None

    if state["features_scaler"] is not None:
        features_scaler = StandardScaler(
            state["features_scaler"]["means"],
            state["features_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        features_scaler = None

    if (
        "atom_descriptor_scaler" in state.keys()
        and state["atom_descriptor_scaler"] is not None
    ):
        atom_descriptor_scaler = StandardScaler(
            state["atom_descriptor_scaler"]["means"],
            state["atom_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        atom_descriptor_scaler = None

    if (
        "bond_descriptor_scaler" in state.keys()
        and state["bond_descriptor_scaler"] is not None
    ):
        bond_descriptor_scaler = StandardScaler(
            state["bond_descriptor_scaler"]["means"],
            state["bond_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        bond_descriptor_scaler = None

    if "atom_bond_scaler" in state.keys() and state["atom_bond_scaler"] is not None:
        atom_bond_scaler = AtomBondScaler(
            state["atom_bond_scaler"]["means"],
            state["atom_bond_scaler"]["stds"],
            replace_nan_token=0,
            n_atom_targets=len(state["args"].atom_targets),
            n_bond_targets=len(state["args"].bond_targets),
        )
    else:
        atom_bond_scaler = None

    return (
        scaler,
        features_scaler,
        atom_descriptor_scaler,
        bond_descriptor_scaler,
        atom_bond_scaler,
    )
