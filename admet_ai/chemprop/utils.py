import csv
import json
import os
from typing import List, Tuple, Union

import numpy as np

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
