import json
import os
import pickle
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from tap import Tap

from admet_ai.chemprop.data import empty_cache, set_cache_mol
from admet_ai.chemprop.features import (
    BatchMolGraph,
    get_atom_fdim,
    get_available_features_generators,
    get_bond_fdim,
    mol2graph,
)
from admet_ai.chemprop.nn_utils import (
    MultiReadout,
    build_ffn,
    get_activation_function,
    index_select_ND,
    initialize_weights,
)
from admet_ai.chemprop.utils import (
    get_header,
    get_mixed_task_names,
    preprocess_smiles_columns,
)

Metric = Literal[
    "auc",
    "prc-auc",
    "rmse",
    "mae",
    "mse",
    "r2",
    "accuracy",
    "cross_entropy",
    "binary_cross_entropy",
    "sid",
    "wasserstein",
    "f1",
    "mcc",
    "bounded_rmse",
    "bounded_mae",
    "bounded_mse",
]


def get_checkpoint_paths(
    checkpoint_path: Optional[str] = None,
    checkpoint_paths: Optional[List[str]] = None,
    checkpoint_dir: Optional[str] = None,
    ext: str = ".pt",
) -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if (
        sum(
            var is not None
            for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]
        )
        > 1
    ):
        raise ValueError(
            "Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths"
        )

    if checkpoint_path is not None:
        return [checkpoint_path]

    if checkpoint_paths is not None:
        return checkpoint_paths

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith(ext):
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(
                f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"'
            )

        return checkpoint_paths

    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal["feature", "descriptor"] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_descriptors: Literal["feature", "descriptor"] = None
    """
    Custom extra bond descriptors.
    :code:`feature`: used as bond features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned bond representation.
    """
    bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """
    constraints_path: str = None
    """
    Path to constraints applied to atomic/bond properties prediction.
    """

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0
        self._bond_descriptors_size = 0
        self._atom_constraints = []
        self._bond_constraints = []

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device("cpu")

        return torch.device("cuda", self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    @property
    def bond_descriptors_size(self) -> int:
        """The size of the bond descriptors."""
        return self._bond_descriptors_size

    @bond_descriptors_size.setter
    def bond_descriptors_size(self, bond_descriptors_size: int) -> None:
        self._bond_descriptors_size = bond_descriptors_size

    def configure(self) -> None:
        self.add_argument("--gpu", choices=list(range(torch.cuda.device_count())))
        self.add_argument(
            "--features_generator", choices=get_available_features_generators()
        )

    def process_args(self) -> None:
        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Validate features
        if (
            self.features_generator is not None
            and "rdkit_2d_normalized" in self.features_generator
            and self.features_scaling
        ):
            raise ValueError(
                "When using rdkit_2d_normalized features, --no_features_scaling must be specified."
            )

        # Validate atom descriptors
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError(
                "If atom_descriptors is specified, then an atom_descriptors_path must be provided "
                "and vice versa."
            )

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError(
                "Atom descriptors are currently only supported with one molecule "
                "per input (i.e., number_of_molecules = 1)."
            )

        # Validate bond descriptors
        if (self.bond_descriptors is None) != (self.bond_descriptors_path is None):
            raise ValueError(
                "If bond_descriptors is specified, then an bond_descriptors_path must be provided "
                "and vice versa."
            )

        if self.bond_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError(
                "Bond descriptors are currently only supported with one molecule "
                "per input (i.e., number_of_molecules = 1)."
            )

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal["regression", "classification", "multiclass", "spectra"]
    """Type of dataset. This determines the default loss function used during training."""
    loss_function: Literal[
        "mse",
        "bounded_mse",
        "binary_cross_entropy",
        "cross_entropy",
        "mcc",
        "sid",
        "wasserstein",
        "mve",
        "evidential",
        "dirichlet",
    ] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    spectra_phase_mask_path: str = None
    """Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."""
    data_weights_path: str = None
    """Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function"""
    target_weights: List[float] = None
    """Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns."""
    split_type: Literal[
        "random",
        "scaffold_balanced",
        "predetermined",
        "crossval",
        "cv",
        "cv-no-test",
        "index_predetermined",
        "random_with_repeated_smiles",
    ] = "random"
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    bias_solvent: bool = False
    """Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True."""
    hidden_size_solvent: int = 300
    """Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True."""
    depth_solvent: int = 3
    """Number of message passing steps for solvent if :code:`reaction_solvent` is True."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal["ReLU", "LeakyReLU", "PReLU", "tanh", "SELU", "ELU"] = "ReLU"
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_phase_features_path: str = None
    """Path to file with phase features for separate val set."""
    separate_test_phase_features_path: str = None
    """Path to file with phase features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_constraints_path: str = None
    """Path to file with constraints for separate val set."""
    separate_test_constraints_path: str = None
    """Path to file with constraints for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal["mean", "sum", "norm"] = "mean"
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    reaction_mode: Literal[
        "reac_prod",
        "reac_diff",
        "prod_diff",
        "reac_prod_balance",
        "reac_diff_balance",
        "prod_diff_balance",
    ] = "reac_diff"
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    """
    reaction_solvent: bool = False
    """
    Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    """
    explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.
    """
    adding_h: bool = False
    """
    Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.
    """
    is_atom_bond_targets: bool = False
    """
    whether this is atomic/bond properties prediction.
    """
    keeping_atom_map: bool = False
    """
    Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with
    the :code:`is_atom_bond_targets`.
    """
    no_shared_atom_bond_ffn: bool = False
    """
    Whether the FFN weights for atom and bond targets should be independent between tasks.
    """
    weights_ffn_num_layers: int = 2
    """
    Number of layers in FFN for determining weights used in constrained targets.
    """
    no_adding_bond_types: bool = False
    """
    Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used
    with the :code:`is_atom_bond_targets`.
    """

    # Training arguments
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""
    spectra_activation: Literal["exp", "softplus"] = "exp"
    """Indicates which function to use in dataset_type spectra training to constrain outputs to be positive."""
    spectra_target_floor: float = 1e-8
    """Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values."""
    evidential_regularization: float = 0
    """Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. 
    Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model."""
    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """
    Overwrites the default bond descriptors with the new ones instead of concatenating them.
    Can only be used if bond_descriptors are used as a feature.
    """
    no_bond_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {
            "rmse",
            "mae",
            "mse",
            "cross_entropy",
            "binary_cross_entropy",
            "sid",
            "wasserstein",
            "bounded_mse",
            "bounded_mae",
            "bounded_rmse",
        }

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return (
            self.features_generator is not None
            or self.features_path is not None
            or self.phase_features_path is not None
        )

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional atom features."
        """
        return not self.no_atom_descriptor_scaling

    @property
    def bond_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional bond features."
        """
        return not self.no_bond_descriptor_scaling

    @property
    def shared_atom_bond_ffn(self) -> bool:
        """
        Whether the FFN weights for atom and bond targets should be shared between tasks.
        """
        return not self.no_shared_atom_bond_ffn

    @property
    def adding_bond_types(self) -> bool:
        """
        Whether the bond types determined by RDKit molecules should be added to the output of bond targets.
        """
        return not self.no_adding_bond_types

    @property
    def atom_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of atomic properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._atom_constraints:
                header = get_header(self.constraints_path)
                self._atom_constraints = [
                    target in header for target in self.atom_targets
                ]
        else:
            self._atom_constraints = [False] * len(self.atom_targets)
        return self._atom_constraints

    @property
    def bond_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of bond properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._bond_constraints:
                header = get_header(self.constraints_path)
                self._bond_constraints = [
                    target in header for target in self.bond_targets
                ]
        else:
            self._bond_constraints = [False] * len(self.bond_targets)
        return self._bond_constraints

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_save_dir  # Prevents the temporary directory from being deleted upon function return

        # Adapt the number of molecules for reaction_solvent mode
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError(
                "In reaction_solvent mode, --number_of_molecules 2 must be specified."
            )

        # Process SMILES columns
        self.smiles_columns = preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        # Load config file
        if self.config_path is not None:
            with open(self.config_path) as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)

        # Determine the target_columns when training atomic and bond targets
        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets, self.molecule_targets = (
                get_mixed_task_names(
                    path=self.data_path,
                    smiles_columns=self.smiles_columns,
                    target_columns=self.target_columns,
                    ignore_columns=self.ignore_columns,
                    keep_h=self.explicit_h,
                    add_h=self.adding_h,
                    keep_atom_map=self.keeping_atom_map,
                )
            )
            self.target_columns = self.atom_targets + self.bond_targets
            # self.target_columns = self.atom_targets + self.bond_targets + self.molecule_targets  # TODO: Support mixed targets
        else:
            self.atom_targets, self.bond_targets = [], []

        # Check whether atomic/bond constraints have been applied on the correct dataset_type
        if self.constraints_path:
            if not self.is_atom_bond_targets:
                raise ValueError(
                    "Constraints on atomic/bond targets can only be used in atomic/bond properties prediction."
                )
            if self.dataset_type != "regression":
                raise ValueError(
                    f"In atomic/bond properties prediction, atomic/bond constraints are not supported for {self.dataset_type}."
                )

        # Check whether the number of input columns is one for the atomic/bond mode
        if self.is_atom_bond_targets:
            if self.number_of_molecules != 1:
                raise ValueError(
                    "In atomic/bond properties prediction, exactly one smiles column must be provided."
                )

        # Check whether the number of input columns is two for the reaction_solvent mode
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError(
                "In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)"
            )

        # Validate reaction/reaction_solvent mode
        if self.reaction is True and self.reaction_solvent is True:
            raise ValueError(
                "Only reaction or reaction_solvent mode can be used, not both."
            )

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_save_dir = TemporaryDirectory()
            self.save_dir = temp_save_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == "classification":
                self.metric = "auc"
            elif self.dataset_type == "multiclass":
                self.metric = "cross_entropy"
            elif self.dataset_type == "spectra":
                self.metric = "sid"
            elif (
                self.dataset_type == "regression"
                and self.loss_function == "bounded_mse"
            ):
                self.metric = "bounded_mse"
            elif self.dataset_type == "regression":
                self.metric = "rmse"
            else:
                raise ValueError(f"Dataset type {self.dataset_type} is not supported.")

        if self.metric in self.extra_metrics:
            raise ValueError(
                f"Metric {self.metric} is both the metric and is in extra_metrics. "
                f"Please only include it once."
            )

        for metric in self.metrics:
            if not any(
                [
                    (
                        self.dataset_type == "classification"
                        and metric
                        in [
                            "auc",
                            "prc-auc",
                            "accuracy",
                            "binary_cross_entropy",
                            "f1",
                            "mcc",
                        ]
                    ),
                    (
                        self.dataset_type == "regression"
                        and metric
                        in [
                            "rmse",
                            "mae",
                            "mse",
                            "r2",
                            "bounded_rmse",
                            "bounded_mae",
                            "bounded_mse",
                        ]
                    ),
                    (
                        self.dataset_type == "multiclass"
                        and metric in ["cross_entropy", "accuracy", "f1", "mcc"]
                    ),
                    (
                        self.dataset_type == "spectra"
                        and metric in ["sid", "wasserstein"]
                    ),
                ]
            ):
                raise ValueError(
                    f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".'
                )

        if self.loss_function is None:
            if self.dataset_type == "classification":
                self.loss_function = "binary_cross_entropy"
            elif self.dataset_type == "multiclass":
                self.loss_function = "cross_entropy"
            elif self.dataset_type == "spectra":
                self.loss_function = "sid"
            elif self.dataset_type == "regression":
                self.loss_function = "mse"
            else:
                raise ValueError(
                    f"Default loss function not configured for dataset type {self.dataset_type}."
                )

        if self.loss_function != "bounded_mse" and any(
            metric in ["bounded_mse", "bounded_rmse", "bounded_mae"]
            for metric in self.metrics
        ):
            raise ValueError(
                "Bounded metrics can only be used in conjunction with the regression loss function bounded_mse."
            )

        # Validate class balance
        if self.class_balance and self.dataset_type != "classification":
            raise ValueError(
                "Class balance can only be applied if the dataset type is classification."
            )

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError(
                "When using features_only, a features_generator or features_path must be provided."
            )

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError(
                "Undirected is unnecessary when using atom_messages "
                "since atom_messages are by their nature undirected."
            )

        # Validate split type settings
        if (
            not (self.split_type == "predetermined")
            == (self.folds_file is not None)
            == (self.test_fold_index is not None)
        ):
            raise ValueError(
                "When using predetermined split type, must provide folds_file and test_fold_index."
            )

        if not (self.split_type == "crossval") == (self.crossval_index_dir is not None):
            raise ValueError(
                "When using crossval split type, must provide crossval_index_dir."
            )

        if not (self.split_type in ["crossval", "index_predetermined"]) == (
            self.crossval_index_file is not None
        ):
            raise ValueError(
                "When using crossval or index_predetermined split type, must provide crossval_index_file."
            )

        if self.split_type in ["crossval", "index_predetermined"]:
            with open(self.crossval_index_file, "rb") as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        # Validate split size entry and set default values
        if self.split_sizes is None:
            if (
                self.separate_val_path is None and self.separate_test_path is None
            ):  # separate data paths are not provided
                self.split_sizes = [0.8, 0.1, 0.1]
            elif (
                self.separate_val_path is not None and self.separate_test_path is None
            ):  # separate val path only
                self.split_sizes = [0.8, 0.0, 0.2]
            elif (
                self.separate_val_path is None and self.separate_test_path is not None
            ):  # separate test path only
                self.split_sizes = [0.8, 0.2, 0.0]
            else:  # both separate data paths are provided
                self.split_sizes = [1.0, 0.0, 0.0]

        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(
                    f"Provided split sizes of {self.split_sizes} do not sum to 1."
                )
            if any([size < 0 for size in self.split_sizes]):
                raise ValueError(
                    f"Split sizes must be non-negative. Received split sizes: {self.split_sizes}"
                )

            if len(self.split_sizes) not in [2, 3]:
                raise ValueError(
                    f"Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s)."
                )

            if (
                self.separate_val_path is None and self.separate_test_path is None
            ):  # separate data paths are not provided
                if len(self.split_sizes) != 3:
                    raise ValueError(
                        f"Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s)."
                    )
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        f"Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}"
                    )
                if self.split_sizes[1] == 0.0:
                    raise ValueError(
                        f"Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}"
                    )

            elif (
                self.separate_val_path is not None and self.separate_test_path is None
            ):  # separate val path only
                if len(self.split_sizes) == 2:  # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], 0.0, self.split_sizes[1]]
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        "Provided split size for train split must be nonzero."
                    )
                if self.split_sizes[1] != 0.0:
                    raise ValueError(
                        f"Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}"
                    )

            elif (
                self.separate_val_path is None and self.separate_test_path is not None
            ):  # separate test path only
                if len(self.split_sizes) == 2:  # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.0]
                if self.split_sizes[0] == 0.0:
                    raise ValueError(
                        "Provided split size for train split must be nonzero."
                    )
                if self.split_sizes[1] == 0.0:
                    raise ValueError(
                        "Provided split size for validation split must be nonzero."
                    )
                if self.split_sizes[2] != 0.0:
                    raise ValueError(
                        f"Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}"
                    )

            else:  # both separate data paths are provided
                if self.split_sizes != [1.0, 0.0, 0.0]:
                    raise ValueError(
                        f"Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}"
                    )

        # Test settings
        if self.test:
            self.epochs = 0

        # Validate features are provided for separate validation or test set for each of the kinds of additional features
        for (
            features_argument,
            base_features_path,
            val_features_path,
            test_features_path,
        ) in [
            (
                "`--features_path`",
                self.features_path,
                self.separate_val_features_path,
                self.separate_test_features_path,
            ),
            (
                "`--phase_features_path`",
                self.phase_features_path,
                self.separate_val_phase_features_path,
                self.separate_test_phase_features_path,
            ),
            (
                "`--atom_descriptors_path`",
                self.atom_descriptors_path,
                self.separate_val_atom_descriptors_path,
                self.separate_test_atom_descriptors_path,
            ),
            (
                "`--bond_descriptors_path`",
                self.bond_descriptors_path,
                self.separate_val_bond_descriptors_path,
                self.separate_test_bond_descriptors_path,
            ),
            (
                "`--constraints_path`",
                self.constraints_path,
                self.separate_val_constraints_path,
                self.separate_test_constraints_path,
            ),
        ]:
            if base_features_path is not None:
                if self.separate_val_path is not None and val_features_path is None:
                    raise ValueError(
                        f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate validation set."
                    )
                if self.separate_test_path is not None and test_features_path is None:
                    raise ValueError(
                        f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate test set."
                    )

        # validate extra atom descriptor options
        if self.overwrite_default_atom_features and self.atom_descriptors != "feature":
            raise NotImplementedError(
                "Overwriting of the default atom descriptors can only be used if the"
                "provided atom descriptors are features."
            )

        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError(
                "Atom descriptor scaling is only possible if additional atom features are provided."
            )

        # validate extra bond descriptor options
        if self.overwrite_default_bond_features and self.bond_descriptors != "feature":
            raise NotImplementedError(
                "Overwriting of the default bond descriptors can only be used if the"
                "provided bond descriptors are features."
            )

        if not self.bond_descriptor_scaling and self.bond_descriptors is None:
            raise ValueError(
                "Bond descriptor scaling is only possible if additional bond features are provided."
            )

        if self.bond_descriptors == "descriptor" and not self.is_atom_bond_targets:
            raise NotImplementedError(
                "Bond descriptors as descriptor can only be used with `--is_atom_bond_targets`."
            )

        # normalize target weights
        if self.target_weights is not None:
            avg_weight = sum(self.target_weights) / len(self.target_weights)
            self.target_weights = [w / avg_weight for w in self.target_weights]
            if min(self.target_weights) < 0:
                raise ValueError("Provided target weights must be non-negative.")

        # check if key molecule index is outside of the number of molecules
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError(
                "The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. "
            )


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(
        self,
        args: TrainArgs,
        atom_fdim: int,
        bond_fdim: int,
        hidden_size: int = None,
        bias: bool = None,
        depth: int = None,
    ):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension.
        :param bias: Whether to add bias to linear layers.
        :param depth: Number of message passing steps.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.is_atom_bond_targets = args.is_atom_bond_targets

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False
        )

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.is_atom_bond_targets:
            self.W_o_b = nn.Linear(self.bond_fdim + self.hidden_size, self.hidden_size)

        if args.atom_descriptors == "descriptor":
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(
                self.hidden_size + self.atom_descriptors_size,
                self.hidden_size + self.atom_descriptors_size,
            )

        if args.bond_descriptors == "descriptor":
            self.bond_descriptors_size = args.bond_descriptors_size
            self.bond_descriptors_layer = nn.Linear(
                self.hidden_size + self.bond_descriptors_size,
                self.hidden_size + self.bond_descriptors_size,
            )

    def forward(
        self,
        mol_graph: BatchMolGraph,
        atom_descriptors_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [
                np.zeros([1, atom_descriptors_batch[0].shape[1]])
            ] + atom_descriptors_batch  # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = (
                torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0))
                .float()
                .to(self.device)
            )

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages
        )
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device),
            f_bonds.to(self.device),
            a2b.to(self.device),
            b2a.to(self.device),
            b2revb.to(self.device),
        )

        if self.is_atom_bond_targets:
            b2br = mol_graph.get_b2br().to(self.device)
            if bond_descriptors_batch is not None:
                forward_index = b2br[:, 0]
                backward_index = b2br[:, 1]
                descriptors_batch = np.concatenate(bond_descriptors_batch, axis=0)
                bond_descriptors_batch = np.zeros(
                    [descriptors_batch.shape[0] * 2 + 1, descriptors_batch.shape[1]]
                )
                for i, fi in enumerate(forward_index):
                    bond_descriptors_batch[fi] = descriptors_batch[i]
                for i, fi in enumerate(backward_index):
                    bond_descriptors_batch[fi] = descriptors_batch[i]
                bond_descriptors_batch = (
                    torch.from_numpy(bond_descriptors_batch).float().to(self.device)
                )

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(
                    message, a2a
                )  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(
                    f_bonds, a2b
                )  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat(
                    (nei_a_message, nei_f_bonds), dim=2
                )  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(
                    message, a2b
                )  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x hidden

        # atom hidden
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(
            message, a2x
        )  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat(
            [f_atoms, a_message], dim=1
        )  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden

        # bond hidden
        if self.is_atom_bond_targets:
            b_input = torch.cat(
                [f_bonds, message], dim=1
            )  # num_bonds x (bond_fdim + hidden)
            bond_hiddens = self.act_func(self.W_o_b(b_input))  # num_bonds x hidden
            bond_hiddens = self.dropout(bond_hiddens)  # num_bonds x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(
                    "The number of atoms is different from the length of the extra atom features"
                )

            atom_hiddens = torch.cat(
                [atom_hiddens, atom_descriptors_batch], dim=1
            )  # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(
                atom_hiddens
            )  # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout(
                atom_hiddens
            )  # num_atoms x (hidden + descriptor size)

        # concatenate the bond descriptors
        if self.is_atom_bond_targets and bond_descriptors_batch is not None:
            if len(bond_hiddens) != len(bond_descriptors_batch):
                raise ValueError(
                    "The number of bonds is different from the length of the extra bond features"
                )

            bond_hiddens = torch.cat(
                [bond_hiddens, bond_descriptors_batch], dim=1
            )  # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.bond_descriptors_layer(
                bond_hiddens
            )  # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.dropout(
                bond_hiddens
            )  # num_bonds x (hidden + descriptor size)

        # Readout
        if self.is_atom_bond_targets:
            return (
                atom_hiddens,
                a_scope,
                bond_hiddens,
                b_scope,
                b2br,
            )  # num_atoms x hidden, remove the first one which is zero padding

        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == "sum":
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self, args: TrainArgs, atom_fdim: int = None, bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.atom_fdim = atom_fdim or get_atom_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features,
            is_reaction=(
                self.reaction if self.reaction is not False else self.reaction_solvent
            ),
        )
        self.bond_fdim = bond_fdim or get_bond_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features,
            overwrite_default_bond=args.overwrite_default_bond_features,
            atom_messages=args.atom_messages,
            is_reaction=(
                self.reaction if self.reaction is not False else self.reaction_solvent
            ),
        )
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.bond_descriptors = args.bond_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if not self.reaction_solvent:
            if args.mpn_shared:
                self.encoder = nn.ModuleList(
                    [MPNEncoder(args, self.atom_fdim, self.bond_fdim)]
                    * args.number_of_molecules
                )
            else:
                self.encoder = nn.ModuleList(
                    [
                        MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                        for _ in range(args.number_of_molecules)
                    ]
                )
        else:
            self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            # Set separate atom_fdim and bond_fdim for solvent molecules
            self.atom_fdim_solvent = get_atom_fdim(
                overwrite_default_atom=args.overwrite_default_atom_features,
                is_reaction=False,
            )
            self.bond_fdim_solvent = get_bond_fdim(
                overwrite_default_atom=args.overwrite_default_atom_features,
                overwrite_default_bond=args.overwrite_default_bond_features,
                atom_messages=args.atom_messages,
                is_reaction=False,
            )
            self.encoder_solvent = MPNEncoder(
                args,
                self.atom_fdim_solvent,
                self.bond_fdim_solvent,
                args.hidden_size_solvent,
                args.bias_solvent,
                args.depth_solvent,
            )

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if not isinstance(batch[0], BatchMolGraph):
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == "feature":
                if len(batch) > 1:
                    raise NotImplementedError(
                        "Atom/bond descriptors are currently only supported with one molecule "
                        "per input (i.e., number_of_molecules = 1)."
                    )

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features,
                    )
                    for b in batch
                ]
            elif self.bond_descriptors == "feature":
                if len(batch) > 1:
                    raise NotImplementedError(
                        "Atom/bond descriptors are currently only supported with one molecule "
                        "per input (i.e., number_of_molecules = 1)."
                    )

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features,
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = (
                torch.from_numpy(np.stack(features_batch)).float().to(self.device)
            )

            if self.features_only:
                return features_batch

        if (
            self.atom_descriptors == "descriptor"
            or self.bond_descriptors == "descriptor"
        ):
            if len(batch) > 1:
                raise NotImplementedError(
                    "Atom descriptors are currently only supported with one molecule "
                    "per input (i.e., number_of_molecules = 1)."
                )

            encodings = [
                enc(ba, atom_descriptors_batch, bond_descriptors_batch)
                for enc, ba in zip(self.encoder, batch)
            ]
        else:
            if not self.reaction_solvent:
                encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
            else:
                encodings = []
                for ba in batch:
                    if ba.is_reaction:
                        encodings.append(self.encoder(ba))
                    else:
                        encodings.append(self.encoder_solvent(ba))

        output = encodings[0] if len(encodings) == 1 else torch.cat(encodings, dim=1)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function

        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]

        self.is_atom_bond_targets = args.is_atom_bond_targets

        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets = args.atom_targets, args.bond_targets
            self.atom_constraints, self.bond_constraints = (
                args.atom_constraints,
                args.bond_constraints,
            )
            self.adding_bond_types = args.adding_bond_types

        self.relative_output_size = 1
        if self.multiclass:
            self.relative_output_size *= args.multiclass_num_classes
        if self.loss_function == "mve":
            self.relative_output_size *= 2  # return means and variances
        if self.loss_function == "dirichlet" and self.classification:
            self.relative_output_size *= (
                2  # return dirichlet parameters for positive and negative class
            )
        if self.loss_function == "evidential":
            self.relative_output_size *= (
                4  # return four evidential parameters: gamma, lambda, alpha, beta
            )

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = nn.Softplus()

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == "descriptor":
            atom_first_linear_dim = first_linear_dim + args.atom_descriptors_size
        else:
            atom_first_linear_dim = first_linear_dim

        if args.bond_descriptors == "descriptor":
            bond_first_linear_dim = first_linear_dim + args.bond_descriptors_size
        else:
            bond_first_linear_dim = first_linear_dim

        # Create FFN layers
        if self.is_atom_bond_targets:
            self.readout = MultiReadout(
                atom_features_size=atom_first_linear_dim,
                bond_features_size=bond_first_linear_dim,
                atom_hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                bond_hidden_size=args.ffn_hidden_size + args.bond_descriptors_size,
                num_layers=args.ffn_num_layers,
                output_size=self.relative_output_size,
                dropout=args.dropout,
                activation=args.activation,
                atom_constraints=args.atom_constraints,
                bond_constraints=args.bond_constraints,
                shared_ffn=args.shared_atom_bond_ffn,
                weights_ffn_num_layers=args.weights_ffn_num_layers,
            )
        else:
            self.readout = build_ffn(
                first_linear_dim=atom_first_linear_dim,
                hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                num_layers=args.ffn_num_layers,
                output_size=self.relative_output_size * args.num_tasks,
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                if self.is_atom_bond_targets:
                    if args.shared_atom_bond_ffn:
                        for param in list(self.readout.atom_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                        for param in list(self.readout.bond_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                    else:
                        for ffn in self.readout.ffn_list:
                            if ffn.constraint:
                                for param in list(ffn.ffn.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                            else:
                                for param in list(ffn.ffn_readout.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                else:
                    for param in list(self.readout.parameters())[
                        0 : 2 * args.frzn_ffn_layers
                    ]:  # Freeze weights and bias for given number of layers
                        param.requires_grad = False

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type: str = "MPN",
    ) -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.readout[:-1](
                self.encoder(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """
        if self.is_atom_bond_targets:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings, constraints_batch, bond_types_batch)
        else:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings)

        # Don't apply sigmoid during training when using BCEWithLogitsLoss
        if (
            self.classification
            and not (self.training and self.no_training_normalization)
            and self.loss_function != "dirichlet"
        ):
            if self.is_atom_bond_targets:
                output = [self.sigmoid(x) for x in output]
            else:
                output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.shape[0], -1, self.num_classes)
            )  # batch size x num targets x num classes per target
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(
                    output
                )  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Modify multi-input loss functions
        if self.loss_function == "mve":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, variances = torch.split(x, x.shape[1] // 2, dim=1)
                    variances = self.softplus(variances)
                    outputs.append(torch.cat([means, variances], axis=1))
                return outputs
            else:
                means, variances = torch.split(output, output.shape[1] // 2, dim=1)
                variances = self.softplus(variances)
                output = torch.cat([means, variances], axis=1)
        if self.loss_function == "evidential":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, lambdas, alphas, betas = torch.split(
                        x, x.shape[1] // 4, dim=1
                    )
                    lambdas = self.softplus(lambdas)  # + min_val
                    alphas = (
                        self.softplus(alphas) + 1
                    )  # + min_val # add 1 for numerical contraints of Gamma function
                    betas = self.softplus(betas)  # + min_val
                    outputs.append(torch.cat([means, lambdas, alphas, betas], dim=1))
                return outputs
            else:
                means, lambdas, alphas, betas = torch.split(
                    output, output.shape[1] // 4, dim=1
                )
                lambdas = self.softplus(lambdas)  # + min_val
                alphas = (
                    self.softplus(alphas) + 1
                )  # + min_val # add 1 for numerical contraints of Gamma function
                betas = self.softplus(betas)  # + min_val
                output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == "dirichlet":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    outputs.append(nn.functional.softplus(x) + 1)
                return outputs
            else:
                output = nn.functional.softplus(output) + 1

        return output
