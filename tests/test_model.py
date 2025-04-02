import random
import unittest

import numpy as np
from rdkit import RDLogger

from admet_ai.admet_model import ADMETModel


class TestModels(unittest.TestCase):
    """
    Test the model.
    """

    def setUp(self):
        random.seed(37)
        np.random.seed(37)
        RDLogger.DisableLog("rdApp.*")

    def test_get_model(self):
        """
        Test compute input surface.
        """

        model = ADMETModel()
        