import os
import glob
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Dict, List, Tuple
from infinity_tools.common.ml.datagen import BaseGenerator


def matrix_to_6DOF(data: npt.NDArray) -> npt.NDArray:
    """Converts (Tx3x3) array of rotation matrices to (Tx6) cont. representation."""
    return data[:, :, :2].reshape(-1, 6)


class SenseFitGenerator(BaseGenerator):
    def __init__(
        self,
        sequences_X: List[npt.NDArray],
        sequences_y: List[npt.NDArray],
        window_len: int,
        batch_size: int,
        class_weights: Dict[int, float],
    ):
        super().__init__(sequences_X, sequences_y, window_len, batch_size, class_weights)

    @staticmethod
    def featurize_data(sequences_X: List[npt.NDArray]) -> List[npt.NDArray]:
        """Featurizes data for model training/inference."""
        X_output = []
        for X_input in sequences_X:
            X_output.append(matrix_to_6DOF(X_input))
        return X_output

    @staticmethod
    def load_data(data_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
        """Loads X,y data from a single example."""
        rot_mat_columns = [
            "rotation_matrix_m11",
            "rotation_matrix_m12",
            "rotation_matrix_m13",
            "rotation_matrix_m21",
            "rotation_matrix_m22",
            "rotation_matrix_m23",
            "rotation_matrix_m31",
            "rotation_matrix_m32",
            "rotation_matrix_m33",
        ]

        df = pd.read_csv(data_path)
        sequence_X = df[rot_mat_columns].values.reshape(-1, 3, 3)
        if "rep_count_from_intermediate" in df.columns:
            sequence_y = df["rep_count_from_intermediate"].values
        else:
            # rep_count_* label not provided for negative (non-rep) data
            sequence_y = 0.5 * np.ones(len(df))
        return sequence_X, sequence_y

    @staticmethod
    def load_data_from_folders(
        folders: List[str],
    ) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
        """Loads all SenseFit data from a list of folders."""

        csv_paths = []
        for folder in folders:
            csv_paths.extend(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))

        sequences_y = []
        sequences_X = []
        for csv_path in csv_paths:
            sequence_X, sequence_y = SenseFitGenerator.load_data(csv_path)
            sequences_X.append(sequence_X)
            sequences_y.append(sequence_y)

        return sequences_X, sequences_y
