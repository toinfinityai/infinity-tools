import random
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from typing import Dict, List, Tuple


class BaseGenerator(tf.keras.utils.Sequence):
    """Defines a Keras generator for ingesting time series data.

    As a subclass of the Keras Sequence class, this generator
    natively supports many of the core Keras APIs such as
    being passed to model.fit().

    The generator is specifically designed to output time series
    in a STATEFUL manner, by breaking up a set of variable
    length time series (the traing data) into discrete windows
    of the same size, and outputting temporally contiguous windows
    in the correct order. The generator implements the notion of a
    "batch cycle" - described below - in order to seamlessly scale
    this process to an arbitrarily large number of input sequences.
    Note that although the generator outputs discrete windows that
    are temporally contiguous, non-stateful training pipelines can
    still be used. This would just mean that each discrete window
    is treated independently of the one before it.

    A single "batch cycle" of the generator represents a finite set
    of mini-batches in which the generator iterates over temporally
    contiguous windows of N different, randomly selected time series,
    where N=batch_size.

    For example, if 64 time series are given to the generator (i.e.
    len(sequences_X) = 64), and batch_size = 32, there would be two
    batch cycles. The first batch cycle would randomly select 32
    time series, and iterate over temporally contigous windows of
    length window_len. The second batch cycle would then iterate over
    the windows of the other 32 time series. In this example, the
    two batch cycles corresponds to one training epoch (since two
    batch cycles are required to iterate over all the data).

    In order standardize the length of each batch cycle, we take the
    minimum lenght of all input sequences (see `fixed_seq_len`).

    TODO: support labels other than binary classification.

    Attributes:
        sequences_X (List[npt.NDArray]): List of featurized, variable-
            length time series data. Each list element is a numpy array
            of shape (T_i x F), where F is the feature dimension.
        sequences_y (List[npt.NDArray]): List of label arrays
            corresponding to featurized data. Each list element is a numpy
            array of shape (T_i x label_dim).
        window_len (int): Number of time points defining the size of the
            window that will be output by the generator.
        batch_size (int): Number of time series sampled from sequences_X that
            make up a single batch cycle (see notes above).
        class_weights (Dict[float, float]): Dictionary mapping each
            class to a sample weight that can by used by a loss function
            to account for imbalanced classes. Classes should be whole
            numbers but are represented as float32 for TF compatibility.
        num_seqs (int): Total number of input sequences.
        num_batch_cycles (int): Total number of batch cycles in a single epoch.
        fixed_seq_len (int): The total number of (contiguous) time points
            that  will be seen for a given time series in one epoch.
            TODO: This is currently calculated as the minimum length across
            all input sequences. Ideally, the generator would be more
            flexible so that passing in a single short time sequence does
            not radically change its behavior.
        num_windows (int): Number of windows generated in a single batch cycle.
        batch_cycle_X (npt.NDArray): Data for current batch cycle that will be
            fed to a model during training. Represented as a numpy array of
            shape (batch_size x fixed_seq_len x feature_dim)
        batch_cycle_y (npt.NDArray): Labels for current batch cycle, to be used
            in loss function. Represented as a numpy array of shape
            (batch_size x fixed_seq_len x label_dim).
        batch_cycle_sample_weight (npt.NDArray): Class weights for current batch
            cycle, used to apply a different weight to every timestep of
            every sample. Represented as a numpy array of shape
            (batch_size x fixed_seq_len x 1)
        batch_cycle_indices (List[List[int]]): Defines which elements of
            sequences_X get assigned to each batch cycle for the current epoch.
    """

    def __init__(
        self,
        sequences_X: List[npt.NDArray],
        sequences_y: List[npt.NDArray],
        window_len: int,
        batch_size: int,
        class_weights: Dict[int, float],
    ):
        """Initializes TrainGenerator object."""

        self.sequences_X = sequences_X
        self.sequences_y = sequences_y
        self.batch_size = batch_size
        self.window_len = window_len
        self.class_weights = class_weights

        self.num_seqs = len(self.sequences_X)
        #  note: using ceil() here is like setting drop_remainder to False
        #  in the tf.data.Dataset API
        self.num_batch_cycles = int(np.ceil(self.num_seqs / self.batch_size))

        seq_lens = [len(seq) for seq in sequences_X]
        self.fixed_seq_len = min(seq_lens) - (min(seq_lens) % window_len)  # round down to be divisible
        self.num_windows = int(self.fixed_seq_len / self.window_len)
        self.prepare_epoch()

    def prepare_epoch(self):
        """Initializes generator state for new epoch.

        This includes assigning a new random set sequences to each batch cycle,
        so that each epoch is different.
        """

        def split_list(lst: List, sz: int) -> List[List]:
            """Splits input into list of lists of specified chunk size.

            See https://stackoverflow.com/a/4119142
            """
            return [lst[i : i + sz] for i in range(0, len(lst), sz)]

        seq_order = list(range(self.num_seqs))
        random.shuffle(seq_order)
        self.batch_cycle_indices = split_list(seq_order, self.batch_size)

    def prepare_batch_cycle(self, batch_cycle: int):
        """Initializes generator state for new batch cycle.

        This includes extracting a random window from each time series
        (assuming they are variable length).

        Args:
            batch_cycle: Index of the current batch cycle within
                the current epoch.
        """

        sequences_X = [self.sequences_X[i] for i in self.batch_cycle_indices[batch_cycle]]
        sequences_y = [self.sequences_y[i] for i in self.batch_cycle_indices[batch_cycle]]

        batch_cycle_X = []
        batch_cycle_y = []
        for seq_X, seq_y in zip(sequences_X, sequences_y):
            max_offset = len(seq_X) - self.fixed_seq_len
            offset = np.random.randint(0, max_offset)
            batch_cycle_X.append(seq_X[offset : (offset + self.fixed_seq_len)])
            batch_cycle_y.append(seq_y[offset : (offset + self.fixed_seq_len)])
        self.batch_cycle_X = np.array(batch_cycle_X).astype(float)
        self.batch_cycle_y = np.array(batch_cycle_y)
        self.batch_cycle_sample_weight = np.where(self.batch_cycle_y, self.class_weights[1], self.class_weights[0])

    def __len__(self) -> int:
        """Returns number of mini-batches generated in one epoch."""
        return int(self.num_windows * self.num_batch_cycles)

    def __getitem__(self, idx) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Returns (x,y,sample_weight) tuple for current mini-batch.

        Args:
            idx: Index of mini-batch to retrieve from current epoch.
        """

        if idx % self.num_windows == 0:
            # swap in new set of sequences
            batch_cycle = idx // self.num_windows
            self.prepare_batch_cycle(batch_cycle)

        window_idx = idx % self.num_windows
        start = window_idx * self.window_len
        stop = (window_idx + 1) * self.window_len
        window_indices = np.arange(start, stop)
        minibatch_X = self.batch_cycle_X[:, window_indices, :]
        minibatch_y = self.batch_cycle_y[:, window_indices, :]
        minibatch_weights = self.batch_cycle_sample_weight[:, window_indices, :]
        return minibatch_X, minibatch_y, minibatch_weights

    def on_epoch_end(self):
        """Handles generator state logic at the end of an epoch."""
        self.prepare_epoch()

    @staticmethod
    def featurize_repcount(y_input: List[npt.NDArray], binarize_threshold: float) -> List[npt.NDArray]:
        """Converts rep count to binary label Based on proximity to rep completion.

        Args:
            y_input: List of rep count arrays that should be binarized into 0/1 labels.
            binarize_threshold: How close to rep completion the rep_count value should
                be to be assigned a positive label. Represented as value
                between (0,1).

        Returns:
            List of binarized rep count array, so that time points close to a
            rep inflection are assigned a value of 1.
        """
        y_output = []
        for y in y_input:
            close_to_completion = np.abs(y % 1 - 0.5) > (0.5 - binarize_threshold)
            binary_label = (close_to_completion).astype(float)
            binary_label = np.expand_dims(binary_label, axis=1)
            y_output.append(binary_label)
        return y_output

    @staticmethod
    def featurize_data(x_input: List[npt.NDArray]):
        raise NotImplementedError()

    @staticmethod
    def load_data(data_path: str):
        raise NotImplementedError()

    def load_data_from_folders(folders: List[str]):
        raise NotImplementedError()
