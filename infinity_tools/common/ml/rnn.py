import os
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy.typing as npt
from collections import Counter
from typing import Dict, List, Optional, Tuple
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback


def shuffle_lists(a: List, b: List) -> Tuple[List, List]:
    """Randomly shuffles two lists with the same re-ordering."""
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def get_class_weights(sequences_y: List[npt.NDArray]) -> Dict[float, float]:
    """Computes weights for each class.

    See https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    Note: current implementation is specific to binary classification.

    Args:
        sequence_y: List of label arrays corresponding to training data.
            Each list element is a numpy array of classes.

    Returns:
        Dictionary mapping each class to a sample weight that can by used by a
        loss function to account for imbalanced classes.
    """
    all_y = [y for sequence_y in sequences_y for y in sequence_y.flatten()]
    counter = Counter(all_y)
    total = sum(counter.values())
    class_weight = {k: (1 / v) * (total / 2.0) for k, v in counter.items()}
    return class_weight


def build_model(
    norm_layer: tf.keras.layers.experimental.preprocessing.Normalization,
    num_feats: int,
    num_units: int,
    feature_dropout: float = 0,
    temporal_dropout: float = 0,
):
    """Returns a compiled two-layer LSTM.

    Args:
        norm_layer: Pre-adapted normalization layer used to make data be
            zero-mean and unit-variance.
        num_feats: Size of feature dimension.
        num_units: Number of units in each LSTM layer.
        feature_dropout: Probability of feature dropout during training.
        temporal_dropout: Probability of time sample dropout during training.

    Returns:
        Compiled Keras model.
    """

    tf.keras.backend.clear_session()
    inputs = layers.Input(shape=(None, num_feats))
    x = norm_layer(inputs)
    x = layers.GaussianNoise(0.2)(x)
    x = layers.Dropout(rate=feature_dropout, noise_shape=(None, 1, num_feats))(x)
    x = layers.Dropout(rate=temporal_dropout, noise_shape=(None, None, 1))(x)
    x = layers.Bidirectional(layers.LSTM(num_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(num_units, return_sequences=True))(x)
    x = layers.Dense(num_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])

    metrics = [
        tf.keras.metrics.Precision(name="precision"),
    ]

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=metrics,
    )

    return model


def convert_probability_to_events(
    y_pred: npt.NDArray, lower_threshold: float, upper_threshold: float
) -> List[int]:
    """Extracts rep indices from single RNN output.

    Args:
        y_pred: Sequence of raw probabilities generated by RNN.
        lower_threshold: Probability threshold, below which the decoder
            enters a ready state.
        upper_threshold: Probability threshold, above which a rep is
            considered to have occured.

    Returns:
        List of indices corresponding to new reps.
    """

    rep_indices = []
    ready = False
    for i, y_i in enumerate(y_pred):
        if y_i < lower_threshold:
            ready = True
        elif ready and y_i > upper_threshold:
            rep_indices.append(i)
            ready = False
    return rep_indices


def convert_events_to_count(rep_indices: List[int], num_frames: int) -> List[int]:
    """Calculates the rep count for each frame based on predicted rep events.

    Args:
        rep_indices: List of frame indices corresponding to predicted rep events.
        num_frames: Total number of frames in the predicted sequence.

    Returns:
        The predicted rep count for each frame as a list of integers.
    """
    pred_count = []
    pointer_idx = 0
    current_rep_count = 0
    for i in range(num_frames):
        if i == rep_indices[pointer_idx]:
            current_rep_count += 1
            pointer_idx += 1
        pred_count.append(current_rep_count)
        if pointer_idx == len(rep_indices):
            pred_count.extend([current_rep_count] * (num_frames - len(pred_count)))
            return pred_count
    return pred_count


class BaseModel:
    def __init__(self):
        pass

    def fit(
        self,
        data_folders: List[str],
        checkpoint_dir: str,
        batch_size: int = 32,
        window_len: int = 60,
        train_ratio: float = 0.9,
        binarize_threshold: float = 0.15,
        num_units: int = 32,
        epochs: int = 300,
        es_patience: int = 50,
        feature_dropout: float = 0.1,
        temporal_dropout: float = 0.02,
    ) -> tf.keras.callbacks.History:
        """Trains RNN rep counting model on synthetic data generated by Infinity API.

        Args:
            data_folders: List of folders containing syntheting training data.
            checkpoint_dir: Folder where model checkpoint will be saved.

        Returns:
            Keras History object containing training information.
        """

        sequences_X, sequences_y = self.generator.load_data_from_folders(data_folders)
        sequences_X = self.generator.featurize_data(sequences_X)
        sequences_y = self.generator.featurize_repcount(sequences_y, binarize_threshold)
        sequences_X, sequences_y = shuffle_lists(sequences_X, sequences_y)

        # Make train/val/test sets
        num_total = len(sequences_X)
        num_train = int(train_ratio * num_total)
        num_val = num_total - num_train

        sequences_X_train = sequences_X[:num_train]
        sequences_y_train = sequences_y[:num_train]

        sequences_X_val = sequences_X[num_train : (num_train + num_val)]
        sequences_y_val = sequences_y[num_train : (num_train + num_val)]

        # Make generators
        class_weights = get_class_weights(sequences_y_train)

        train_gen = self.generator(
            sequences_X=sequences_X_train,
            sequences_y=sequences_y_train,
            window_len=window_len,
            class_weights=class_weights,
            batch_size=batch_size,
        )

        val_gen = self.generator(
            sequences_X=sequences_X_val,
            sequences_y=sequences_y_val,
            window_len=window_len,
            class_weights=class_weights,
            batch_size=batch_size,
        )

        # Compile model
        norm_layer = tf.keras.layers.Normalization()
        norm_layer.adapt(np.vstack([batch[0] for batch in train_gen]))
        num_feats = train_gen.__getitem__(0)[0].shape[-1]

        self.model = build_model(
            norm_layer=norm_layer,
            num_feats=num_feats,
            num_units=num_units,
            feature_dropout=feature_dropout,
            temporal_dropout=temporal_dropout,
        )

        # Train model
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        monitor = "val_loss"
        mode = "auto"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=es_patience,
                restore_best_weights=True,
                mode=mode,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_dir,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                mode=mode,
            ),
            TqdmCallback(verbose=0),
        ]

        history = self.model.fit(
            x=train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        history_df = pd.DataFrame(history.history)
        fig, ax = plt.subplots(dpi=150)
        ax.plot(history_df)
        ax.legend(history_df.columns)
        ax.set_xlabel("Epoch")
        plt.show()

        return history

    def predict(
        self,
        data_path: str,
        lower_threshold: float = 0.2,
        upper_threshold: float = 0.7,
        output_tag: Optional[str] = None,
    ):
        """Performs model inference on specified data path."""

        test_sequences_X, _ = self.generator.load_data(data_path)
        test_sequences_X = self.generator.featurize_data([test_sequences_X])

        self.model.reset_states()
        y_pred = self.model.predict(test_sequences_X[0][None])[0]
        rep_indices = convert_probability_to_events(
            y_pred, lower_threshold, upper_threshold
        )
        pred_count = convert_events_to_count(rep_indices, num_frames=len(y_pred))
        self.display_predictions(data_path, pred_count, output_tag=output_tag)

    def display_predictions(self):
        raise NotImplementedError
