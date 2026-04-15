from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


###########################################################
#            Data Paths and HASH Config                   #
###########################################################
BASE_DIR   = Path("Data")
TRAIN_DIR  = BASE_DIR / "Training"
TEST_DIR   = BASE_DIR / "Test"


HASH_THRESHOLD = 4
HASH_CACHE_DIR = Path("hash_cache")


###########################################################
#            Image and model config                       #
###########################################################

IMG_SIZE   = (254, 254)   # we can downsample from 254x254 for CPU efficiency
BATCH_SIZE = 32
EPOCHS=100
VAL_FRACTION=0.15
SEED = 42


###########################################################
#            Model Metrics and Callbacks Config           #
###########################################################


######## Binary Classification (Fire vs No Fire) ########
binary_metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]

binary_loss= tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)

binary_callbacks_v1 = [
    EarlyStopping(monitor="val_auc", patience=8, mode="max",
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_cnn_binary_v1.keras", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=0),]

binary_callbacks_v2 = [
    EarlyStopping(monitor="val_auc", patience=8, mode="max",
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_cnn_binary_v2.keras", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=0),]


######## Ternary Classification (Fire vs No Fire vs Lake) ########
ternary_loss    = "sparse_categorical_crossentropy"


ternary_metrics = ["accuracy",
                   keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")]

ternary_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8, mode="max",
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_cnn_ternary.keras", monitor="val_accuracy",
                    mode="max", save_best_only=True, verbose=0),
]
