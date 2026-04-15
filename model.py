import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report

import numpy as np


import config as ml_config
from utils import IMG_SIZE
import utils as ml_utils

def micro_block(x, filters, dropout_rate=0.2):
    # Single depthwise separable conv — no double conv
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 1, use_bias=False)(x)  # pointwise
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling2D(2)(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def binary_micro_cnn(input_shape=(*IMG_SIZE, 3)):
    inp = keras.Input(shape=input_shape)

    # Stem: aggressive early downsampling
    x = layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(inp)  # →32
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Micro blocks with increasing filters and dropout
    x = micro_block(x, 16,  dropout_rate=0.10)   # 32→16
    x = micro_block(x, 32,  dropout_rate=0.15)   # 16→8
    x = micro_block(x, 64,  dropout_rate=0.20)   # 8→4
    x = micro_block(x, 96,  dropout_rate=0.25)   # 4→2

    # Head: global pooling + dropout, no dense layers
    x = layers.GlobalAveragePooling2D()(x)        # 96 values
    x = layers.Dropout(0.35)(x)

    # output layer with sigmoid activation for binary classification
    out  = layers.Dense(1, activation="sigmoid")(x)

    # create and compile model
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss= ml_config.binary_loss,
        metrics=ml_config.binary_metrics
    )
    return model

def ternary_micro_cnn(input_shape=(*IMG_SIZE, 3)):
    inp = keras.Input(shape=input_shape)

    # Stem: aggressive early downsampling
    x = layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Micro blocks with increasing filters and dropout
    x = micro_block(x, 16, dropout_rate=0.10)   # 32→16
    x = micro_block(x, 32, dropout_rate=0.15)   # 16→8
    x = micro_block(x, 64, dropout_rate=0.20)   # 8→4
    x = micro_block(x, 96, dropout_rate=0.25)   # 4→2

    # Head: global pooling + dropout, no dense layers

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)

    # output layer with softmax activation for ternary classification
    out = layers.Dense(3, activation="softmax")(x)


    # create and compile model
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=ml_config.ternary_loss,
        metrics=ml_config.ternary_metrics
    )
    return model


if __name__ == "__main__":

    train_nofire_files = list((ml_config.TRAIN_DIR / "No_Fire").glob("*.jpg"))
    n_lake_train = sum(1 for f in train_nofire_files if f.name.startswith("lake"))
    n_nofire_train = len(train_nofire_files) - n_lake_train

    test_nofire_files  = list((ml_config.TEST_DIR / "No_Fire").glob("*.jpg"))
    n_lake_test  = sum(1 for f in test_nofire_files if f.name.startswith("lake"))
    n_nofire_test = len(test_nofire_files) - n_lake_test


    fire_paths  = list((ml_config.TRAIN_DIR / "Fire").glob("*.jpg"))
    nofire_paths = [f for f in train_nofire_files if not f.name.startswith("lake")]
    lake_paths   = [f for f in train_nofire_files if f.name.startswith("lake")]

    print("Deduplicating Fire frames …")
    fire_sorted   = sorted(fire_paths,   key=ml_config.frame_number)
    fire_deduped  = ml_config.deduplicate(fire_sorted)

    print("Deduplicating No Fire (non-lake) frames …")
    nofire_sorted  = sorted(nofire_paths,  key=ml_config.frame_number)
    nofire_deduped = ml_config.deduplicate(nofire_sorted)

    print("Deduplicating Lake frames …")
    lake_sorted   = sorted(lake_paths,   key=ml_config.frame_number)
    lake_deduped  = ml_config.deduplicate(lake_sorted)



    model=  int(input("What model (binary=0, ternary=1): "))

    if model == 0:

        bin_train_ds, bin_val_ds, bin_test_ds, bin_train_labels, bin_val_labels,bin_test_labels = (
         ml_utils.build_binary_datasets(fire_deduped, nofire_deduped, lake_deduped, lake_as_nofire=True))

        binary_model = binary_micro_cnn()
        print("Binary Micro CNN Summary:")
        binary_model.summary()

        n_fire   = bin_train_labels.count(1)
        n_nofire = bin_train_labels.count(0)
        total    = n_fire + n_nofire
        binary_class_weights = {0: total / (2 * n_nofire),
                                1: total / (2 * n_fire)}

        binary_model.fit(
            bin_train_ds,
            validation_data=bin_val_ds,
            epochs=ml_config.EPOCHS,
            class_weight=binary_class_weights,
            callbacks=ml_config.binary_callbacks_v1
        )

        ml_utils.plot_history(binary_model,    "Binary Training History")

        results_binary = []


        results_binary.append(ml_utils.evaluate_binary(binary_model,      bin_test_ds,
                                            bin_test_labels, "Binary V1 Confusion Matrix (test)"))

        ml_utils.plot_confusion_matrix(results_binary[0][0],results_binary[0][1],['No_Fire','Fire'], title="Binary V2 Confusion Matrix (test)")



    elif model == 1:
        tern_train_ds, tern_val_ds, tern_test_ds, tern_train_labels, tern_val_labels, tern_test_labels = (
    ml_utils.build_ternary_datasets(fire_deduped, nofire_deduped, lake_deduped)
)

        n_fire   = tern_train_labels.count(0)
        n_nofire = tern_train_labels.count(1)
        n_lake   = tern_train_labels.count(2)
        total    = n_fire + n_nofire + n_lake

        ternary_class_weights = {
            0: total / (3 * n_fire),
            1: total / (3 * n_nofire),
            2: total / (3 * n_lake),
}

        ternary_model = ternary_micro_cnn()
        print("Ternary Micro CNN Summary:")
        ternary_model.summary()

        history_micro_cnn_ternary = ternary_model.fit(
                tern_train_ds,
                validation_data=tern_val_ds,
                epochs=100,
                class_weight=ternary_class_weights,
                callbacks=ml_config.ternary_callbacks,
                verbose=1,
)
        ml_utils.plot_history(history_micro_cnn_ternary, "")

        # test set performance
        test_proba     = ml_utils.micro_cnn_ternary.predict(tern_test_ds)
        test_preds_arr = test_proba.argmax(axis=1)
        test_true_arr  = np.array(tern_test_labels)

        #   Plot confusion matrix for test set
        ml_utils.plot_confusion_matrix(
            test_true_arr, test_preds_arr,
            class_names=["Fire", "No Fire", "Lake"],
            title="Ternary Confusion Matrix (Test)"
        )

        # print classification report with 3 decimal places for precision, recall, and F1
        print(classification_report(
            test_true_arr, test_preds_arr,
            target_names=["Fire", "No Fire", "Lake"], digits=3
        ))
