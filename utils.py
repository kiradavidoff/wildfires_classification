
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import imagehash
import seaborn as sns
import matplotlib.pyplot as plt

# Importing my configuration module with paths, hyperparameters, and callback configurations

from config import ( TEST_DIR,
                     HASH_THRESHOLD, HASH_CACHE_DIR,
                     IMG_SIZE, BATCH_SIZE, VAL_FRACTION, SEED)

import tensorflow as tf

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)

# Reproducibility

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# Create hash cache directory if it doesn't exist
HASH_CACHE_DIR.mkdir(exist_ok=True)

###########################################################
#            Data loading & preprocessing utilities.      #
###########################################################


def count_images(directory):
    """Count JPEG images per class sub-directory.

    Iterates over the immediate sub-directories of ``directory`` (sorted
    alphabetically) and counts how many ``*.jpg`` files each one contains.
    Useful for a quick data-inventory check before training.

    Parameters
    ----------
    directory : Path
        Root directory whose immediate children are class folders
        (e.g. ``Data/Training``).

    Returns
    -------
    dict[str, int]
        Mapping of ``{class_folder_name: image_count}`` for every
        sub-directory found.

    Examples
    --------
    >>> count_images(TRAIN_DIR)
    {'Fire': 3974, 'Lake': 812, 'No_Fire': 5200}
    """
    counts = {}
    for cls_dir in sorted(directory.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name] = len(list(cls_dir.glob("*.jpg")))
    return counts



def compute_hashes(paths, cache_file):
    """Compute perceptual hashes for a list of image paths, with CSV caching.

    On the first call the function iterates over every path, computes a
    64-bit perceptual hash (pHash) via :func:`imagehash.phash`, and
    persists the results to ``cache_file`` as a two-column CSV
    (``path``, ``hash``).  On subsequent calls the CSV is loaded directly,
    making repeated runs fast.

    Parameters
    ----------
    paths : list[str | Path]
        Image file paths to hash.
    cache_file : str | Path
        Destination for the CSV cache.  If the file already exists the
        hashing step is skipped entirely.

    Returns
    -------
    dict[str, str | None]
        Mapping of ``{str(path): hex_hash_string}``.  Paths that could
        not be opened are stored with a ``None`` value.

    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        print(f"  Loading hashes from cache: {cache_file}")
        data = pd.read_csv(cache_file)
        return dict(zip(data["path"], data["hash"]))

    hashes = {}
    for i, p in enumerate(paths):
        try:
            h = str(imagehash.phash(Image.open(p)))
        except Exception:
            h = None
        hashes[str(p)] = h
        if (i + 1) % 5000 == 0:
            print(f"  Hashed {i+1}/{len(paths)}")

    pd.DataFrame({"path": list(hashes.keys()),
                  "hash": list(hashes.values())}).to_csv(cache_file, index=False)
    return hashes


def deduplicate(paths, threshold=HASH_THRESHOLD):
    """Remove near-duplicate images using perceptual hashing.


    Parameters
    ----------
    paths : list[str | Path]
        Candidate image paths.  Order matters: the first image in a
        near-duplicate cluster is always kept.
    threshold : int, optional
        Maximum Hamming distance (inclusive) at which two hashes are
        treated as duplicates.  Defaults to ``HASH_THRESHOLD`` (4).

    Returns
    -------
    list[str | Path]
        Deduplicated subset of ``paths`` preserving the original order.

    Notes
    -----
    Images that cannot be opened are skipped without raising an exception.
    """
    kept = []
    seen_hashes = []   # list of imagehash objects

    for p in paths:
        try:
            h = imagehash.phash(Image.open(p))
        except Exception:
            continue

        is_dup = any(abs(h - s) <= threshold for s in seen_hashes)
        if not is_dup:
            kept.append(p)
            seen_hashes.append(h)
    return kept

def temporal_split(paths, val_fraction=VAL_FRACTION):
    """Split image paths into train and validation sets preserving temporal order.

    Parameters
    ----------
    paths : list[str | Path]
        Image paths.  Frame numbers are extracted via :func:`frame_number`.
    val_fraction : float, optional
        Approximate fraction of images to place in validation.  Converted
        to a block size as ``round(1 / val_fraction)``.  Defaults to
        ``VAL_FRACTION`` (0.15 ≈ every 7th image).

    Returns
    -------
    train : list[str | Path]
        Training-set paths (~85 % of input).
    val : list[str | Path]
        Validation-set paths (~15 % of input).
    """
    paths = sorted(paths, key=frame_number)
    n = len(paths)
    block_size = max(1, int(round(1 / val_fraction)))
    val, train = [], []
    for i, p in enumerate(paths):
        if i % block_size == (block_size - 1):
            val.append(p)
        else:
            train.append(p)
    return train, val


def frame_number(p):
    """Extract a numeric frame index from an image filename.


    Parameters
    ----------
    p : str or Path
        Path to an image file whose stem contains a frame number, e.g.
        ``"clip01_frame0042.jpg"``.

    Returns
    -------
    int
        The integer frame index extracted from the filename stem, or ``0``
        if no digits are found after the ``"frame"`` marker.
    """
    m = re.search(r'(\d+)', Path(p).stem.split("frame")[-1])
    return int(m.group(1)) if m else 0


def load_image(path):
    """Load and preprocess a JPEG image for model input.

    Parameters
    ----------
    path : tf.string
        Scalar string tensor (or Python ``str``) containing the path to a
        JPEG image file.

    Returns
    -------
    tf.Tensor
        Float32 tensor of shape ``(*IMG_SIZE, 3)`` with values in
        ``[0, 1]``.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment(img):
    """Apply standard data augmentation to a single image tensor.

    Applies the following stochastic transforms in order: random
    horizontal flip, random vertical flip, random brightness adjustment
    (max delta ±0.15), random contrast adjustment (factor in
    ``[0.8, 1.2]``), and a final clip to ``[0, 1]``.  Intended for the
    Fire and Lake classes during training.

    Parameters
    ----------
    img : tf.Tensor
        Float32 image tensor of shape ``(*IMG_SIZE, 3)`` with values in
        ``[0, 1]``.

    Returns
    -------
    tf.Tensor
        Augmented float32 tensor of the same shape as ``img``, with
        values clipped to ``[0, 1]``.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def augment_nofire(img):
    """Apply aggressive colour-jitter augmentation to a No Fire image tensor.

    Applies stronger photometric distortions than :func:`augment` to force
    the model to learn structural features rather than colour temperature.
    Without this, No Fire scenes with warm hues (sunsets, dry grass, red
    rooftops) are systematically mis-classified as Fire.

    Transforms applied (in order):

    * Random horizontal and vertical flips
    * Strong brightness shift  (max delta ±0.40)
    * Wide contrast stretch    (factor in [0.5, 1.8])
    * Random hue rotation      (max delta ±0.20, i.e. ±72° on the colour wheel)
    * Random saturation scale  (factor in [0.3, 2.0])
    * Final clip to [0, 1]

    Parameters
    ----------
    img : tf.Tensor
        Float32 image tensor of shape ``(*IMG_SIZE, 3)`` with values in
        ``[0, 1]``.

    Returns
    -------
    tf.Tensor
        Augmented float32 tensor of the same shape as ``img``, with
        values clipped to ``[0, 1]``.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.40)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.8)
    img = tf.image.random_hue(img, max_delta=0.20)
    img = tf.image.random_saturation(img, lower=0.3, upper=2.0)
    return tf.clip_by_value(img, 0.0, 1.0)


def make_dataset(paths, labels, augment_data=False, shuffle=False):
    """Build a batched, prefetched ``tf.data.Dataset`` from path and label lists.


    Parameters
    ----------
    paths : list[str or Path]
        File-system paths to JPEG images.
    labels : list[int]
        Integer class labels, one per path.
    augment_data : bool, optional
        If ``True``, apply :func:`augment` to every image after loading.
        Default is ``False``.
    shuffle : bool, optional
        If ``True``, shuffle the dataset with a full buffer and ``SEED``
        before mapping.  Default is ``False``.

    Returns
    -------
    tf.data.Dataset
        Dataset yielding ``(image_tensor, label)`` pairs where
        ``image_tensor`` is a float32 tensor of shape
        ``(BATCH_SIZE, *IMG_SIZE, 3)``.
    """
    paths_t  = tf.constant([str(p) for p in paths])
    labels_t = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(lambda p, l: (load_image(p), l), num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(lambda x, l: (augment(x), l), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

###########################################################
#            Binary  and Tertiary    Dataset              #
###########################################################

def build_binary_datasets(fire_deduped, nofire_deduped, lake_deduped, lake_as_nofire=True):
    """Construct train/val/test (tf.data) datasets for binary classification.

    Splits each deduplicated class list into train and validation subsets.
    Labels are Fire=1 and No Fire=0.

    When (lake_as_nofire=True) lake images are folded into class 0 for
    both train and validation.

    When (lake_as_nofire=False) lake images are excluded from train and validation entirely.

    The test set always uses the official (Test/Fire) and (Test/No_Fire) folders regardless
    of this flag.

    Training data is shuffled and augmented; validation and
    test data are not.

    Parameters
    ----------
    fire_deduped : list[str or Path]
        Deduplicated paths for the Fire class.
    nofire_deduped : list[str or Path]
        Deduplicated paths for the No Fire class.
    lake_deduped : list[str or Path]
        Deduplicated paths for the Lake class.
    lake_as_nofire : bool, optional
        If True (default), lake images are included as class 0.
        If False, lake images are excluded from train and validation.

    Returns
    -------
    train_ds : tf.data.Dataset
    val_ds : tf.data.Dataset
    test_ds : tf.data.Dataset
    train_labels : list[int]
    val_labels : list[int]
    test_labels : list[int]
    """
    fire_train,   fire_val   = temporal_split(fire_deduped)
    nofire_train, nofire_val = temporal_split(nofire_deduped)
    lake_train,   lake_val   = temporal_split(lake_deduped)

    if lake_as_nofire==True:
        # train
        b_train_paths  = fire_train + nofire_train + lake_train
        b_train_labels = ([1]*len(fire_train) + [0]*len(nofire_train)
                        + [0]*len(lake_train))
        # val
        b_val_paths    = fire_val + nofire_val + lake_val
        b_val_labels   = ([1]*len(fire_val) + [0]*len(nofire_val)
                        + [0]*len(lake_val))
    if lake_as_nofire==False:
        # train
        b_train_paths  = fire_train + nofire_train
        b_train_labels = [1]*len(fire_train) + [0]*len(nofire_train)
        # val
        b_val_paths    = fire_val + nofire_val
        b_val_labels   = [1]*len(fire_val) + [0]*len(nofire_val)
    # test
    test_fire_paths   = list((TEST_DIR / "Fire").glob("*.jpg"))
    test_nofire_paths = list((TEST_DIR / "No_Fire").glob("*.jpg"))
    b_test_paths  = test_fire_paths + test_nofire_paths
    b_test_labels = [1]*len(test_fire_paths) + [0]*len(test_nofire_paths)

    train_ds = make_dataset(b_train_paths, b_train_labels,
                            augment_data=True, shuffle=True)
    val_ds   = make_dataset(b_val_paths,   b_val_labels)
    test_ds  = make_dataset(b_test_paths,  b_test_labels)
    return train_ds, val_ds, test_ds, b_train_labels,b_val_labels, b_test_labels

def build_ternary_datasets(fire_deduped, nofire_deduped, lake_deduped):
    """Construct train/val/test tf.data datasets for 3-class classification.

    Split strategy
    --------------
    - Fire / No Fire / Lake : temporal_split (80/20 train/val)
    - Official test set     : Test/Fire + Test/No_Fire (no lake images)
    - Lake test set         : 10% of lake_deduped sampled at even intervals,
                              held out from training entirely.
                              The remaining 90% go through temporal_split
                              as usual (lake_train / lake_val).

    Labels: Fire=0, No Fire=1, Lake=2.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    train_labels, val_labels, test_labels : list[int]
    """
    # ── Lake: carve out 10% for test first, before any train/val split ────────
    lake_sorted  = sorted(lake_deduped, key=frame_number)
    n_lake_test  = max(1, len(lake_sorted) // 10)
    # sample evenly across the sequence so all scenes are represented
    test_indices = list(range(0, len(lake_sorted), len(lake_sorted) // n_lake_test))[:n_lake_test]
    lake_test    = [lake_sorted[i] for i in test_indices]
    lake_rest    = [p for i, p in enumerate(lake_sorted) if i not in set(test_indices)]

    fire_train,   fire_val   = temporal_split(fire_deduped)
    nofire_train, nofire_val = temporal_split(nofire_deduped)
    lake_train,   lake_val   = temporal_split(lake_rest)

    # ── Train ─────────────────────────────────────────────────────────────────
    t_train_paths  = fire_train + nofire_train + lake_train
    t_train_labels = ([0]*len(fire_train) + [1]*len(nofire_train)
                    + [2]*len(lake_train))

    # ── Val ───────────────────────────────────────────────────────────────────
    t_val_paths  = fire_val + nofire_val + lake_val
    t_val_labels = ([0]*len(fire_val) + [1]*len(nofire_val)
                  + [2]*len(lake_val))

    # ── Test ──────────────────────────────────────────────────────────────────
    test_fire_paths   = list((TEST_DIR / "Fire").glob("*.jpg"))
    test_nofire_paths = list((TEST_DIR / "No_Fire").glob("*.jpg"))

    t_test_paths  = test_fire_paths + test_nofire_paths + lake_test
    t_test_labels = ([0]*len(test_fire_paths) + [1]*len(test_nofire_paths)
                   + [2]*len(lake_test))

    train_ds = make_dataset(t_train_paths, t_train_labels,
                                     augment_data=True, shuffle=True)
    val_ds   = make_dataset(t_val_paths,   t_val_labels)
    test_ds  = make_dataset(t_test_paths,  t_test_labels)

    print(f"Lake  — test: {len(lake_test)}  |  train: {len(lake_train)}  |  val: {len(lake_val)}")
    print(f"Fire  — train: {len(fire_train)}  |  val: {len(fire_val)}")
    print(f"No Fire — train: {len(nofire_train)}  |  val: {len(nofire_val)}")

    return train_ds, val_ds, test_ds, t_train_labels, t_val_labels, t_test_labels




###########################################################
#           Model Evaluation                              #
###########################################################
def plot_history(history, title, metrics=("loss", "accuracy", "auc")):
    """Plot training vs. validation curves for a Keras training history.


    Parameters
    ----------
    history : keras.callbacks.History
        Object returned by ``model.fit()``.
    title : str
        Overall figure title displayed as a bold super-title.
    metrics : tuple[str], optional
        Metric names to plot.

    Returns
    -------
    None
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), dpi=150)
    for ax, m in zip(axes, metrics):
        if m not in history.history:
            ax.set_visible(False)
            continue
        ax.plot(history.history[m],     label="train")
        ax.plot(history.history[f"val_{m}"], label="val", linestyle="--")
        ax.set_title(m.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Render a row-normalised confusion matrix as an annotated seaborn heatmap.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth integer class labels.
    y_pred : array-like of int
        Predicted integer class labels.
    class_names : list[str]
        Ordered list of human-readable class names used for axis tick
        labels.  Must align with the integer label encoding.
    title : str, optional
        Plot title.  Default is ``"Confusion Matrix"``.

    Returns
    -------
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised (recall per class)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_norm,
        annot=False,          # we'll add custom annotations below
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Recall"},
        ax=ax,
    )

    # Annotate each cell with both the count and the percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct   = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j + 0.5, i + 0.5,
                    f"{pct:.1%}\n({count:,})",
                    ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    ax.set_xticks([x + 0.5 for x in range(len(class_names))])
    ax.set_yticks([y + 0.5 for y in range(len(class_names))])
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12, rotation=0)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

def evaluate_binary(model, test_ds, test_labels, model_name):
    """Run inference and report metrics for a binary classifier.


    Parameters
    ----------
    model : keras.Model
        Trained binary classification model whose output is a scalar
        sigmoid probability.
    test_ds : tf.data.Dataset
        Batched, prefetched dataset of ``(image_tensor, label)`` pairs.
    test_labels : list[int] or array-like
        Ground-truth labels corresponding to the order of samples in
        ``test_ds``.
    model_name : str
        Human-readable model identifier used in printed output and stored
        in the returned metrics dictionary.

    Returns
    -------
    y_true : np.ndarray of int
        Ground-truth binary labels.
    y_pred : np.ndarray of int
        Predicted binary labels (thresholded at 0.5).
    metrics_dict : dict
        Dictionary with keys ``"model"`` (str), ``"accuracy"`` (float),
        and ``"auc"`` (float).
    """
    y_proba = model.predict(test_ds, verbose=0).flatten()
    y_true  = np.array(test_labels)
    y_pred  = (y_proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score   = auc(fpr, tpr)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(
        y_true, y_pred,
        target_names=["No Fire", "Fire"]
    ))
    print(f"AUC: {auc_score:.4f}")



    return y_true, y_pred, y_proba,{"model": model_name, "accuracy": (y_pred==y_true).mean(),
            "auc": auc_score}
