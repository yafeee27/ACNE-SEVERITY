# -*- coding: utf-8 -*-
"""
Acne Severity Classification — Training Script
================================================
MobileNetV3Small transfer learning for 3-class acne severity detection.
Classes: Parah (severe), Ringan (mild), Sedang (moderate)

Usage:
    python train.py
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ===========================================================
# Paths — all relative to this script's directory
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "test", "test")  # Parah / Ringan / Sedang

SPLIT_DIR   = os.path.join(BASE_DIR, "acne_split")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
FIGURE_DIR  = os.path.join(BASE_DIR, "figures")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")

# Auto-create output directories
for d in [MODEL_DIR, FIGURE_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# ===========================================================
# Hyperparameters
# ===========================================================

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 0.0001


def count_dataset():
    """Print per-class image counts."""
    total = 0
    for cls in sorted(os.listdir(DATASET_DIR)):
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        count = len(os.listdir(cls_path))
        total += count
        print(f"  {cls}: {count}")
    print("-" * 30)
    print(f"  Total: {total}")
    return total


def prepare_splits():
    """Create stratified train / val / test splits (70/15/15)."""

    # Reset split folder
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    os.makedirs(SPLIT_DIR)
    print("Split folder berhasil direset")

    # Build dataframe of all images
    data = []
    for label in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            data.append([os.path.join(class_path, img_name), label])

    df = pd.DataFrame(data, columns=["filepath", "label"])
    print(f"\nTotal images: {len(df)}")
    print(df.head())

    # Stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    print(f"\nTrain : {len(train_df)}")
    print(f"Val   : {len(val_df)}")
    print(f"Test  : {len(test_df)}")

    # Create directories and copy files
    for split in ["train", "val", "test"]:
        split_path = os.path.join(SPLIT_DIR, split)
        os.makedirs(split_path, exist_ok=True)
        for cls in sorted(df["label"].unique()):
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)

    def copy_files(dataframe, split_name):
        for _, row in dataframe.iterrows():
            src = row["filepath"]
            dst = os.path.join(
                SPLIT_DIR, split_name, row["label"], os.path.basename(src)
            )
            shutil.copy2(src, dst)

    copy_files(train_df, "train")
    copy_files(val_df, "val")
    copy_files(test_df, "test")
    print("Copy selesai")

    # Verify
    for split in ["train", "val", "test"]:
        print(f"\n===== {split.upper()} =====")
        split_path = os.path.join(SPLIT_DIR, split)
        total = 0
        for cls in sorted(os.listdir(split_path)):
            count = len(os.listdir(os.path.join(split_path, cls)))
            total += count
            print(f"  {cls}: {count}")
        print(f"  Total: {total}")


def create_generators():
    """Create image data generators for train / val / test."""

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    test_val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def build_model():
    """Build MobileNetV3Small with custom classification head."""

    base_model = MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train(model, train_gen, val_gen):
    """Train the model with class weights and callbacks."""

    # Compute class weights for imbalanced data
    labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weights}")
    print(f"Class indices: {train_gen.class_indices}")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, checkpoint],
    )

    return history


def evaluate(test_gen):
    """Evaluate the best model on the test set."""

    best_model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))

    test_loss, test_acc = best_model.evaluate(test_gen)
    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    # Classification report
    pred_probs = best_model.predict(test_gen)
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=test_gen.class_indices.keys(),
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save final model
    final_path = os.path.join(MODEL_DIR, "final_model.keras")
    best_model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # Also copy to project root for app.py
    root_model = os.path.join(BASE_DIR, "final_model.keras")
    shutil.copy2(final_path, root_model)
    print(f"Model copied to: {root_model}")

    return best_model


# ===========================================================
# Main
# ===========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Acne Severity Classification — Training")
    print("=" * 50)
    print(f"\nTensorFlow : {tf.__version__}")
    print(f"Base Dir   : {BASE_DIR}")
    print(f"Dataset    : {DATASET_DIR}")
    print()

    # 1. Count dataset
    print(">>> Dataset Summary")
    count_dataset()

    # 2. Prepare splits
    print("\n>>> Preparing train/val/test splits...")
    prepare_splits()

    # 3. Create generators
    print("\n>>> Creating data generators...")
    train_gen, val_gen, test_gen = create_generators()

    # 4. Build model
    print("\n>>> Building MobileNetV3Small model...")
    model = build_model()

    # 5. Train
    print("\n>>> Training...")
    history = train(model, train_gen, val_gen)

    # 6. Evaluate
    print("\n>>> Evaluating on test set...")
    evaluate(test_gen)

    print("\n" + "=" * 50)
    print("Training selesai!")
    print("=" * 50)
