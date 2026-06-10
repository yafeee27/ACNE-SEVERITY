# -*- coding: utf-8 -*-
"""
Acne Severity Classification — YOLOv8 Training Script
======================================================
YOLOv8 nano classification (yolov8n-cls) transfer learning for 3-class acne severity detection.
Classes: Parah (severe), Ringan (mild), Sedang (moderate)

Usage:
    python train_yolo.py
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ===========================================================
# Paths
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "test", "test")  # Parah / Ringan / Sedang
SPLIT_DIR   = os.path.join(BASE_DIR, "acne_split")

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

def main():
    print("=" * 50)
    print("Acne Severity Classification — YOLOv8 Training")
    print("=" * 50)

    # 1. Pastikan dataset split sudah siap
    dataset_ready = False
    if os.path.exists(SPLIT_DIR):
        subfolders = ["train", "val", "test"]
        if all(os.path.exists(os.path.join(SPLIT_DIR, f)) for f in subfolders):
            try:
                train_images_count = sum(
                    len(os.listdir(os.path.join(SPLIT_DIR, "train", c)))
                    for c in ["Parah", "Ringan", "Sedang"]
                    if os.path.exists(os.path.join(SPLIT_DIR, "train", c))
                )
                if train_images_count > 0:
                    dataset_ready = True
                    print(f"Dataset split ditemukan dan valid di: {SPLIT_DIR} ({train_images_count} gambar latihan)")
            except Exception:
                pass

    if not dataset_ready:
        print("Dataset split tidak ditemukan atau tidak valid. Menyiapkan dataset split...")
        if os.path.exists(DATASET_DIR):
            prepare_splits()
        else:
            print(f"❌ Error: Folder dataset asal tidak ditemukan di: {DATASET_DIR}")
            return

    # 2. Memuat pre-trained model YOLOv8 Klasifikasi (Nano)
    print("\n>>> Memuat model pre-trained YOLOv8 nano classification (yolov8n-cls)...")
    model = YOLO("yolov8n-cls.pt")

    # 3. Jalankan training
    abs_split_dir = os.path.abspath(SPLIT_DIR)
    print(f"\n>>> Memulai training YOLOv8 pada dataset: {abs_split_dir}")
    
    # Melatih selama 15 epoch untuk demo/pengujian awal
    results = model.train(
        data=abs_split_dir,
        epochs=15,
        imgsz=224,
        batch=16,
        project="models",
        name="yolo_acne",
        exist_ok=True,
    )

    # 4. Copy model terbaik (best.pt) ke root project
    save_dir = getattr(model.trainer, 'save_dir', None)
    if save_dir:
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
    else:
        # Fallback manual path jika trainer tidak dapat diakses
        best_model_path = os.path.join("runs", "classify", "models", "yolo_acne", "weights", "best.pt")
        
    # Tambahan pengecekan fallback alternatif
    alternative_paths = [
        best_model_path,
        os.path.join(BASE_DIR, "runs", "classify", "models", "yolo_acne", "weights", "best.pt"),
        os.path.join("models", "yolo_acne", "weights", "best.pt")
    ]
    
    found_path = None
    for p in alternative_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if found_path:
        dest_path = os.path.join(BASE_DIR, "final_model_yolo.pt")
        shutil.copy2(found_path, dest_path)
        print(f"\n✅ Training selesai!")
        print(f"   Model terbaik ditemukan di: {found_path}")
        print(f"   Model disalin ke root project: {dest_path}")
    else:
        print("\n❌ Error: File best.pt tidak ditemukan di folder output training.")
        print("   Silakan periksa folder 'runs/classify/models/yolo_acne/weights/' secara manual.")

if __name__ == "__main__":
    main()
