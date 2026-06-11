# -*- coding: utf-8 -*-
"""
Acne Dataset Merger & Formatter
==============================
Menggabungkan data dari dataset lama ('test/test') dan dataset baru ('Acne Level Detection.v1i.folder')
ke dalam satu folder terpadu ('dataset_unified') dengan struktur folder kelas berangka agar sinkron 
dengan indeks kelas YOLOv8:
  - Class 0: 0_Aman
  - Class 1: 1_Ringan
  - Class 2: 2_Sedang
  - Class 3: 3_Berat (gabungan Parah/Berat)

Usage:
    python merge_datasets.py
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_DATA_DIR = os.path.join(BASE_DIR, "test", "test")
NEW_DATA_DIR = os.path.join(BASE_DIR, "Acne Level Detection.v1i.folder")
UNIFIED_DIR  = os.path.join(BASE_DIR, "dataset_unified")

# Mapping dari nama folder asal ke folder baru
CLASS_MAPPING = {
    # Dataset Lama
    "Ringan": "1_Ringan",
    "Sedang": "2_Sedang",
    "Parah": "3_Berat",
    
    # Dataset Baru
    "acne_level_0": "0_Aman",
    "acne_level_1": "1_Ringan",
    "acne_level_2": "2_Sedang",
    "acne_level_3": "3_Berat"
}

def main():
    print("=" * 65)
    print("   PENGGABUNGAN DATASET ACNE KE FORMAT 4 KELAS UNIFIED")
    print("=" * 65)
    
    # 1. Reset / Buat folder target
    if os.path.exists(UNIFIED_DIR):
        print(f"Mengosongkan folder lama: {UNIFIED_DIR}")
        shutil.rmtree(UNIFIED_DIR)
    os.makedirs(UNIFIED_DIR, exist_ok=True)
    
    for c in ["0_Aman", "1_Ringan", "2_Sedang", "3_Berat"]:
        os.makedirs(os.path.join(UNIFIED_DIR, c), exist_ok=True)
        
    counts = {"0_Aman": 0, "1_Ringan": 0, "2_Sedang": 0, "3_Berat": 0}
    
    # 2. Salin data dari dataset lama (test/test)
    if os.path.exists(OLD_DATA_DIR):
        print("\nMenyalin gambar dari dataset lama (test/test)...")
        for old_class in ["Ringan", "Sedang", "Parah"]:
            old_path = os.path.join(OLD_DATA_DIR, old_class)
            if not os.path.exists(old_path):
                continue
                
            dest_class = CLASS_MAPPING[old_class]
            dest_path = os.path.join(UNIFIED_DIR, dest_class)
            
            for file_name in os.listdir(old_path):
                src_file = os.path.join(old_path, file_name)
                if os.path.isfile(src_file) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    dst_file = os.path.join(dest_path, f"old_{file_name}")
                    shutil.copy2(src_file, dst_file)
                    counts[dest_class] += 1
    else:
        print("\n⚠️ Peringatan: Folder dataset lama 'test/test' tidak ditemukan.")

    # 3. Salin data dari dataset baru (Acne Level Detection.v1i.folder)
    if os.path.exists(NEW_DATA_DIR):
        print("\nMenyalin gambar dari dataset baru (Acne Level Detection.v1i.folder)...")
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(NEW_DATA_DIR, split)
            if not os.path.exists(split_path):
                continue
                
            for new_class in ["acne_level_0", "acne_level_1", "acne_level_2", "acne_level_3"]:
                new_path = os.path.join(split_path, new_class)
                if not os.path.exists(new_path):
                    continue
                    
                dest_class = CLASS_MAPPING[new_class]
                dest_path = os.path.join(UNIFIED_DIR, dest_class)
                
                for file_name in os.listdir(new_path):
                    src_file = os.path.join(new_path, file_name)
                    if os.path.isfile(src_file) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        dst_file = os.path.join(dest_path, f"new_{split}_{file_name}")
                        shutil.copy2(src_file, dst_file)
                        counts[dest_class] += 1
    else:
        print("\n⚠️ Peringatan: Folder dataset baru 'Acne Level Detection.v1i.folder' tidak ditemukan.")
        
    print("\n" + "=" * 65)
    print("              RINGKASAN INTEGRASI DATASET")
    print("=" * 65)
    print(f"  🔵 0_Aman   : {counts['0_Aman']} gambar")
    print(f"  🟢 1_Ringan : {counts['1_Ringan']} gambar")
    print(f"  🟡 2_Sedang : {counts['2_Sedang']} gambar")
    print(f"  🔴 3_Berat  : {counts['3_Berat']} gambar")
    print("-" * 65)
    total_imgs = sum(counts.values())
    print(f"  Total gambar terpadu : {total_imgs} gambar")
    print("=" * 65)
    print("✅ Penggabungan selesai! Gambar disimpan di folder 'dataset_unified'.")
    print("👉 Silakan jalankan 'python train_yolo.py' untuk melatih ulang model.")

if __name__ == "__main__":
    main()
