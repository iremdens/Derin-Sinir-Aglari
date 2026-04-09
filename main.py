import os
import shutil
import pandas as pd

BASE_DIR = r"C:\Users\irem\Downloads\archive\ODIR-5K"
IMG_DIR = os.path.join(BASE_DIR, r"C:\Users\irem\Downloads\archive\ODIR-5K\ODIR-5K\Training Images")
CSV_PATH = os.path.join(BASE_DIR, r"C:\Users\irem\Downloads\archive\full_df.csv")

OUTPUT_DIR = "data/train"

df = pd.read_csv(CSV_PATH)

classes = {
    "N": "normal",
    "C": "cataract",
    "G": "glaucoma"
}

for cls in classes.values():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

for _, row in df.iterrows():
    labels = row['labels']
    img_name = row['Left-Fundus']  # sağ göz de var ama karışmasın

    for key in classes:
        if key in labels:
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(OUTPUT_DIR, classes[key], img_name)

            if os.path.exists(src):
                shutil.copy(src, dst)
            break  # sadece 1 label al

print("Bitti ✅")
