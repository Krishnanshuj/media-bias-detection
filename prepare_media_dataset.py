import os
import glob
import pandas as pd

BASE_PATH = r"C:\Users\krish\.cache\kagglehub\datasets\surajkarakulath\labelled-corpus-political-bias-hugging-face\versions\1"

folder_label_map = {
    "Center Data": "Neutral",
    "Left Data": "Left-Leaning",
    "Right Data": "Right-Leaning",
}


MAX_FILES_PER_CLASS = 2000  

records = []

print("Base path:", BASE_PATH)

for folder_name, label_name in folder_label_map.items():
    folder_path = os.path.join(BASE_PATH, folder_name)
    print(f"\nLooking in folder (recursively): {folder_path}")

    txt_files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
    print(f"  Found TXT files: {len(txt_files)}")

    if not txt_files:
        print(f"  WARNING: No TXT files found under {folder_path}")
        continue

    txt_files = txt_files[:MAX_FILES_PER_CLASS]
    print(f"  Will process at most {len(txt_files)} files for class {label_name}")

    for i, f in enumerate(txt_files, start=1):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            print(f"    ERROR reading {f}: {e}")
            continue

        if not text or not text.strip():
            continue

        records.append({
            "headline": text,
            "bias_3class": label_name
        })

        if i % 500 == 0:
            print(f"    Processed {i} files in {folder_name}")

if not records:
    raise FileNotFoundError("No text data loaded from any of the folders! (even after recursive search)")

df = pd.DataFrame(records)
print("\nDataframe shape:", df.shape)

print("\nLabel distribution:")
print(df["bias_3class"].value_counts())

df["len_words"] = df["headline"].str.split().apply(len)
print("\nLength stats:")
print(df["len_words"].describe())

df_final = df[["headline", "bias_3class"]]

output_path = r"C:\Users\krish\Downloads\Media Bias Detection\political_bias_3class.csv"
df_final.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset to: {output_path}")
