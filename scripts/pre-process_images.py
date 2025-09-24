import pandas as pd
import os
import shutil

# Paths
image_dir = r'Dataset\images'  # replace with your actual images directory path
metadata_path = r'Dataset\preprocess-meta_data.csv'            # path to your metadata CSV
output_base_dir = r'Dataset\raw' # output CSV file


# Load preprocessed metadata
df = pd.read_csv(metadata_path)

# Make sure output folders exist for each label
labels = df['label'].unique()
for label in labels:
    label_dir = os.path.join(output_base_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

# Move files
moved = 0
for _, row in df.iterrows():
    image_file = f"{row['image_id']}.jpg"
    src_path = os.path.join(image_dir, image_file)
    dst_path = os.path.join(output_base_dir, str(row['label']), image_file)

    # Check if the image actually exists
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        moved += 1
    else:
        print(f"[WARN] File not found: {src_path}")

print(f"✅ Done! Moved {moved} images to '{output_base_dir}/[label]/' folders.")


import splitfolders

# Source directory with label folders: dataset/0, dataset/1, ..., dataset/4
input_folder = r'Dataset\raw'  # Where your organized images by label exist
output_folder = r'Dataset\dataset_split'  # New folder to create train/val/test split

# Split with equal class distribution
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .15, .15), group_prefix=None)

print("✅ Dataset split complete!")

