import pandas as pd
import os

# Paths
image_dir = r'Dataset\images'  # replace with your actual images directory path
metadata_path = r'Dataset\meta_deta.csv'            # path to your metadata CSV
output_csv = r'Dataset\preprocess-meta_data.csv' # output CSV file

# Step 1: Load metadata
metadata_df = pd.read_csv(metadata_path)
print(f"[INFO] Total rows in metadata: {len(metadata_df)}")

# Step 2: Ensure image_id column is clean
metadata_df['image_id'] = metadata_df['image_id'].astype(str).str.strip().str.lower()

# Step 3: Get actual image filenames (with .jpg)
valid_image_filenames = {f.strip().lower() for f in os.listdir(image_dir) if f.lower().endswith('.jpg')}
print(f"[INFO] Total images in directory: {len(valid_image_filenames)}")
print(f"[DEBUG] First 5 image filenames: {list(valid_image_filenames)[:5]}")

# Step 4: Filter rows where image_id (which already contains .jpg) exists in the image folder
filtered_df = metadata_df[metadata_df['image_id'].isin(valid_image_filenames)]
print(f"[INFO] Matching entries found: {len(filtered_df)}")

# Step 5: Remove .jpg from image_id (optional, if needed later)
filtered_df['image_id'] = filtered_df['image_id'].str.replace('.jpg', '', regex=False)

# Step 6: Save only image_id and label columns
filtered_df[['image_id', 'label']].to_csv(output_csv, index=False)
print(f"✅ Saved {len(filtered_df)} entries to '{output_csv}'")
