print("SCRIPT STARTED")

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

# ------------------------------
# CONFIG
# ------------------------------

IMAGE_FOLDER = "data/train"

# Skip macOS .DS_Store files
def is_valid_folder(name):
    return not name.startswith('.')

# ------------------------------
# LOAD DATASET
# ------------------------------

data = []

for class_name in os.listdir(IMAGE_FOLDER):
    if not is_valid_folder(class_name):
        continue

    class_path = os.path.join(IMAGE_FOLDER, class_name)

    if not os.path.isdir(class_path):
        continue

    print("READING CLASS:", class_name)

    for img_name in os.listdir(class_path):
        if img_name.startswith('.'):  
            continue

        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_name)
            continue

        h, w, c = img.shape
        brightness = img.mean()

        data.append([img_name, class_name, w, h, c, brightness])

df = pd.DataFrame(data, columns=["filename", "class_name", "width", "height", "channels", "brightness"]) 

print("\nDataset loaded successfully!")
print(df.head())
print(df['class_name'].value_counts())

# ------------------------------
# 1. CLASS DISTRIBUTION
# ------------------------------

plt.figure(figsize=(8,5))
sns.countplot(x=df["class_name"])
plt.title("Class Distribution")
plt.show()

# ------------------------------
# 2. IMAGE SIZE ANALYSIS (WIDTH/HEIGHT)
# ------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df['class_name'], y=df['width'])
plt.title("Width Distribution by Class")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['class_name'], y=df['height'])
plt.title("Height Distribution by Class")
plt.show()

# ------------------------------
# 3. ASPECT RATIO ANALYSIS
# ------------------------------

df['aspect_ratio'] = df['width'] / df['height']

plt.figure(figsize=(8,5))
sns.histplot(df["aspect_ratio"], bins=40, kde=True)
plt.title("Aspect Ratio Distribution")
plt.show()

# ------------------------------
# 4. BRIGHTNESS ANALYSIS
# ------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x=df['class_name'], y=df['brightness'])
plt.title("Brightness Comparison by Class")
plt.show()

# ------------------------------
# 5. RGB COLOR ANALYSIS
# ------------------------------

def avg_rgb(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    return np.mean(R), np.mean(G), np.mean(B)

r_vals, g_vals, b_vals = [], [], []

for row in df.itertuples():
    r, g, b = avg_rgb(f"data/train/{row.class_name}/{row.filename}")
    r_vals.append(r)
    g_vals.append(g)
    b_vals.append(b)

df['r_mean'] = r_vals
df['g_mean'] = g_vals
df['b_mean'] = b_vals

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="r_mean", y="g_mean", hue="class")
plt.title("RGB Color Mean (Red vs Green)")
plt.show()

# ------------------------------
# 6. CORRUPTED IMAGE DETECTION
# ------------------------------

corrupted = []

for row in df.itertuples():
    img_path = f"data/train/{row.class_name}/{row.filename}"
    img = cv2.imread(img_path)
    if img is None:
        corrupted.append(img_path)

print("\nCorrupted Images Found:", corrupted)

# ------------------------------
# 7. DUPLICATE IMAGE DETECTION
# ------------------------------

hashes = {}
duplicates = []

for row in df.itertuples():
    img_path = f"data/train/{row.class_name}/{row.filename}"
    with open(img_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash in hashes:
        duplicates.append(img_path)
    else:
        hashes[file_hash] = img_path

print("\nDuplicate Images Found:", duplicates)

# ------------------------------
# 8. SHOW RANDOM SAMPLES PER CLASS
# ------------------------------

for cls in df['class_name'].unique():
    subset = df[df['class_name'] == cls].sample(4)

    plt.figure(figsize=(8,4))
    for i, row in enumerate(subset.itertuples(), 1):
        img = cv2.imread(f"data/train/{row.class_name}/{row.filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,4,i)
        plt.imshow(img)
        plt.title(row.class_name)
        plt.axis("off")
    plt.suptitle(f"Sample Images for {cls}")
    plt.show()

print("\nANALYSIS COMPLETE!")