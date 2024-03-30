# MLP_trafic_signal
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
data_dir = Path("D:/021_Devarshi/FullIJCNN2013")
image_dir = os.path.join(data_dir)
gt_file = Path("D:/021_Devarshi/FullIJCNN2013/gt.txt")

grd_truth = []
with open(gt_file, "r") as file:
    for line in file:
        data = line.strip().split(";")
        filename = data[0]
        leftCol = int(data[1])
        topRow = int(data[2])
        rightCol = int(data[3])
        bottomRow = int(data[4])
        classID = int(data[5])
        grd_truth.append([filename, (leftCol, topRow, rightCol, bottomRow), classID])

grd_truth
gt_array = np.array(grd_truth)
gt_array
df = pd.DataFrame(gt_array, columns=["Filename","Dimensions","ClassID"])
df
from PIL import Image
features = []
for filename, dim, _ in gt_array:
    img = Image.open(os.path.join(image_dir, filename))
    width, height = img.size
    y1, y2, x1, x2 = max(0, dim[1]), min(height, dim[3]), max(0, dim[0]), min(width, dim[2])
    img = img.crop((x1, y1, x2, y2))
    img = img.resize((30,30))
    img_arr = np.array(img)
    features.append(img_arr)
    
features
labels = df["ClassID"]
labels
len(features), len(labels)
labels = list(labels.values)
# Plot image
import math

num_class, num_subplots = len(np.unique(labels)), 10
num_rows = math.ceil(num_class/num_subplots)

fig, axes = plt.subplots(num_rows,num_subplots, figsize=(10,num_rows))

for ind, class_label in enumerate(np.unique(labels)):
    sample = features[labels.index(class_label)]
    ax = axes[ind // num_subplots, ind % num_subplots]
    ax.imshow(sample)
    ax.set_title(f"Class: {class_label}")
    
for ax in axes.flatten():
    if not ax.images:
        ax.axis("off")

plt.tight_layout()
plt.show()
# Plot distribution of classes
plt.hist(labels, bins=num_class, rwidth=0.8)
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Distribution of classes")
plt.show()
# Normalize the features
features = np.array(features) / 255.0
features = np.array([img.flatten() for img in features])
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
len(X_train), len(X_test), len(y_train), len(y_test)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# Define classifier
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f"Accuracy: {accuracy_score(y_pred, y_test)}")
print(classification_report(y_pred, y_test))
