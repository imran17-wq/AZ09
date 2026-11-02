import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.random.mtrand import rand
# Change this to your actual dataset folder path
DATASET_PATH = Path(r"C:\Users\dudek\Downloads\A2Z")
images = []
labels = []
# Read the 26 letter folders
for folder in sorted(DATASET_PATH.iterdir()):
    if not folder.is_dir():
        continue
    label = folder.name.upper()
    if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        continue
    for file_path in folder.glob("*.jpg"):
        try:
            img = Image.open(file_path).convert("L")
            img = img.resize((28, 28))
            pixels = np.array(img, dtype=np.float32) / 255.0
            images.append(pixels.flatten())
            labels.append(label)
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
X=np.array(images,dtype=np.float32)
y=np.array(labels)
letters = np.unique(y)
label_to_id = {letter: i for i, letter in enumerate(letters)}
y_numeric = np.array([label_to_id[label] for label in y])
np.random.seed(42)
indices = np.random.permutation(len(X))
X = X[indices]
y_numeric = y_numeric[indices]
split_point = int(len(X) * 0.8)
X_train = X[:split_point]
y_train = y_numeric[:split_point]
X_test = X[split_point:]
y_test = y_numeric[split_point:]
w1=[]
for v in range(128):
    for i in range(784):
        w=random.uniform(-1,1)
        w1.append(w)
b1=[0]*128
bias=np.array(b1)
w1=np.array(w1)
w1=np.reshape(w1,(128,784)).T
def relu(x):
    return np.maximum(0, x)
z1 = np.dot(X_train[0], w1) + bias
a1 = relu(z1)
w2=[]
for v in range(128):
    for i in range(26):
        g=random.uniform(-1,1)
        w2.append(g)
w2=np.array(w2)
w2=np.reshape(w2,(128,26))
b2=[0]*26
b2=np.array(b2)
z2 = np.dot(a1,w2) + b2
z2=np.exp(z2)
total=sum(z2)
probs=z2/total
d=y_train[0]
p=probs[d]
target_values=[0]*26
target_values=np.array(target_values)
target_values[d]=1
output_error=probs-target_values
dw2=np.outer(a1,output_error)
db2=output_error
hidden_error=np.dot(w2,output_error)
relu_mask=[]
for i in z1:
    if i>0:
        relu_mask.append(True)
    else:
        relu_mask.append(False)
hidden_delta=[]
for i in range(len(relu_mask)):
    if relu_mask[i] is True:
        hidden_delta.append(hidden_error[i])
    else:
        hidden_delta.append(0)
dw1=np.outer(X_train[0],hidden_delta)
print(len(dw1),dw1.shape,dw1)