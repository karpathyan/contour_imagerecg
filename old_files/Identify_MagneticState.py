import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_colwidth", None)
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm
import torchvision.models as models
from types import SimpleNamespace
cfg = SimpleNamespace(**{})
cfg.image_size = 256
cfg.device = "cpu"
model0 = torch.load('./new_file.h5')
model0.eval()

# Load the image
image_path = "./sat1.png"  # Replace with the path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (cfg.image_size, cfg.image_size))
image = image / 255.0
#image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(cfg.device)
image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(cfg.device)

# Set the model to evaluation mode
#model.eval()

# Perform the prediction
with torch.no_grad():
    output = model0(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

# Print the predicted label
class_labels = ["helical", "skyrmions"]
print("Predicted label:", predicted_label)

