#import numpy as np 
import matplotlib.pyplot as plt
import cv2
import torch
from types import SimpleNamespace
import glob, re

cfg = SimpleNamespace(**{})


cfg.image_size = 256
cfg.device = "cpu"
model0 = torch.load('./new_file.h5')
model0.eval()

def LoadImage (image_path=str('./img.png')):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (cfg.image_size, cfg.image_size))
    image = image / 255.0
    #image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2).to(cfg.device)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(cfg.device)
    return image_tensor


def GetLabel (image_tensor):
    with torch.no_grad():
        output = model0(image_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label

class_labels = ["helical", "skyrmions", "random", "saturatedzp"]

"""
image_path = "./img_sk2.png"  
image_tensor fig, axs = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        # Load and plot the image
        image = plt.imread(my_files[i*2 + j])
        axs[i, j].imshow(image)
        axs[i, j].axis("off")

        # Add the corresponding name as a title
        axs[i, j].set_title(my_names[i*2 + j])

plt.tight_layout()
plt.show()= LoadImage(image_path=image_path)
predicted_label = GetLabel(image_tensor=image_tensor)
print("Predicted label:", predicted_label, class_labels[predicted_label])
"""


my_files = glob.glob("test*")
my_names = []
# Sort the file names based on the number at the end
#my_files.sort(key=lambda x: float(re.findall(r"\d+\.\d+$", x)[0]))

for my_img in my_files:
    my_image_tensor = LoadImage(my_img)
    predicted_label = GetLabel(my_image_tensor)
    name = class_labels[predicted_label]
    my_names.append(name)
    print (f'filename= {my_img}; label= {predicted_label};, name= {name}')

fig, axs = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        # Load and plot the image
        image = plt.imread(my_files[i*2 + j])
        axs[i, j].imshow(image)
        axs[i, j].axis("off")

        # Add the corresponding name as a title
        axs[i, j].set_title(my_names[i*2 + j])

plt.tight_layout()
plt.show()
