from matplotlib.patches import Rectangle
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from types import SimpleNamespace
import os
from albumentations.pytorch import ToTensorV2
import albumentations as A
import timm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

pd.set_option("display.max_colwidth", None)
cfg = SimpleNamespace(**{})
cfg.root_dir = './images/'
cfg.n_classes = 4
cfg.batch_size = 32
cfg.image_size = 256
cfg.backbone = "resnet18"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.learning_rate = 1e-4
cfg.lr_min = 1e-5
cfg.epochs = 5
cfg.seed = 42
cfg.n_folds = 3  # 5
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg.num_workers = 4
sub_folders = ["helical/", "skyrmions/", "./random",  "./saturatedzp" ]
labels = [0,1,2,3]
label_names = ['helical', 'skyrmions', 'random', 'saturatedzp'] 
data = []
print(cfg.root_dir)
for s, l in zip(sub_folders, labels):
    # print (s,l)
    for r, d, f in os.walk(cfg.root_dir + s):
        for file in f:
            # print (file)
            if ".png" in file:
                # print ('found image')
                data.append((os.path.join(s, file), l))
df = pd.DataFrame(data, columns=['file_name', 'label'])
df = df.sample(frac=1, random_state=cfg.seed).reset_index(
    drop=True)  # , inplace=True)
df.head()
df.label.value_counts()
df.info()
test_len = 1
test_df = df[-test_len:].reset_index(drop=True)
df = df[:-test_len]
test_df


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self,
                 cfg,
                 df,
                 transform=None,
                 mode="val"):
        self.root_dir = cfg.root_dir
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df['label'].values
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(cfg.image_size, cfg.image_size),
                ToTensorV2(),])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file_path and label for index
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # Convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        augmented = self.transform(image=image)
        image = augmented['image']

        # Normalize because ToTensorV2() doesn't normalize the image
        image = image/255

        return image, label


example_dataset = CustomDataset(cfg, df)

example_dataloader = DataLoader(example_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=0,)


def calculate_metric(y, y_pred):
    metric = accuracy_score(y, y_pred)
    return metric


def train_one_epoch(dataloader, model, optimizer, scheduler, cfg, lrs):
    model.train()
    final_y = []
    final_y_pred = []
    final_loss = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg.device)
        y = batch[1].to(cfg.device)
        #print ("type_y ",type(y))
        #print (y)
        #quit()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()
            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)

    return metric, loss, lrs


def validate_one_epoch(dataloader, model, cfg):
    model.eval()
    final_y = []
    final_y_pred = []
    final_loss = []
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg.device)
        y = batch[1].to(cfg.device)
        with torch.no_grad():
            y_pred = model(X)
            loss = criterion(y_pred, y)
            y = y.detach().cpu().numpy().tolist()
            y_pred = y_pred.detach().cpu().numpy().tolist()
            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)
    return metric, loss


def fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader=None):
    lrs = []
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        set_seed(cfg.seed + epoch)
        acc, loss, lrs = train_one_epoch(
            train_dataloader, model, optimizer, scheduler, cfg, lrs)
        if valid_dataloader:
            val_acc, val_loss = validate_one_epoch(
                valid_dataloader, model, cfg)
        print(f'Loss: {loss:.4f} Acc: {acc:.4f}')
        acc_list.append(acc)
        loss_list.append(loss)
        if valid_dataloader:
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
    return acc_list, loss_list, val_acc_list, val_loss_list, model, lrs


model = timm.create_model(cfg.backbone,
                          pretrained=True,
                          num_classes=cfg.n_classes)

X = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size)
y = model(X)
y.shape

transform_soft = A.Compose([A.Resize(cfg.image_size, cfg.image_size),
                            A.Rotate(p=0.6, limit=[-45, 45]),
                            A.HorizontalFlip(p=0.6),
                            A.CoarseDropout(
                                max_holes=1, max_height=64, max_width=64, p=0.3),
                            ToTensorV2()])

example_dataset = CustomDataset(cfg, df, transform=transform_soft)

example_dataloader = DataLoader(
    example_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,)
df["kfold"] = -1
skf = StratifiedKFold(n_splits=cfg.n_folds)
for fold, (train_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = fold

for fold in range(cfg.n_folds):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = CustomDataset(cfg, train_df, transform=transform_soft)
    valid_dataset = CustomDataset(cfg, valid_df)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=0,)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=0,)

    model = timm.create_model(cfg.backbone,
                              pretrained=True,
                              num_classes=cfg.n_classes)

    model = model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate,
                                 weight_decay=0,)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=np.ceil(
                                                               len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs,
                                                           eta_min=cfg.lr_min)

    acc, loss, val_acc, val_loss, model, lrs = fit(
        model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader)


train_df = df.copy()

train_dataset = CustomDataset(cfg, train_df, transform=transform_soft)

train_dataloader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=0,
                              )

model = timm.create_model(cfg.backbone,
                          pretrained=True,
                          num_classes=cfg.n_classes)

model = model.to(cfg.device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=cfg.learning_rate,
                             weight_decay=0,
                             )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=np.ceil(
                                                           len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs,
                                                       eta_min=cfg.lr_min)

acc, loss, val_acc, val_loss, model, lrs = fit(
    model, optimizer, scheduler, cfg, train_dataloader)

test_dataset = CustomDataset(cfg, test_df)

test_dataloader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=0,
                             )


# def predict(dataloader, model, cfg):
dataloader = test_dataloader
# Validation mode
model.eval()

final_y = []
final_y_pred = []

# Iterate over data
for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    X = batch[0].to(cfg.device)
    y = batch[1].to(cfg.device)
    print ('X =',X)
    print ()
    with torch.no_grad():
        y_pred = model(X)
        y = y.detach().cpu().numpy().tolist()
        y_pred = y_pred.detach().cpu().numpy().tolist()
        final_y.extend(y)
        final_y_pred.extend(y_pred)
final_y_pred_argmax = np.argmax(final_y_pred, axis=1)
metric = calculate_metric(final_y, final_y_pred_argmax)
print(metric)
calculate_metric(final_y, final_y_pred_argmax)
test_df['prediction'] = final_y_pred_argmax
#test_df

tdate = datetime.datetime.now().strftime("%H_%M__%d_%m_%Y")
fname_h5 = 'new_file' + str(tdate) + '.h5'
fname_pth = 'trained_model'+str(tdate)+'pth'


torch.save(model.state_dict(), fname_pth)
torch.save(model, fname_h5)
quit()
for (idx, batch) in enumerate(test_dataloader):
    fig, ax = plt.subplots(2, 4, figsize=(10, 6))
    idx = 0
    for i in range(2):
        for j in range(4):
            label = batch[1][idx]
            image = batch[0][idx]
            pred = final_y_pred_argmax[idx]
            # Reshape image
            image = image.permute(1, 2, 0)
            ax[i, j].imshow(image)
            # \n{df.file_name[idx]}")#, fontsize=14)
            my_label = label_names[label]
            my_pred = label_names[pred]
            ax[i, j].set_title(
                f"Ground Truth: {my_label}  \n Prediction: {my_pred}  ")
            ax[i, j].axis('off')
            idx = idx+1
            color = 'green' if label == pred else 'red'
            ax[i, j].add_patch(Rectangle((0, 0), cfg.image_size, cfg.image_size,
                                         alpha=1, edgecolor=color, linewidth=5, fill=None))
    plt.tight_layout()
    fig.savefig(f'./inference_prediction.jpg',
                transparent=False,  facecolor='white')
    plt.show()
    plt.show
