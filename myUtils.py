import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


def make_annotations_file(dataset_path: str, type: str) -> None:

    '''
    Generates an annotation file based on elements in the specified folder.

    Parameters:
    - dataset_path (str): Path to the folder containing the data.
    - type (str): Type of annotation to include in the file name.

    Returns:
    None

    Reads the list of elements in the dataset_path folder, sorts them, and creates a DataFrame
    with a 'labels' column, where 1 indicates that the element name contains 'opened', and 0 otherwise.
    Saves the created DataFrame to a CSV file in the './Annotations/' folder
    with the name 'annotation_file_{type}.csv' without adding an index column.

    Example:
    make_annotations_file('/path/to/folder', 'annotation_type')
    '''

    list_of_elements = sorted(os.listdir(dataset_path))

    annotation = pd.DataFrame()
    annotation['labels'] = [1 if 'opened' in element else 0 for element in list_of_elements]
    
    annotation.to_csv(path_or_buf=f'./Annotations/annotation_file_{type}.csv', sep=',', index=False)


class myDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.index[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
dataset = myDataset(annotations_file='./Annotations/annotation_file_train.csv',
                    img_dir='./Dataset/Train/',
                    transform=None,
                    target_transform=None)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

import matplotlib.pyplot as plt

train_features, train_labels = next(iter(dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")