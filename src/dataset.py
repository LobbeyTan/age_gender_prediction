import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


# Create a WikiDataset which extend torch Dataset class
class WikiDataset(Dataset):

    def __init__(self, root_dir, train=True, tts_ratio=0.8, balance=False) -> None:
        # Specify the root dir
        self.root_dir = root_dir

        # Read the metadata csv file
        self.data = pd.read_csv(
            os.path.join(
                root_dir, "balance_data.csv" if balance else "preprocessed_data.csv")
        )

        # Obtain the training size after train-test split
        train_size = int(self.data.shape[0] * tts_ratio)
        # Obtain the test size
        test_size = self.data.shape[0] - train_size

        # Set the total size of datasets
        self.size = train_size if train else test_size

        # Configure transform process for input image (image preprocessing)
        self.transform = transforms.Compose(
            [
                transforms.Resize(286, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        # Return the size of dataset
        return self.size

    def __getitem__(self, index: int):
        if index >= self.size:
            raise IndexError(
                f"{index} is out of bound, dataset has only {self.size} records"
            )

        # Get the full image path
        path = os.path.join(self.root_dir, self.data['image_path'][index])
        # Load image using PIL and convert to RGB
        img = Image.open(path).convert('RGB')

        # Get the target age value
        age = int(self.data['age'][index])
        # Get the target gender value
        gender = int(self.data['gender'][index])

        # Return in the form of Dictionary which is required for dataloader
        return {
            'img': self.transform(img),
            'path': path,
            'age': age,
            'gender': gender,
        }

    def extractImages(self):
        # Legacy code to extract all the images and load into memory.
        # The dataset is too big (3GB++) and will cause memory exceed limit error if load all images in one time
        paths = []
        images = []

        for file in self.data['image_path']:
            path = os.path.join(self.root_dir, file)
            img = Image.open(path).convert('RGB')

            paths.append(path)
            images.append(img)

        return images, paths
