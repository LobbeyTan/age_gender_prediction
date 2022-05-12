import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class WikiDataset(Dataset):

    def __init__(self, root_dir, train=True, tts_ratio=0.8) -> None:

        self.root_dir = root_dir

        self.data = pd.read_csv(
            os.path.join(root_dir, "preprocessed_data.csv")
        )

        train_size = int(self.data.shape[0] * tts_ratio)
        test_size = self.data.shape[0] - train_size

        self.size = train_size if train else test_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(286, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if index >= self.size:
            raise IndexError(
                f"{index} is out of bound, dataset has only {self.size} records"
            )

        path = os.path.join(self.root_dir, self.data['image_path'][index])
        img = Image.open(path).convert('RGB')

        age = int(self.data['age'][index])
        gender = int(self.data['gender'][index])

        return {
            'img': self.transform(img),
            'path': path,
            'age': age,
            'gender': gender,
        }

    def extractImages(self):
        paths = []
        images = []

        for file in self.data['image_path']:
            path = os.path.join(self.root_dir, file)
            img = Image.open(path).convert('RGB')

            paths.append(path)
            images.append(img)

        return images, paths
