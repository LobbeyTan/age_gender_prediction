import os
from torch import nn
import torch
import torchvision.transforms as transforms
from src.network import Classifier
from src.vgg16 import load_vgg16, vgg_preprocess


class AgeGenderPredictor(nn.Module):

    def __init__(self, lr=0.001, beta1=0.5, device=torch.device("cpu"), vgg_path="./src/pretrained/vgg16.weight") -> None:

        super(AgeGenderPredictor, self).__init__()

        self.device = device

        self.vgg = load_vgg16(vgg_path, self.device)

        self.age_model = Classifier(n_output=8).to(self.device)

        self.gender_model = Classifier(n_output=2).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.age_optimizer = torch.optim.Adam(
            self.age_model.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        self.gender_optimizer = torch.optim.Adam(
            self.gender_model.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, x):
        self.image_input = vgg_preprocess(x['img']).to(self.device)
        self.age_real = x['age'].to(self.device)
        self.gender_real = x['gender'].to(self.device)

        self.age_pred = self.age_model(self.vgg(self.image_input))
        self.gender_pred = self.gender_model(self.vgg(self.image_input))

        return torch.argmax(self.age_pred), torch.argmax(self.gender_pred)

    def optimize_parameters(self):

        self.age_optimizer.zero_grad()
        self.age_loss = self.criterion(self.age_pred, self.age_real)
        self.age_loss.backward()
        self.age_optimizer.step()

        self.gender_optimizer.zero_grad()
        self.gender_loss = self.criterion(self.gender_pred, self.gender_real)
        self.gender_loss.backward()
        self.gender_optimizer.step()

    def eval(self):
        self.age_model.eval()
        self.gender_model.eval()

    def test(self, x):
        with torch.no_grad():
            age, gender = self.forward(x)

        return age, gender

    def predict(self, img):
        img = self.transform(img)
        img = torch.unsqueeze(img, dim=0)
        img = vgg_preprocess(img)
        print(img.shape)
        with torch.no_grad():
            age = torch.argmax(self.age_model(self.vgg(img)))
            gender = torch.argmax(self.gender_model(self.vgg(img)))

        return age, gender

    def save_model(self, directory, epoch):
        self._save_network(self.age_model, "age", directory, epoch)
        self._save_network(self.gender_model, "gender", directory, epoch)

    def _save_network(self, network: torch.nn.Module, name, directory, epoch):
        directory = os.path.join(directory, "epoch_%s/" % epoch)

        try:
            os.makedirs(directory)
        except:
            pass

        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        torch.save(network.state_dict(), path)

    def load_model(self, directory):
        self._load_model(self.age_model, "age", directory)
        self._load_model(self.gender_model, "gender", directory)

    def _load_model(self, network: torch.nn.Module, name, directory):
        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        network.load_state_dict(torch.load(path, map_location=self.device))

        network.to(self.device)
