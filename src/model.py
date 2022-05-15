import os
from torch import nn
import torch
import torchvision.transforms as transforms
from src.network import Classifier
from src.vgg16 import load_vgg16, vgg_preprocess


# The Age and Gender Prediction model that extends torch.nn.Module class
class AgeGenderPredictor(nn.Module):

    def __init__(self, lr=0.001, beta1=0.5, device=torch.device("cpu"), vgg_path="./src/pretrained/vgg16.weight") -> None:

        super(AgeGenderPredictor, self).__init__()

        # Store the torch.device to be applied
        self.device = device

        # Create a vgg16 with pretrained weight loaded
        self.vgg = load_vgg16(vgg_path, self.device)

        # Create age classifier with 8 outputs
        self.age_model = Classifier(n_output=8).to(self.device)

        # Create gender classifier with 2 outputs
        self.gender_model = Classifier(n_output=2).to(self.device)

        # Use Cross Entropy Loss as criterion for classification problem
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Use Adam optimizer for weights update
        self.age_optimizer = torch.optim.Adam(
            self.age_model.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        self.gender_optimizer = torch.optim.Adam(
            self.gender_model.parameters(), lr=lr, betas=(beta1, 0.999)
        )

        # Specify the transform process for input images
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, x):
        # Call to make age and gender prediction
        self.image_input = vgg_preprocess(x['img']).to(self.device)
        self.age_real = x['age'].to(self.device)
        self.gender_real = x['gender'].to(self.device)

        # Predict age using age classifier
        self.age_pred = self.age_model(self.vgg(self.image_input))
        # Predict gender using gender classifier
        self.gender_pred = self.gender_model(self.vgg(self.image_input))

        # Use argmax to obtained the index of class with highest probability
        return torch.argmax(self.age_pred, dim=1), torch.argmax(self.gender_pred, dim=1)

    def optimize_parameters(self):
        # Reset the gradient of optimizer to 0
        self.age_optimizer.zero_grad()
        # Use the criterion to calculate total loss
        self.age_loss = self.criterion(self.age_pred, self.age_real)
        # Call backward for autograd
        self.age_loss.backward()
        # Update the weights of all related variables using backpropagation
        self.age_optimizer.step()

        # Reset the gradient of optimizer to 0
        self.gender_optimizer.zero_grad()
        # Use the criterion to calculate total loss
        self.gender_loss = self.criterion(self.gender_pred, self.gender_real)
        # Call backward for autograd
        self.gender_loss.backward()
        # Update the weights of all related variables using backpropagation
        self.gender_optimizer.step()

    def eval(self):
        # Set the classifiers to evaluation mode
        self.age_model.eval()
        self.gender_model.eval()

    def test(self, x):
        # Test function use to test prediction result on test dataset

        # Specify that we do not need to use gradient for inferencing
        with torch.no_grad():
            # Get the predicted output
            age, gender = self.forward(x)

        return age, gender

    def predict(self, img):
        # Predict function use to predict age and gender of an input image

        # First preprocess and transform the input image
        img = self.transform(img)
        # Then unsqueeze the image dimension to 4D e.g [1, 3, 224, 224]
        img = torch.unsqueeze(img, dim=0)
        # Preprocess image before input into VGG16 model
        img = vgg_preprocess(img)

        # Gradient is not needed for inferencing (faster prediction)
        with torch.no_grad():
            # Get the age and gender prediction
            age = torch.argmax(self.age_model(self.vgg(img)))
            gender = torch.argmax(self.gender_model(self.vgg(img)))

        return age, gender

    def save_model(self, directory, epoch):
        # Save the model into specific directory
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
        # Load the model from specific directory
        self._load_model(self.age_model, "age", directory)
        self._load_model(self.gender_model, "gender", directory)

    def _load_model(self, network: torch.nn.Module, name, directory):
        filename = "net_%s.pth" % name
        path = os.path.join(directory, filename)

        network.load_state_dict(torch.load(path, map_location=self.device))

        network.to(self.device)
