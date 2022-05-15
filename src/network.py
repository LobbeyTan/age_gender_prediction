from torch import nn


# Classifier that received input after VGG16 feature extractor
class Classifier(nn.Module):

    def __init__(self, n_output) -> None:
        super(Classifier, self).__init__()

        # Specify the fully connected output layers
        self.layers = [
            nn.Flatten(),
            nn.Linear(32768, 4096),
            nn.Linear(4096, n_output),
            nn.Softmax(dim=1)
        ]

        # Use nn.Sequential to make layers in sequence
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # Get the prediction result
        return self.model(x)
