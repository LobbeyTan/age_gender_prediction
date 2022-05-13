from torch import nn


class Classifier(nn.Module):

    def __init__(self, n_output) -> None:
        super(Classifier, self).__init__()

        self.layers = [
            nn.Flatten(),
            nn.Linear(131072, n_output),
            nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
