from torch import nn


class Classifier(nn.Module):

    def __init__(self, n_output) -> None:
        super(Classifier, self).__init__()

        self.layers = [
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.Flatten(),
            nn.Linear(32768, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 64),
            nn.Linear(64, n_output),
            nn.Softmax(dim=1)
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
