import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils

class FashionSimpleNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 7 * 7)
        x = self.classifier(x)
        return F.log_softmax(x)

if __name__ == '__main__':
    net = FashionSimpleNet()
    size = utils.calculate_feature_size(net.features,(28,28))
    print(size)
