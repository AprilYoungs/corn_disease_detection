import torch.nn as nn
import torchvision.models as models

class ClassifiCNN(nn.Module):
    def __init__(self, class_size):
        super(ClassifiCNN, self).__init__()
        densenet = models.densenet161(pretrained=True)
        for param in densenet.parameters():
            param.requires_grad_(False)
        
        modules = list(densenet.children())[:-1]
        self.densenet = nn.Sequential(*modules)
        self.fc = nn.Linear(densenet.classifier.in_features, class_size)
    
    def forward(self, images):
        features = self.densenet(images)
        # the output size is x * 2208 * 7 * 7
        features =  nn.AvgPool2d(7)(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features