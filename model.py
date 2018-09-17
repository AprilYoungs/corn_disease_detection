import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features
    

class FinalClassify(nn.Module):
    def __init__(self, in_features, class_size):
        super(FinalClassify, self).__init__()
        self.fc = nn.Linear(in_features, class_size)
    
    def forward(self, features):
        y = self.fc(features)
        return features