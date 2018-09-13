import torch.nn as nn

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
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features