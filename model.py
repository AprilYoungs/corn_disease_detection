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

class EncoderClearCut(nn.Module):
    """remove the last conv network with no pool """
    def __init__(self):
        super(EncoderClearCut, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        return features
    
class EncoderCut(nn.Module):
    """remove the last conv network"""
    def __init__(self):
        super(EncoderCut, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(14)
        
    def forward(self, images):
        features = self.resnet(images)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features

class EncoderCut2Layer(nn.Module):
    """remove the last 2 conv network"""
    def __init__(self):
        super(EncoderCut2Layer, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(28)
        
    def forward(self, images):
        features = self.resnet(images)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features
    

class FinalClassify(nn.Module):
    def __init__(self, in_features, class_size):
        super(FinalClassify, self).__init__()
        self.fc = nn.Linear(in_features, class_size)
    
    def forward(self, features):
        y = self.fc(features)
        return y
    

class MultiClassify(nn.Module):
    def __init__(self, in_features, class_size):
        super(MultiClassify, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features//3)
        self.fc2 = nn.Linear(in_features//3, class_size)
    
    def forward(self, features):
        y = self.fc1(features)
        y = self.fc2(y)
        return y
    
class ConvClassify(nn.Module):
    """Get the last conv layer of resnet to train"""
    def __init__(self, in_features, class_size):
        super(ConvClassify, self).__init__()
        resnet = models.resnet152()
        self.conv = list(resnet.children())[-3]
        self.pool = list(resnet.children())[-2]
        self.fc = nn.Linear(in_features, class_size)
    
    def forward(self, features):
        y = self.conv(features)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y