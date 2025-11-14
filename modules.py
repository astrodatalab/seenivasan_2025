import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {name}")

class ResNet18PlusClassifier(nn.Module):
    """
    ResNet18 Model with Classifier/Regressor built from input hidden_dims parameter 
    specifying number of neurons per linear layer and activation.
    """
    def __init__(self, hidden_dims=[256, 128, 64, 32], 
                 dropout_rate=0, activation='relu'):
        super(ResNet18PlusClassifier, self).__init__()
        # Load the ResNet18 model without pretrained weights
        model = models.resnet18(weights=None)
        # Modify the first convolutional layer (to accept 5 input channels)
        model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Keep the original layers up to the last fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])  # Everything except the last fc layer
        
        # Build dynamic classifier based on hidden_dims
        layers = []
        input_dim = 512  # ResNet18 feature dimension 
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x)  # Pass through the original ResNet feature layers
        x = x.view(x.size(0), -1)  # Flatten the output from the ResNet feature extractor
        x = self.classifier(x)
        return x
