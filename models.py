import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# Feature Extractor using ResNet50
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers first
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        # Unfreeze the last 15 layers
        num_layers = len(list(self.resnet_model.children()))
        layers_to_unfreeze = list(self.resnet_model.children())[num_layers - 15:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        # Modify the ResNet model to not include the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


# Orientation Layer to predict orientation probabilities
class OrientationLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OrientationLayer, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x


# Mean Shift Layer to convert softmax probabilities to a continuous angle
class MeanShiftLayer(nn.Module):
    def __init__(self, num_classes):
        super(MeanShiftLayer, self).__init__()
        self.num_classes = num_classes

    def forward(self, softmax_probs):
        base_angles = torch.linspace(0, 360, self.num_classes, device=softmax_probs.device)
        weighted_sum = torch.sum(softmax_probs * base_angles.unsqueeze(0), dim=1)
        return weighted_sum


# Angle Predictor combining all the components
class AnglePredictor(nn.Module):
    def __init__(self, num_classes=8):
        super(AnglePredictor, self).__init__()

        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor()
        self.orientation_layer = OrientationLayer(2048,
                                                  num_classes)  # Assuming ResNet50's penultimate features are 2048-dim
        self.mean_shift_layer = MeanShiftLayer(num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        orientation_probs = self.orientation_layer(features)
        if self.training:
            return orientation_probs
        # use mean shift only for predict
        continuous_angle = self.mean_shift_layer(orientation_probs)
        return continuous_angle


def main():

    # Example usage
    model = AnglePredictor()
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
    output_angles = model(input_tensor)
    print(output_angles)


if __name__ == '__main__':
    main()

