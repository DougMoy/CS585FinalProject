import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


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

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet_model.children())[:-1])

    def forward(self, x):
        return self.features(x)


# class MultiHeadOrientationLayer(nn.Module):
#     def __init__(self, input_dim, num_classes, num_heads):
#         super(MultiHeadOrientationLayer, self).__init__()
#         self.heads = nn.ModuleList([
#             nn.Linear(input_dim, num_classes) for _ in range(num_heads)
#         ])
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         outputs = [head(x) for head in self.heads]
#         return torch.stack(outputs, dim=1)  # shape: [batch, num_heads, num_classes]


class ObjectOrientLayer(nn.Module):
    def __init__(self, input_dim, num_classes, num_sets, dropout_prob=0.3):
        super(ObjectOrientLayer, self).__init__()
        # fc
        self.num_classes = num_classes
        self.num_sets = num_sets
        self.fc_layer = nn.Linear(input_dim, num_classes * num_sets)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        output = self.fc_layer(x)

        # output = self.dropout(output)
        # print(output.shape)

        output = output.view(-1, self.num_sets, self.num_classes)

        output = F.softmax(output, dim=-1)

        return output


class MeanShiftLayer(nn.Module):
    def __init__(self, num_classes, num_sets):
        super(MeanShiftLayer, self).__init__()
        self.num_classes = num_classes
        self.num_sets = num_sets
        # Create angles for each class, assuming equal distribution over 360 degrees
        base_angles = torch.arange(0, 360, 360 / num_classes)  # e.g., if num_classes=8 -> [0, 45, ..., 315]
        # Adjust each set's angles by its offset
        self.angles = torch.stack([
            (base_angles + i * 360 / num_classes / num_sets) % 360
            for i in range(num_sets)
        ])  # Shape: [num_sets, num_classes]
        self.register_buffer('registered_angles', self.angles)  # Registering angles as a constant buffer

    def forward(self, logits):
        # Convert logits to probabilities across each class for each set of logits
        # logits shape is expected to be [batch, num_sets, num_classes]
        probabilities = F.softmax(logits, dim=-1)  # Applying softmax on the class dimension

        # Convert angles to radians and compute the x and y components
        radians = torch.deg2rad(self.registered_angles)
        x_components = torch.cos(radians)
        y_components = torch.sin(radians)

        # Weight probabilities by the x and y components of their corresponding angles
        x_weighted = probabilities * x_components[None, :, :]  # Shape: [batch, num_sets, num_classes]
        y_weighted = probabilities * y_components[None, :, :]  # Shape: [batch, num_sets, num_classes]

        # Sum the components across all classes to get the vector sum for each set
        resultant_x = torch.sum(x_weighted, dim=-1)  # Shape: [batch, num_sets]
        resultant_y = torch.sum(y_weighted, dim=-1)  # Shape: [batch, num_sets]

        # Calculate the angle of the resultant vector using atan2
        resultant_angles = torch.atan2(resultant_y, resultant_x)  # Shape: [batch, num_sets]
        mean_angle = torch.mean(resultant_angles, dim=-1)  # Take mean across sets for final angle
        mean_angle_degrees = torch.rad2deg(mean_angle)  # Convert back to degrees

        # strange -20

        return mean_angle_degrees % 360  # Shape: [batch]


class AnglePredictor(nn.Module):
    def __init__(self, input_dim=2048, num_classes=8, num_heads=9):
        super(AnglePredictor, self).__init__()
        self.feature_extractor = FeatureExtractor()
        # self.orientation_layer = MultiHeadOrientationLayer(input_dim, num_classes, num_heads)
        self.orientation_layer = ObjectOrientLayer(input_dim, num_classes, num_heads)
        self.mean_shift_layer = MeanShiftLayer(num_classes, num_heads)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.orientation_layer(features)
        return logits

    def predict(self, logits):
        return self.mean_shift_layer(logits)

