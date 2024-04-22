import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from scipy.stats import norm


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

        print(self.registered_angles)

    def transform_angle_to_classes(self, angle):
        """
              Transforms a continuous angle into multiple sets of discrete class indices.

              Args:
                  angle (float): The original angle in degrees.

              Returns:
                  torch.Tensor: A tensor containing class indices for each set.
        """
        offset_step = 360 / self.num_classes / self.num_sets
        offsets = torch.arange(0, 360, offset_step, dtype=torch.float32)[:self.num_sets]
        adjusted_angles = (angle + offsets) % 360
        class_indices = torch.floor(adjusted_angles / (360 / self.num_classes))

        return class_indices.long()

    def forward(self, logits):
        # Convert logits to probabilities across each class for each set of logits
        # logits shape is expected to be [batch, num_sets, num_classes]
        # probabilities = F.softmax(logits, dim=-1)  # Applying softmax on the class dimension
        #
        # print(probabilities)

        probabilities = logits

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

        return mean_angle_degrees % 360  # Shape: [batch]


def generate_gaussian_logits(angle, num_classes, std_dev=0.1):
    # num_classes: 总的类别数
    # std_dev: 高斯分布的标准差

    # 假设 mean_shift_layer 是已经定义好的层，且有 transform_angle_to_classes 方法
    class_indices = mean_shift_layer.transform_angle_to_classes(angle)

    # 创建一个 logits tensor，形状为 [num_sets, num_classes]
    logits = torch.zeros((len(class_indices), num_classes), dtype=torch.float32)

    # 对每个 set 生成高斯分布的 logits
    x = torch.arange(num_classes).float()  # [0, 1, ..., num_classes-1]
    for i, index in enumerate(class_indices):
        # 为每个 set 的中心索引创建一个高斯分布
        # 高斯函数：exp(-((x - mu)^2 / (2 * sigma^2)))
        logits[i] = torch.exp(-0.5 * ((x - index) ** 2) / (std_dev ** 2))

    return logits


# Assuming the MeanShiftLayer with added method is defined
mean_shift_layer = MeanShiftLayer(num_classes=8, num_sets=9)

angles = np.random.random(10) * 360

for angle in angles:
    print("====")
    print(angle)

    # Generate logits from the class indices
    logits = generate_gaussian_logits(angle, mean_shift_layer.num_classes, 0.1)
    predicted_angle = mean_shift_layer(logits.unsqueeze(0))  # Process logits through the mean shift layer

    print(predicted_angle)  # Display the predicted angle
