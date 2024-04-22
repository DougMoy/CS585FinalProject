import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

import load_data
from models import AnglePredictor


class VehicleDataset(Dataset):
    def __init__(self, images, labels, bboxes, num_classes=8, num_sets=9, mode="train"):
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.num_classes = num_classes
        self.num_sets = num_sets
        self.mode = mode  # train val test
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        bbox = self.bboxes[idx]
        cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        image_tensor = self.transforms(cropped_image)

        # Process the label based on mode
        continuous_angle = self.labels[idx]
        if self.mode == "train" or self.mode == "val":
            class_indices = self.transform_angle_to_classes(continuous_angle)
            # print(continuous_angle, class_indices)
            return image_tensor, class_indices
        else:
            assert self.mode == "test"
            return image_tensor, torch.tensor([continuous_angle])

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


def get_data_loaders(train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes, batch_size=4):
    train_dataset = VehicleDataset(train_images, train_labels, train_bboxes, mode="train")
    val_dataset = VehicleDataset(test_images, test_labels, test_bboxes, mode="val")
    test_dataset = VehicleDataset(test_images, test_labels, test_bboxes, mode="test")
    _test_train_dataset = VehicleDataset(train_images, train_labels, train_bboxes, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    _test_train_loader = DataLoader(_test_train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, _test_train_loader


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.squeeze()

            outputs_discrete = model(inputs)
            outputs = model.predict(outputs_discrete)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f"Test MAE Loss: {average_loss}")
    return average_loss


def calculate_accuracy(model, test_loader, device):
    """
    Calculate the accuracy for each start angle in a model that outputs multiple sets of class predictions.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device tensors are on, e.g., 'cuda' or 'cpu'.

    Returns:
        list: List of accuracies for each start angle.
    """
    model.eval()  # Ensure the model is in evaluation mode
    total_correct_preds = torch.zeros(test_loader.dataset.num_sets, device=device)
    total_samples = torch.zeros(test_loader.dataset.num_sets, device=device)

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Calculate predictions and correctness for all start angles at once
            predictions = torch.argmax(outputs, dim=2)
            correct = predictions.eq(labels).type(torch.long)

            # print(predictions, labels)

            # Sum correct predictions and total samples for each start angle
            total_correct_preds += correct.sum(dim=0)
            total_samples += labels.size(0) * torch.ones(test_loader.dataset.num_sets, device=device)

    # Compute average accuracy for each start angle
    accuracies = (total_correct_preds / total_samples).tolist()
    print("Accuracies per start angle:", np.mean(accuracies), accuracies)

    return accuracies


def circular_variance_loss(outputs, labels, num_classes, var_weight=0.1):
    """
    Custom loss function that penalizes predictions deviating significantly from the circular mean.
    This function does not apply softmax as it assumes outputs are already probabilities or treated accordingly.

    Args:
        outputs (torch.Tensor): The model outputs; expected shape [batch_size, num_classes].
        labels (torch.Tensor): The actual labels; shape [batch_size].
        num_classes (int): The total number of classes, used for circular calculation.
        var_weight (float): Weight of the variance penalty relative to the main classification loss.

    Returns:
        torch.Tensor: Total loss combining CrossEntropy and circular variance penalty.
    """
    # Main classification loss
    classification_loss = F.cross_entropy(outputs, labels)

    # Predicted class values
    predicted_classes = torch.argmax(outputs, dim=1)

    # Calculate mean predicted class, considering circular nature
    sin_components = torch.sin(2 * torch.pi * predicted_classes / num_classes)
    cos_components = torch.cos(2 * torch.pi * predicted_classes / num_classes)
    mean_angle_sin = torch.mean(sin_components)
    mean_angle_cos = torch.mean(cos_components)
    mean_class = torch.atan2(mean_angle_sin, mean_angle_cos) * num_classes / (2 * torch.pi)
    mean_class = torch.remainder(mean_class, num_classes)  # Ensure within [0, num_classes)

    # Calculate the circular differences from the mean
    differences = torch.abs(predicted_classes - mean_class)
    differences = torch.minimum(differences, num_classes - differences)  # Handle circular nature

    # Variance penalty for deviations greater than 1
    variance_penalty = torch.mean((differences > 1).float())

    # Combine losses
    total_loss = classification_loss + var_weight * variance_penalty
    return total_loss


def train_model(device, model, train_loader, val_loader, num_epochs=30, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    # criterion = circular_variance_loss

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # print(outputs.shape)
            # print(labels.shape)

            # Process each set of outputs and labels
            loss = 0
            for i in range(outputs.shape[1]):  # Iterate over each set
                loss += criterion(outputs[:, i, :], labels[:, i])

            loss = loss / outputs.shape[1]  # Average loss over all sets

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}")

        # scheduler.step()

        calculate_accuracy(model, train_loader, device)

        calculate_accuracy(model, val_loader, device)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", default=4, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--cls_n", default=8, type=int, help="num of classes")
    parser.add_argument("--set_n", default=9, type=int, help="num of classes")

    args = parser.parse_args()
    print(args)

    train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes = load_data.load_data()

    train_loader, val_loader, test_loader, _test_train_loader = get_data_loaders(
        train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes,
        args.bz,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AnglePredictor(num_classes=args.cls_n)
    model = train_model(device, model, train_loader, val_loader, num_epochs=args.epoch)

    evaluate(model, _test_train_loader, device)
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
