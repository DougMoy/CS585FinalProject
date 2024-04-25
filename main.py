import argparse
import json
import os

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
        class_indices = self.transform_angle_to_classes(continuous_angle)

        return image_tensor, torch.tensor([continuous_angle]), class_indices

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
        # image, angle, logits
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.squeeze()

            logits, angles = model(inputs)

            loss = criterion(angles, labels.float())
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

    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, indices = inputs.to(device), indices.to(device)

            logits, angles = model(inputs)

            # Calculate predictions and correctness for all start angles at once
            predictions = torch.argmax(logits, dim=2)
            correct = predictions.eq(indices).type(torch.long)

            # print(predictions, indices)

            # Sum correct predictions and total samples for each start angle
            total_correct_preds += correct.sum(dim=0)
            total_samples += indices.size(0) * torch.ones(test_loader.dataset.num_sets, device=device)

    # Compute average accuracy for each start angle
    accuracies = (total_correct_preds / total_samples).tolist()
    mean_acc = float(np.mean(accuracies))
    print("Accuracies per start angle:", mean_acc, accuracies)

    return mean_acc, accuracies


def train_model(device, model, train_loader, val_loader, num_epochs=30, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.1)

    criterion_logits = nn.CrossEntropyLoss()
    criterion_angles = nn.L1Loss()

    logs = {}

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels, indices in train_loader:
            inputs, labels, indices = inputs.to(device), labels.to(device), indices.to(device)

            optimizer.zero_grad()
            logits, angles = model(inputs)

            # print(outputs.shape)
            # print(labels.shape)

            # Process each set of outputs and labels
            loss_logits = 0
            for i in range(logits.shape[1]):  # Iterate over each set
                loss_logits += criterion_logits(logits[:, i, :], indices[:, i])
            loss_logits = loss_logits / logits.shape[1]  # Average loss over all sets

            loss_angles = criterion_angles(angles, labels)

            loss = loss_logits
            # loss = loss_logits + 0.001 * loss_angles

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")

        # scheduler.step()

        train_mean_acc, train_acc = calculate_accuracy(model, train_loader, device)

        test_mean_acc, test_acc = calculate_accuracy(model, val_loader, device)

        logs[epoch] = dict(
            train_loss=train_loss,
            train_mean_acc=train_mean_acc,
            test_mean_acc=test_mean_acc,
        )

    return model, logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", default=4, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--cls_n", default=8, type=int, help="num of classes")
    parser.add_argument("--set_n", default=9, type=int, help="num of classes")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--transfer", action="store_true")

    args = parser.parse_args()
    print(args)

    train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes = load_data.load_data()
    print(len(train_images), len(test_images))

    if args.transfer:
        import random
        sample_n = int(0.05 * len(test_labels))
        sample_indices = random.sample(range(len(test_labels)), sample_n)

        sampled_images = [test_images[i] for i in sample_indices]
        sampled_labels = [test_labels[i] for i in sample_indices]
        sampled_bboxes = [test_bboxes[i] for i in sample_indices]

        train_images = np.concatenate((train_images, sampled_images), axis=0)
        train_labels = np.concatenate((train_labels, sampled_labels), axis=0)
        train_bboxes = np.concatenate((train_bboxes, sampled_bboxes), axis=0)

    print(len(train_images), len(test_images))

    train_loader, val_loader, test_loader, _test_train_loader = get_data_loaders(
        train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes,
        args.bz,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AnglePredictor(num_classes=args.cls_n)

    arg_info = f"cls={args.cls_n}_set={args.set_n}_e={args.epoch}"
    if args.transfer:
        arg_info += "_transfer"

    model_save_path = f'models/checkpoint_discrete_{arg_info}.pth'

    if args.eval:
        model.load_state_dict(torch.load(model_save_path), strict=False)
        model.to(device)
        evaluate(model, _test_train_loader, device)
        evaluate(model, test_loader, device)
        return

    model, train_logs = train_model(device, model, train_loader, val_loader, num_epochs=args.epoch)

    os.makedirs("results", exist_ok=True)
    with open(f"results/train_logs_{arg_info}.json", "w") as f:
        json.dump(train_logs, f, indent=4)

    evaluate(model, _test_train_loader, device)
    evaluate(model, test_loader, device)

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
