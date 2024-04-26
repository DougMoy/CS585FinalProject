import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--cls_n", default=8, type=int, help="num of classes")
parser.add_argument("--set_n", default=9, type=int, help="num of classes")
parser.add_argument("--transfer", action="store_false")

args = parser.parse_args()

arg_info = f"cls={args.cls_n}_set={args.set_n}_e={args.epoch}"
if args.transfer:
    arg_info += "_transfer"

# Ensure the figures directory exists
fig_dir = "figs/"
os.makedirs(fig_dir, exist_ok=True)


def draw_train_logs():

    with open(f"results/train_logs_{arg_info}.json", "r") as f:
        train_logs = json.load(f)

    # init
    epochs = []
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch, log in train_logs.items():
        epochs.append(int(epoch))
        train_losses.append(log['train_loss'])
        train_accs.append(log['train_mean_acc'])
        val_accs.append(log['test_mean_acc'])

    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(f"{fig_dir}/train_loss.png")

    # acc
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title('Discrete Angle Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(f"{fig_dir}/acc.png")


def draw_train_logs_together():
    with open(f"results/train_logs_{arg_info}.json", "r") as f:
        train_logs = json.load(f)

    # Initialize data storage
    epochs = []
    train_losses = []
    train_accs = []
    val_accs = []

    # Extract data from logs
    for epoch, log in train_logs.items():
        epochs.append(int(epoch))
        train_losses.append(log['train_loss'])
        train_accs.append(log['train_mean_acc'])
        val_accs.append(log['test_mean_acc'])

    # Setup a figure with two subplots
    plt.figure(figsize=(20, 10))  # Larger figure size to accommodate both plots

    # Plot for loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(epochs, train_losses, label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accs, label='Validation Accuracy', color='green')
    plt.title('Discrete Angle Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure containing both subplots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{fig_dir}/combined_train_log.png")
    # plt.show()  # Uncomment to display the plot interactively


def softmax(x):
    """ Compute softmax values for each set of scores in x. """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _draw_mean_shift(ax, base_degrees, logits, title):
    # ps = softmax(logits)  # Apply softmax to logits
    ps = logits

    xs = ps * np.cos(np.deg2rad(base_degrees))
    ys = ps * np.sin(np.deg2rad(base_degrees))

    if np.mean(ps) == ps[0]:
        mean_x = 1.5 * np.mean(xs)
        mean_y = 1.5 * np.mean(ys)
    else:
        mean_x = np.sum(xs)
        mean_y = np.sum(ys)

    # Draw individual vectors with fine arrowheads
    for i in range(len(xs)):
        ax.quiver(0, 0, xs[i], ys[i], scale=1, scale_units='xy', angles='xy', color='blue', width=0.0025, headwidth=6, headlength=7)

    # Draw mean vector
    ax.quiver(0, 0, mean_x, mean_y, scale=1, scale_units='xy', angles='xy', color='red', width=0.005, headwidth=8, headlength=10)

    xy = 0
    while xy < max(mean_x, mean_y):
        xy += 0.2

    ax.set_xlim(-xy, xy)
    ax.set_ylim(-xy, xy)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)


def draw_mean_shift():
    # Example usage:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots

    base_degrees1 = np.linspace(0, 360, 8, endpoint=False)

    # Logits for the first subplot: Randomly generated
    # logits1 = np.random.randn(8)
    logits1 = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.2, 0.1, 0.1])
    logits1 = softmax(logits1)

    base_degrees2 = np.random.uniform(low=60, high=100, size=8)
    # Logits for the second subplot: Identical for all angles
    constant_value = np.random.randn(1)  # Generate a single random number
    logits2 = np.full(8, constant_value)
    logits2 = softmax(logits2)

    # Call the function with specific axes and titles
    _draw_mean_shift(axes[0], base_degrees1, logits1, f"Probabilities={[round(logit, 3) for logit in logits1]}")
    _draw_mean_shift(axes[1], base_degrees2, logits2, f"Weights={[round(logit, 3) for logit in logits2]}\n(Scale changed for visualization)")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{fig_dir}/mean_shift.png")


if __name__ == '__main__':
    # draw_train_logs()
    # draw_train_logs_together()

    draw_mean_shift()
