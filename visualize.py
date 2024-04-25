import argparse
import json
import os

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

fig_dir = "figs/"
os.makedirs(fig_dir, exist_ok=True)

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
