import torch
import os

from utils import load_data, build_vocab, encode_text
from baseline_model import BaselineModel
from improved_model import ImprovedModel


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# PATH
if os.path.exists("/kaggle/input"):
    BASE_PATH = "/kaggle/input/datasets/vya0651/tweetnlp-project-kavya/TweetNLP/dataset"
else:
    BASE_PATH = "dataset"


# Load data
train_texts, train_labels = load_data(
    os.path.join(BASE_PATH, "train_text.txt"),
    os.path.join(BASE_PATH, "train_labels.txt")
)

val_texts, val_labels = load_data(
    os.path.join(BASE_PATH, "val_text.txt"),
    os.path.join(BASE_PATH, "val_labels.txt")
)


# Build vocab
vocab = build_vocab(train_texts)


# Encode
val_encoded = [encode_text(t, vocab) for t in val_texts]


# Padding
def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return padded


max_len = 30
val_padded = pad_sequences(val_encoded, max_len)


# Tensor
X_val = torch.tensor(val_padded).to(device)
y_val = torch.tensor(val_labels).to(device)


# MODEL SELECTION
MODEL_TYPE = "improved"  # "baseline" or "improved"

if MODEL_TYPE == "baseline":
    model = BaselineModel(len(vocab)+1, 100, 128, 3)
    load_path = "baseline_model.pth"
else:
    model = ImprovedModel(len(vocab)+1, 100, 128, 3)
    load_path = "improved_model.pth"


model.load_state_dict(torch.load(load_path))
model = model.to(device)


# Evaluation
model.eval()

with torch.no_grad():
    outputs = model(X_val)
    _, predicted = torch.max(outputs, 1)


# Metrics
num_classes = 3
precision, recall, f1 = [], [], []

for cls in range(num_classes):

    tp = ((predicted == cls) & (y_val == cls)).sum().item()
    fp = ((predicted == cls) & (y_val != cls)).sum().item()
    fn = ((predicted != cls) & (y_val == cls)).sum().item()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

    precision.append(p)
    recall.append(r)
    f1.append(f)


print("\n===== METRICS =====")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Average F1:", sum(f1) / num_classes)


# CONFUSION MATRIX (with labels)

conf_matrix = [[0]*num_classes for _ in range(num_classes)]

for t, p in zip(y_val, predicted):
    conf_matrix[t.item()][p.item()] += 1

labels = ["Negative", "Neutral", "Positive"]

print("\n===== CONFUSION MATRIX =====")
print("Rows = Actual, Columns = Predicted\n")

# Header
print(f"{'':<10} {'Pred_Neg':<10} {'Pred_Neu':<10} {'Pred_Pos':<10}")

# Rows
for i, row in enumerate(conf_matrix):
    print(f"{labels[i]:<10} {row[0]:<10} {row[1]:<10} {row[2]:<10}")