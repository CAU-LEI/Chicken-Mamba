import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress

# Import model from network1.py
from network1 import ConvGRURegression

# Dataset definition
class SliceDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.samples = self._group_samples()

    def _group_samples(self):
        file_paths = glob.glob(os.path.join(self.image_dir, "*.png"))
        grouped = {}
        for path in file_paths:
            file_name = os.path.basename(path)
            sample_id = file_name.split('_')[0]
            if sample_id in self.labels:
                if sample_id not in grouped:
                    grouped[sample_id] = []
                grouped[sample_id].append(path)
        for key in grouped:
            grouped[key] = sorted(grouped[key])
        return [(k, v) for k, v in grouped.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, image_paths = self.samples[idx]
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        label = torch.tensor(self.labels[sample_id], dtype=torch.float32)
        return images, label, sample_id


# Split training and test sets
def train_test_split(labels, test_size=0.2):
    sample_ids = list(labels.keys())
    torch.manual_seed(42)
    shuffled_ids = torch.randperm(len(sample_ids)).tolist()
    split_idx = int(len(sample_ids) * (1 - test_size))
    train_ids = [sample_ids[i] for i in shuffled_ids[:split_idx]]
    test_ids = [sample_ids[i] for i in shuffled_ids[split_idx:]]
    train_labels = {k: labels[k] for k in train_ids}
    test_labels = {k: labels[k] for k in test_ids}
    return train_labels, test_labels


# Train model
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return losses


# Test model
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for images, labels, sample_ids in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_ids.extend(sample_ids)
    return all_labels, all_preds, all_ids


# Main program
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_file_path = "label1.xlsx"
    label_data = pd.read_excel(label_file_path)
    label_data['ids'] = label_data['ids'].apply(lambda x: f"{x:04d}")
    labels = {row['ids']: row['left'] for _, row in label_data.iterrows()}

    # Model parameters
    image_dir = "SegmentationClass"
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    batch_size = 16
    hidden_dim = [64, 128]
    kernel_size = (3, 3)
    num_layers = 2
    input_size = (64, 64)
    input_dim = 3
    output_dim = 1

    # Data loading
    train_labels, test_labels = train_test_split(labels, test_size=0.2)
    train_dataset = SliceDataset(image_dir, train_labels, transform=transform)
    test_dataset = SliceDataset(image_dir, test_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = ConvGRURegression(input_size, input_dim, hidden_dim, kernel_size, num_layers, output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train model
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)

    # Test model
    actual_values, predicted_values, sample_ids = evaluate_model(model, test_loader, device=device)

    # Save .npz file
    np.savez("results.npz", losses=losses, actual=actual_values, predicted=predicted_values)

    # Save Excel table
    results_df = pd.DataFrame({
        'ID': sample_ids,
        'Actual': actual_values,
        'Predicted': predicted_values
    })
    results_df.to_excel("predictions_vs_actual.xlsx", index=False)
    print("Prediction results saved to predictions_vs_actual.xlsx")

    # Evaluation metrics
    r2 = r2_score(actual_values, predicted_values)
    slope, intercept, r_value, p_value, std_err = linregress(actual_values, predicted_values)
    print(f"R²: {r2:.4f}, Slope: {slope:.4f}, Intercept: {intercept:.4f}, p-value: {p_value:.4g}")

    # Plot prediction vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_values, predicted_values, alpha=0.7)
    plt.plot([min(actual_values), max(actual_values)],
             [min(actual_values), max(actual_values)],
             'r--', label='Ideal')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Prediction vs Actual (R² = {r2:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")
    plt.close()
    print("Scatter plot saved as prediction_vs_actual.png")

    # Plot training loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_curve.png")
    plt.close()
    print("Training loss curve saved as training_loss_curve.png")
