import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
data = np.load('D:/Python and ML/CRWU bearing/CWRU_48k_load_1_CNN_data.npz')
X_raw = data['data']
y_raw = data['labels']

print(f"Original shape: {X_raw.shape}")

# Convert labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Convert to tensor
X = torch.tensor(X_raw, dtype=torch.float32)
y = torch.tensor(y_encoded, dtype=torch.long)

# Fix shape
print(f"X shape before fix: {X.shape}")

if X.ndim == 3:
    X = X.unsqueeze(1)
elif X.ndim == 4 and X.shape[-1] == 1:
    X = X.squeeze(-1).unsqueeze(1)
elif X.ndim == 4 and X.shape[1] != 1 and X.shape[1] != 32:
    X = X.permute(0, 3, 1, 2)

print(f"Final X shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# CNN
class cnn(nn.Module):
    def __init__(self, in_channels):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = cnn(in_channels=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}')