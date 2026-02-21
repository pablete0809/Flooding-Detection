import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import FloodTimeSeriesDataset
from src.model import SimpleUNet

def train():
    # Configuration
    DATASET_DIR = 'dataset_sen12flood_v1'
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Parameters
    HISTORY_LENGTH = 3
    FORECAST_HORIZON = 5

    # Input Channels: (S2[8] + S1[3]) * history + Terrain[5]
    # = (8 + 3) * 3 + 5 = 38
    IN_CHANNELS = (8 + 3) * HISTORY_LENGTH + 5
    OUT_CHANNELS = FORECAST_HORIZON
    
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Dataset
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please run the updated main.ipynb to generate the data first.")
        return

    dataset = FloodTimeSeriesDataset(DATASET_DIR, history_length=HISTORY_LENGTH, forecast_horizon=FORECAST_HORIZON)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. No valid time-series samples found.")
        print("Please ensure you have downloaded enough data covering multiple dates for the same tiles.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Prepare Model
    model = SimpleUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 4. Save Model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), 'checkpoints/flood_forecast_model.pth')
    print("Model saved to checkpoints/flood_forecast_model.pth")

if __name__ == '__main__':
    train()
