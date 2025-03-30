import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HandwritingDataset
from model import HandwritingLSTM

def train_model(train_data, valid_data, device, num_epochs=10, batch_size=64, learning_rate=0.001):
    """
    Trains the HandwritingLSTM model def in model.py file on the provided training data and validates
    Returns the trained model.
    """
    # datasets and loaders
    train_dataset = HandwritingDataset(train_data)
    valid_dataset = HandwritingDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # hyperparams
    model = HandwritingLSTM(input_size=3, hidden_size=256, num_layers=3, output_size=3, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_inputs.size(0)
        
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in valid_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                outputs, _ = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                valid_loss += loss.item() * batch_inputs.size(0)
        avg_valid_loss = valid_loss / len(valid_dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    
    return model
