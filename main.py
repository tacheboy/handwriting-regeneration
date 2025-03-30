import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocess import load_data, normalize_stroke, pad_strokes
from train import train_model
from generate import generate_handwriting

def main():
    train_path = './deepwriting_training.npz'
    valid_path = './deepwriting_validation.npz'
    
    # Load and preprocess training data
    train_strokes, data_mean, data_std = load_data(train_path)
    valid_strokes, _, _ = load_data(valid_path)
    
    # Normalize 
    train_strokes_norm = [normalize_stroke(s, data_mean, data_std) for s in train_strokes]
    valid_strokes_norm = [normalize_stroke(s, data_mean, data_std) for s in valid_strokes]
    
    max_seq_length = 489
    train_strokes_padded = pad_strokes(train_strokes_norm, max_seq_length)
    valid_strokes_padded = pad_strokes(valid_strokes_norm, max_seq_length)
    
    print("Padded training strokes shape:", train_strokes_padded.shape)
    print("Padded validation strokes shape:", valid_strokes_padded.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train_model(train_strokes_padded, valid_strokes_padded, device, num_epochs=10, batch_size=64, learning_rate=0.001)
    
    # Generation: Use the first timestep of the first validation sample
    from torch.utils.data import TensorDataset
    seed = torch.tensor(valid_strokes_padded[0:1, :1, :]).to(device)  # shape (1, 1, 3)
    generated_sequence = generate_handwriting(model, seed, device, seq_length=100)
    
    # Plot the generated sequence
    plt.figure(figsize=(6, 4))
    plt.plot(generated_sequence[:, 0], generated_sequence[:, 1], marker='o')
    plt.title("Generated Handwriting Sequence")
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    main()
