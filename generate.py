import torch
import numpy as np

def generate_handwriting(model, seed, device, seq_length=200):
    """
    Generates a sequence of strokes given an initial seed.
    - model: trained HandwritingLSTM model
    - seed: initial stroke (tensor of shape (1, 1, input_size))
    - seq_length: number of additional timesteps to generate
    Returns the generated stroke sequence as a NumPy array.
    """
    model.eval()
    generated = [seed.squeeze(0).cpu().numpy()]
    input_seq = seed.to(device)
    hidden = None
    for _ in range(seq_length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
        # take the prediction of the last time-step
        next_step = output[:, -1:, :]
        generated.append(next_step.squeeze(0).cpu().numpy())
        # prepare next input by removing the first time-step and appending the new prediction
        input_seq = torch.cat([input_seq[:, 1:, :], next_step], dim=1)
    return np.array(generated)
