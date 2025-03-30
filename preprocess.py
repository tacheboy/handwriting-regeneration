import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(npz_path):
    """
    loads the npz file and returns the strokes along with normalization statistics.
    """
    data = np.load(npz_path, allow_pickle=True)
    strokes = data['strokes']
    data_mean = data['mean']
    data_std = data['std']
    # Ensure strokes are float32 arrays
    strokes = [np.array(s, dtype='float32') for s in strokes]
    return strokes, data_mean, data_std

def normalize_stroke(stroke, mean, std):
    return (stroke - mean) / std

def pad_strokes(strokes, max_seq_length):
    """
    Pads a list of stroke sequences
    """
    return pad_sequences(strokes, maxlen=max_seq_length, padding='post', dtype='float32')
