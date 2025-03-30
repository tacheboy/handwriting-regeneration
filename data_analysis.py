import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

def analyze_npz(file_path, dataset_name="Dataset"):
    # Load the dataset
    data = np.load(file_path, allow_pickle=True)
    print(f"\n=== Analysis for {dataset_name} ===")
    print("Keys in the dataset:", data.files)
    
    # basic eda
    for key in data.files:
        array = data[key]
        print(f"\nKey: {key}")
        print("  Type:", type(array))
        if isinstance(array, np.ndarray):
            print("  Shape:", array.shape)
            print("  Dtype:", array.dtype)
        else:
            print("  Value:", array)
    
    return data

train_file = './deepwriting_training.npz'
valid_file = './deepwriting_validation.npz'

train_data = analyze_npz(train_file, "Training Data")
valid_data = analyze_npz(valid_file, "Validation Data")

train_strokes = train_data['strokes']
valid_strokes = valid_data['strokes']

# Get normalization statistics
data_min = train_data['min']
data_max = train_data['max']
data_mean = train_data['mean']
data_std = train_data['std']

print("\nNormalization statistics (from training data):")
print("  Min:", data_min)
print("  Max:", data_max)
print("  Mean:", data_mean)
print("  Std:", data_std)

def normalize_strokes(stroke_seq, mean, std):
    return (stroke_seq - mean) / std

# Visualize a few normalized stroke samples from training data
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
for i in range(num_samples):
    stroke_sample = train_strokes[i]
    normalized_stroke = normalize_strokes(stroke_sample, data_mean, data_std)
    
    # Plot assuming the first two features are x and y coordinates
    axes[i].plot(normalized_stroke[:, 0], normalized_stroke[:, 1], marker='o')
    axes[i].set_title(f"Train Sample {i}")
    axes[i].invert_yaxis()  # Invert y-axis for correct handwriting orientation
plt.suptitle("Normalized Stroke Samples - Training Data")
plt.show()

#visualisation
num_samples_val = 5
fig, axes = plt.subplots(1, num_samples_val, figsize=(15, 3))
for i in range(num_samples_val):
    stroke_sample = valid_strokes[i]
    normalized_stroke = normalize_strokes(stroke_sample, data_mean, data_std)
    
    axes[i].plot(normalized_stroke[:, 0], normalized_stroke[:, 1], marker='o')
    axes[i].set_title(f"Valid Sample {i}")
    axes[i].invert_yaxis()
plt.suptitle("Normalized Stroke Samples - Validation Data")
plt.show()

max_seq_length = max(len(seq) for seq in train_strokes)
print("Maximum sequence length in training data:", max_seq_length)

#padded seq
train_strokes_padded = pad_sequences(train_strokes, maxlen=max_seq_length, padding='post', dtype='float32')
valid_strokes_padded = pad_sequences(valid_strokes, maxlen=max_seq_length, padding='post', dtype='float32')
print("Padded training strokes shape:", train_strokes_padded.shape)
print("Padded validation strokes shape:", valid_strokes_padded.shape)
