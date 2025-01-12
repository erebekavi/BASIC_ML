"""
PROBLEM STATEMENT
------------------------------------------------------------------
HANDWRITTEN TEXT GENERATION
------------------------------------------------------------------
Implement a character-level recurrent neural network (RNN) to
generate handwritten-like text. Train the model on a dataset of
handwritten text examples, and let it generate new text based on
the learned patterns.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(data):
    data['x'] = data['x'] / max(abs(data['x']))
    data['y'] = data['y'] / max(abs(data['y']))
    return data

def generate_sequences(data, seq_length=100):
    inputs, targets = [], []
    for i in range(0, len(data) - seq_length):
        seq_input = data[i:i + seq_length]
        seq_output = data[i + 1:i + seq_length + 1]
        inputs.append(seq_input)
        targets.append(seq_output)
    return np.array(inputs), np.array(targets)

def create_model(input_dim, output_dim):
    inputs = layers.Input(shape=(None, input_dim))
    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.LSTM(256, return_sequences=True)(x)
    delta_outputs = layers.Dense(2)(x)
    pen_state_outputs = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs, [delta_outputs, pen_state_outputs])
    model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'], loss_weights=[1.0, 0.2])
    return model

def generate_handwriting(model, start_seq, num_steps=200):
    generated = start_seq
    for _ in range(num_steps):
        delta, pen_state = model.predict(np.expand_dims(generated, axis=0), verbose=0)
        pen_state_argmax = np.argmax(pen_state[:, -1, :], axis=1)
        pen_state_argmax = np.expand_dims(pen_state_argmax, axis=-1)
        next_step = np.hstack([delta[:, -1, :], pen_state_argmax])
        generated = np.vstack([generated, next_step])
    return generated

def plot_handwriting(strokes):
    x, y = 0, 0
    x_vals, y_vals = [x], [y]
    for stroke in strokes:
        dx, dy, pen_state = stroke
        x += dx
        y += dy
        if pen_state == 1:
            x_vals.append(x)
            y_vals.append(y)
        else:
            plt.plot(x_vals, y_vals, 'k-', lw=2)
            x_vals, y_vals = [x], [y]
    plt.plot(x_vals, y_vals, 'k-', lw=2)
    plt.axis('off')
    plt.savefig("result.png")
    print(f"Handwriting plot saved to result.png")

if __name__ == "__main__":
    data = {
        'x': np.random.uniform(-10, 10, 1000),
        'y': np.random.uniform(-10, 10, 1000),
        'pen_state': np.random.choice([0, 1, 2], 1000)
    }
    
    processed_data = preprocess_data(data)
    combined_data = np.stack([processed_data['x'], processed_data['y'], processed_data['pen_state']], axis=1)
    
    seq_length = 100
    inputs, targets = generate_sequences(combined_data, seq_length)
    delta_targets = targets[:, :, :2]
    pen_state_targets = tf.keras.utils.to_categorical(targets[:, :, 2], num_classes=3)

    model = create_model(input_dim=3, output_dim=3)
    model.fit(inputs, [delta_targets, pen_state_targets], epochs=100, batch_size=64)
    
    start_seq = np.zeros((seq_length, 3))
    generated_strokes = generate_handwriting(model, start_seq)
    plot_handwriting(generated_strokes)
