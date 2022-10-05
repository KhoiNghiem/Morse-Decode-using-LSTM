import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import torch
from scipy import signal
from scipy.io.wavfile import write
from main import MorseBatchedMultiLSTM
from morse import ALPHABET, SAMPLE_FREQ, get_spectrogram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("input")
    args = parser.parse_args()

    rate, data = scipy.io.wavfile.read(args.input)

    length = len(data) / rate
    new_length = int(length * SAMPLE_FREQ)

    data = signal.resample(data, new_length)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    # Create spectrogram
    spec = get_spectrogram(data)
    spec_orig = spec.copy()
    spectrogram_size = spec.shape[0]

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MorseBatchedMultiLSTM()