import matplotlib.pyplot as plt
import os
import time
import win32api
import numpy as np
import librosa
from typing import Any
from sklearn.model_selection import train_test_split

def plot_audio(data):
    if data.ndim > 1:
        data = data.flatten()
    plt.subplots()
    plt.plot(data)
    plt.title("Audio Signal")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(data, sampling_rate):
    power_spectrum, frequency_bins, time, image_axis = plt.specgram(data.flatten(), Fs=sampling_rate)

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title("Spectrogram")
    plt.xlabel("Time (samples)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

    return power_spectrum, frequency_bins, time


def n_pow2(x):
    return 1 << (x - 1).bit_length()

def check_and_create_directory(path_to_save: str) -> None:
    """
    Checks if directory exists and if not it creates it
    :param path_to_save: path of directory
    :return: None
    """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)


def get_timestamp():
    ltime = time.localtime(time.time())
    return f'{ltime.tm_year}_{ltime.tm_mon}_{ltime.tm_mday}_{ltime.tm_hour}_{ltime.tm_min}_{ltime.tm_sec}'


def pause(key):
    while True:
        if win32api.GetAsyncKeyState(key) & 0x0001 > 0:
            return True

def normalize_data(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    X_normalized = (array - mean) / std
    return X_normalized, mean, std

def feature_chromagram(waveform, sample_rate):
    # STFT computed here explicitly; mel spectrogram and MFCC functions do this under the hood
    stft_spectrogram = np.abs(librosa.stft(waveform))
    # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0)
    return chromagram

def feature_melspectrogram(waveform, sample_rate):
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128, fmax=8000).T,
                             axis=0)
    return melspectrogram

def feature_mfcc(waveform, sample_rate):
    # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # 40 filterbanks = 40 coefficients
    mfc_coefficients = np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=60).T, axis=0)
    return mfc_coefficients

def train_test_val_split(data_x, data_y, test_fraction, val_fraction) -> tuple[Any, Any, Any, Any, Any, Any]:
    """
    Splits data in train, test and validation
    :return: None
    """

    test_fraction = test_fraction / (1 - val_fraction)

    X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=val_fraction,
                                                      random_state=42, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_fraction,
                                                        random_state=42, shuffle=True)
    return X_train,  X_val, X_test, y_train, y_val, y_test

def create_feature_matrix(recording, rate):
    chromagram = feature_chromagram(recording, rate)
    melspectrogram = feature_melspectrogram(recording, rate)
    mfc_coefficients = feature_mfcc(recording, rate)
    # use np.hstack to stack our feature arrays horizontally to create a feature matrix
    return np.hstack((chromagram, melspectrogram, mfc_coefficients))

def add_noise(wav, amplitude, spread = 1):
    return wav + amplitude * np.random.normal(0, spread, len(wav))

def time_shift(wav, sr, amount):
    return np.roll(wav, int(sr / amount))

def time_stretch(wav, factor):
    factor = 0.4
    return librosa.effects.time_stretch(wav, factor)