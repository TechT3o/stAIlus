import matplotlib.pyplot as plt
import os
import time
import win32api
import numpy as np


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