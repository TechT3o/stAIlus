import matplotlib.pyplot as plt


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
