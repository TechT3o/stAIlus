import os
from recorder import Recorder
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from statics import normalize_data, create_feature_matrix, plot_spectrogram, plot_audio, add_noise, time_shift, time_stretch
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class DataProcessor:
    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.BATCH_SIZE = 32
        self.TEST_FRAC = 0.2
        self.VAL_FRAC = 0.2
        self.ACTION_SIZE = 2 # sec
        self.recorder = Recorder()
        self.recordings = []
        self.signal_lengths = []
        self.processed_data = []
        self.labels = []
        self.min_max_scaler = MinMaxScaler()

        self.augmented_data = []
        self.augmented_labels = []
        self.augmented_signal_lengths = []

        self.load_data()


    # keep our unscaled features just in case we need to process them alternatively

    def process_data(self):
        for index, recording in enumerate(self.recordings):
            freq, time, spec = spectrogram(recording, self.recorder.RATE)
            # print(freq, time)
            # self.recordings[index] = spec / np.linalg.norm(spec)
            self.recordings[index] = spec

    def process_data_irregular_shapes(self):

        for index, recording in enumerate(self.recordings):
            recording_length = len(recording) / self.recorder.RATE
            self.signal_lengths.append(recording_length)
            freq, time, spec = spectrogram(recording, self.recorder.RATE)
            resized = cv2.resize(spec, (129, 430))
            self.recordings[index] = resized

    def pre_process_representation_ensemble(self):
        for recording in self.recordings:

            # use np.hstack to stack our feature arrays horizontally to create a feature matrix
            feature_matrix = create_feature_matrix(recording, self.recorder.RATE)
            self.processed_data.append(feature_matrix)
        self.processed_data = self.min_max_scaler.fit_transform(self.processed_data)

    def pre_process_spectrogram(self):
        for recording in self.recordings:
            freq, time, spec = spectrogram(recording, self.recorder.RATE)
            #resized = cv2.resize(spec, (129, 430))
            self.processed_data.append(spec)
        # self.processed_data = self.min_max_scaler.fit_transform(self.processed_data)

    def load_data(self):
        for index, action in enumerate(os.listdir(self.DATA_PATH)):
            action_path = os.path.join(self.DATA_PATH, action)
            # action_sample_numbers.append(len(os.listdir(action_path)))
            for sample in os.listdir(action_path):
                print(os.path.join(action_path, sample))
                rate, signal = read(os.path.join(action_path, sample))
                self.recordings.append(signal.T[0])
                recording_length = len(signal.T[0]) / self.recorder.RATE
                self.signal_lengths.append(recording_length)
            # plot_audio(signal.T[0])
            # plot_spectrogram(signal.T[0], self.recorder.RATE)
            self.labels.append([index] * len(os.listdir(action_path)))
        self.labels = sum(self.labels, [])
        print(len(self.recordings), len(self.labels), len(self.signal_lengths))

    def train_test_split_data(self):
        print(self.processed_data.shape, np.array(self.labels).shape)
        return train_test_split(self.processed_data, self.labels, test_size=0.2, random_state=69)

    def augment_data(self):
        print('Augmenting data')
        time_shifts = [self.recorder.RATE*i for i in [-0.4, -0.25, 0.25, 0.4]]
        time_stretches = [1.5, 0.75]
        noise_amps = [0.005, 0.002, 0.001]

        for index, recording in tqdm(enumerate(self.recordings)):
            self.augmented_data.append(recording)
            self.augmented_labels.append(self.labels[index])
            self.augmented_signal_lengths.append(self.signal_lengths[index])

            for noise_amplitude in noise_amps:
                noisy_sample = add_noise(recording, noise_amplitude, spread=1)
                self.augmented_data.append(noisy_sample)
                self.augmented_labels.append(self.labels[index])
                self.augmented_signal_lengths.append(self.signal_lengths[index])

                for shift in time_stretches:
                    # augmented = time_shift(noisy_sample, self.recorder.RATE, shift)
                    augmented = time_stretch(noisy_sample, shift)
                    self.augmented_data.append(augmented)
                    self.augmented_labels.append(self.labels[index])
                    self.augmented_signal_lengths.append(self.signal_lengths[index])

    def pre_process_augmented_ensemble(self):
        print('Preprocessing augmented data')
        for recording in tqdm(self.augmented_data):
            # use np.hstack to stack our feature arrays horizontally to create a feature matrix
            feature_matrix = create_feature_matrix(recording, self.recorder.RATE)
            # freq, time, spec = spectrogram(recording, self.recorder.RATE)
            self.processed_data.append(feature_matrix)
        print('Normalizing')
        self.processed_data = self.min_max_scaler.fit_transform(self.processed_data)

    def augmented_train_test_split_data(self):
        print('Splitting augmented data')
        print(self.processed_data.shape, np.array(self.augmented_labels).shape)
        return train_test_split(self.processed_data, self.augmented_labels, test_size=0.2, random_state=69)


if __name__ == '__main__':

    processor = DataProcessor('audio_data')
    processor.load_data()
    processor.process_data()
    normalized_data, mean, std = normalize_data(processor.recordings)
    print(normalized_data.shape, normalized_data)



