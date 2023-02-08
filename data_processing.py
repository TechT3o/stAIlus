import os
from recorder import Recorder
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from statics import normalize_data
import numpy as np


class DataProcessor:
    def __init__(self):
        self.DATA_PATH = 'audio_data'
        self.BATCH_SIZE = 32
        self.TEST_FRAC = 0.2
        self.VAL_FRAC = 0.2
        self.ACTION_SIZE = 2 # sec
        self.recorder = Recorder()
        self.recordings = []

    def load_data(self):
        for recording_sess in os.listdir(self.DATA_PATH):
            recording_folder = os.path.join(self.DATA_PATH, recording_sess)
            for recording in os.listdir(recording_folder):
                # self.recorder.play_from_file(os.path.join(recording_folder, recording))
                rate, signal = read(os.path.join(recording_folder, recording))
                self.recordings.append(signal)

    def process_data(self):
        for index, recording in enumerate(self.recordings):
            freq, time, spec = spectrogram(recording, self.recorder.RATE)
            # print(freq, time)
            # self.recordings[index] = spec / np.linalg.norm(spec)
            self.recordings[index] = spec


if __name__ == '__main__':

    processor = DataProcessor()
    processor.load_data()
    processor.process_data()
    normalized_data, mean, std = normalize_data(processor.recordings)
    print(normalized_data.shape, normalized_data)



