from typing import Tuple, Any
import numpy as np
from ml_model import MLModel
from scipy.io.wavfile import read
from data_processing import DataProcessor
import os
from statics import normalize_data, plot_audio, plot_spectrogram, train_test_val_split


DATA_PATH = r'C:\Users\thpap\PycharmProjects\stAIlus\HCI_Trials'

processor = DataProcessor(DATA_PATH)
model = MLModel()

action_sample_numbers = []

for index, action in enumerate(os.listdir(DATA_PATH)):
    action_path = os.path.join(DATA_PATH, action)
    action_sample_numbers.append(len(os.listdir(action_path)))
    for sample in os.listdir(action_path):
        print(os.path.join(action_path, sample))
        rate, signal = read(os.path.join(action_path, sample))

        processor.recordings.append(signal.T[0])

print(signal.T[0].shape)
# plot_audio(signal.T[0])
# plot_spectrogram(signal.T[0], processor.recorder.RATE)

processor.process_data_irregular_shapes()
normalized_data, mean, std = normalize_data(processor.recordings)
labels = [[1, 0, 0]] * action_sample_numbers[0] + [[0, 1, 0]] * action_sample_numbers[1] + [[0, 0, 1]] * action_sample_numbers[2]
print(processor.signal_lengths)
X_train,  X_val, X_test, y_train, y_val, y_test = train_test_val_split(normalized_data, labels, 0.1, 0.3)
# X_train,  X_val, X_test, y_train, y_val, y_test = train_test_val_split(normalized_data, labels, 0.2, 0.2)
sec_train,  sec_val, sec_test, y_train, y_val, y_test = train_test_val_split(processor.signal_lengths, labels, 0.1, 0.3)
# print(processor.signal_lengths)
# model.fcn_model()
print(X_train.shape)
model.double_input_model((430, 129, 1))
model.train_model(x_train=[np.expand_dims(X_train, -1), np.array(sec_train)], y_train=np.array(y_train),
                  x_val=[np.expand_dims(X_val, -1), np.array(sec_val)], y_val=np.array(y_val))
model.test_model(x_test=[np.expand_dims(X_test, -1), np.array(sec_test)], y_test=np.array(y_test))




