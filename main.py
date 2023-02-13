from data_processing import DataProcessor
from ml_model import MLModel
from recorder import Recorder
from statics import normalize_data
import numpy as np


recorder = Recorder()
ml_model = MLModel()
data_processing = DataProcessor()


def oh_encode(number):
    if number == 1:
        return [1, 0, 0]
    if number == 2:
        return [0, 1, 0]
    if number == 3:
        return [0, 0, 1]


model = MLModel()
processor = DataProcessor()
processor.load_data()
processor.process_data()
normalized_data, mean, std = normalize_data(processor.recordings)

un = normalized_data[0][:, :120, :]
dos = normalized_data[1][:, :20, :]
tres = normalized_data[2][:, :90, :]
normalized_data = normalized_data.tolist()
normalized_data[0] = un.tolist()
normalized_data[1] = dos.tolist()
normalized_data[2] = tres.tolist()
#print(normalized_data.shape, normalized_data[0])
model.fcn_model()
labels = [[1, 0, 0]]*5 + [[0, 1, 0]]*5 + [[0, 0, 1]]*5
model.train_model(normalized_data, np.array(labels))



