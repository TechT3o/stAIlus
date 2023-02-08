from data_processing import DataProcessor
from ml_model import MLModel
from recorder import Recorder
from scipy.signal import spectrogram
import win32api

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


while True:
    if win32api.GetAsyncKeyState(ord('S'))&0x0001 > 0:
        print('Recording started')
        data_array = []
        labels = []
        for action in range(2):
            recordings = []
            for recording in range(4):
                print(f'Action {action+1} recording {recording+1}')
                audio_recording = recorder.record_audio(seconds=3)
                freq, time, spec = spectrogram(audio_recording, recorder.RATE)
                print(spec.shape)
                recordings.append(spec)
            data_array.append(recordings)
            labels.append(oh_encode(action))

    if win32api.GetAsyncKeyState(ord('T'))&0x0001 > 0:
        print('Training started')
        ml_model.fcn_model()
        ml_model.train_model(data_array, labels)

    if win32api.GetAsyncKeyState(ord('P'))&0x0001 > 0:
        print('Testing started')
        while True:
            if win32api.GetAsyncKeyState(ord('P')) & 0x0001 > 0:
                print('Testing started')
                audio_recording = recorder.record_audio(seconds=3)
                freq, time, spec = spectrogram(audio_recording, recorder.RATE)
                prediction = ml_model.model.predict(spec)
                print(prediction)


