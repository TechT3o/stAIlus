from ml_model import MLModel
from statics import pause, create_feature_matrix, plot_audio, plot_spectrogram

# DATA_PATH = r'C:\Users\thpap\PycharmProjects\stAIlus\HCI_Trials'
DATA_PATH = r'C:\Users\thpap\PycharmProjects\stAIlus\audio_data'
model = MLModel(DATA_PATH)
svm = model.build_svm()

while True:
    if pause(ord('S')):
        print('Recording started')
        recording = model.processor.recorder.record_audio(4).flatten()
        plot_audio(recording)
        plot_spectrogram(recording, model.processor.recorder.RATE)
        print('Recording stopped')
        feat_mat = create_feature_matrix(recording, model.processor.recorder.RATE)
        input = model.processor.min_max_scaler.transform([feat_mat])
        print(svm.predict_proba(input))
