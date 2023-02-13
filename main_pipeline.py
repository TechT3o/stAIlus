import pickle
from ml_model import MLModel
from statics import pause, create_feature_matrix, plot_audio, plot_spectrogram

DATA_PATH = r'C:\Users\thpap\PycharmProjects\stAIlus\HCI_Trials2'
# # DATA_PATH = r'C:\Users\thpap\PycharmProjects\stAIlus\audio_data'
model = MLModel(DATA_PATH)
# model.augmentation()
# # model.normal()
# svm = model.build_train_test_svm()
# pickle.dump(svm, open('svm.sav', 'wb'))

svm = pickle.load(open('svm.sav', 'rb'))

while True:
    if pause(ord(' ')):
        print('Recording started')
        # recording = model.processor.recorder.record_audio(4).flatten()
        recording = model.processor.recorder.record_audio_key().flatten()
        plot_audio(recording)
        # plot_spectrogram(recording, model.processor.recorder.RATE)
        print('Recording stopped')
        feat_mat = create_feature_matrix(recording, model.processor.recorder.RATE)
        # nput = model.processor.min_max_scaler.transform([feat_mat])
        input = model.processor.min_max_scaler.fit_transform([feat_mat])
        print(svm.predict_proba(input))
