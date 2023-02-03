from scipy.signal import spectrogram, resample, decimate

class DataProcessor:
    def __init__(self):
        self.DATA_PATH = ''
        self.BATCH_SIZE = 32
        self.TEST_FRAC = 0.2
        self.VAL_FRAC = 0.2
        self.ACTION_SIZE = 2 # sec

    def resample_to_action_size(self, signal):
        sample_size = len(signal)
        resampling_factor = sample_size/self.ACTION_SIZE

        if resampling_factor>1:
            decimate(signal)
        elif resampling_factor<1:

        else:
            resampled_signal = signal
        return signal


