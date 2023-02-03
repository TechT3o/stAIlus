import pyaudio
import numpy as np
from statics import plot_audio, plot_spectrogram


class Recorder:

    def __init__(self):

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

    def record_audio(self):
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            int_data = np.frombuffer(data, dtype=np.int16)
            frames.append(int_data)
        return np.array(frames)

    def terminate_audio_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


if __name__ == "__main__":

    rec = Recorder()
    recording = rec.record_audio()
    plot_audio(recording)
    plot_spectrogram(recording, rec.RATE)
    rec.terminate_audio_stream()
