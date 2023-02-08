import pyaudio
import numpy as np
from scipy.io.wavfile import write, read
from statics import plot_audio, plot_spectrogram


class Recorder:

    def __init__(self):

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.p = pyaudio.PyAudio()
        self.stream_in = self.p.open(format=self.FORMAT,
                                     channels=self.CHANNELS,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.CHUNK)

        self.stream_out = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      output=True,
                                      frames_per_buffer=self.CHUNK)

    def record_audio(self, seconds):
        self.stream_in.start_stream()

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * seconds)):
            data = self.stream_in.read(self.CHUNK)
            int_data = np.frombuffer(data, dtype=np.int16)
            frames.append(int_data)

        self.stream_in.stop_stream()

        return np.array(frames)

    def terminate_pyaudio(self):
        if self.stream_in.is_active():
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.stream_out.is_active():
            self.stream_out.stop_stream()
            self.stream_out.close()
        self.p.terminate()

    def write_audio_wav(self, file_path, audio):
        write(file_path, self.RATE, np.round(audio).astype(np.int16))

    def play_back_sound(self, audio):
        self.stream_out.start_stream()
        # print(audio)
        for chunk in audio:
            self.stream_out.write(chunk.tobytes())

        self.stream_out.stop_stream()

    def play_from_file(self, filename):
        rate, signal = read(filename)
        self.play_back_sound(signal)

    def list_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_index(i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.p.get_device_info_by_index(i).get('name'))
            if (self.p.get_device_info_by_index(i).get('maxOutputChannels')) > 0:
                print("Output Device id ", i, " - ", self.p.get_device_info_by_index(i).get('name'))


if __name__ == "__main__":
    # import cv2
    # from scipy.signal import spectrogram
    #
    # rec = Recorder()
    # audio_recording = rec.record_audio(3)
    # freq, time, spec = spectrogram(audio_recording, rec.RATE)
    # print(spec.shape)
    # cv2.imshow('spectrog', spec)
    # cv2.waitKey(0)
    #
    # resized = cv2.resize(spec, (400, 600))
    # cv2.imshow('resized_spec', resized)
    # cv2.waitKey(0)


    rec = Recorder()
    rec.list_devices()
    recording = rec.record_audio(10)
    plot_audio(recording)
    power_spectrum, f, t = plot_spectrogram(recording, rec.RATE)
    print(np.array(power_spectrum).shape)
    rec.write_audio_wav('test.wav', recording)
    rec.play_from_file('test.wav')
    rec.terminate_pyaudio()

