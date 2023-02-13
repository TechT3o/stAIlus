import pyaudio
import wave
import numpy as np
from sklearn import svm
import tkinter as tk


def start_countdown(event=None):
    #button.config(state="disable")
    for i in range(5, 0, -1):
        label.config(text=str(i))
        root.update()
        root.after(1000, lambda: None)
    label.config(text="Recording...")
    root.update()
    start_recording()

def start_recording():
    # Your code to start recording goes here
    # Start recording audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "record.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a wave file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Load the recorded audio and run the SVM model on it
    audio_data = np.frombuffer(b''.join(frames), np.int16)
    X = audio_data.reshape(-1, 1)
    clf = svm.SVC(kernel='linear')
    clf.fit(X, [0])
    y_pred = clf.predict(X)
    interaction_pred = int(y_pred[0])

    # Update the GUI with the class prediction
    label.config(text=f"Class: {interaction_pred}")
    #button.config(state="normal")

root = tk.Tk()
root.title("STaiLUS Classifier")
root.geometry("600x400")

label = tk.Label(root, text="Press SPACE to Start Recording", font=("Ubuntu Condensed", 20))
label.pack(pady=10)

#button = tk.Button(root, text="Start", command=start_countdown)
root.bind("<space>", start_countdown)
#button.pack(pady=10)

root.mainloop()