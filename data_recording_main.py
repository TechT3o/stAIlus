import os
from recorder import Recorder
import win32api
from statics import plot_audio, get_timestamp, pause, check_and_create_directory
import os
import time

rec = Recorder()
SAVE_PATH = os.path.join(os.getcwd(), 'audio_data')
check_and_create_directory(SAVE_PATH)
SAMPLE_NO = 2
SECONDS = 3

# save_dir = os.path.join(SAVE_PATH, f'recording_{get_timestamp()}')
# check_and_create_directory(save_dir)
actions = ['B', 'K', 'L']

while True:
    if pause(ord('S')):
        time.sleep(2)
        print('Recording session started')
        for action in actions:
            save_dir = os.path.join(SAVE_PATH, action)
            check_and_create_directory(save_dir)
            print(f'Recording {action} action')
            for sample in range(SAMPLE_NO):
                if pause(ord(' ')):
                    print('Recording: ' + str(sample))
                    recording = rec.record_audio(SECONDS)
                    print('Recording stopped')
                    rec.write_audio_wav(os.path.join(save_dir, f'recording_{action}_no_{sample}_{get_timestamp()}.wav'),
                                        recording)
                    # plot_audio(recording)
