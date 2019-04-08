import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
fname = "audio.wav"


wf = wave.open(fname, 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
data = wf.readframes(CHUNK)

# play stream (3)
cnt=0
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK) # 1024 frames
    print('cnt=', cnt)
    cnt+=1
    # if cnt==100:
    #     break

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()


