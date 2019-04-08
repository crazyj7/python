"""PyAudio Example: Play a wave file."""

import pyaudio
import wave
import sys

CHUNK = 1024

fname = 'a2002011001-e02.wav'
# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)
# wf = wave.open(sys.argv[1], 'rb')
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
    if cnt==100:
        break

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()

