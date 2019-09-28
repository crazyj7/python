import os,sys
import glob
# convert...
from pydub import AudioSegment
'''
MP3 파일을 wav 로 변환한다음
다시 MP3로 변환한다.   (왜?? meta/copyright 제거.)
'''


files = glob.glob(r'D:\Music\이지클래식1\*.mp3')
cnt=0

for file in files:
    cnt+=1
    f,e = os.path.splitext(file)
    wavfile = f+'.wav'
    mp3file = f+'@.mp3'
    # print(file, wavfile, mp3file)
    print(cnt, '/', len(files), file)

    if True:
        wa = AudioSegment.from_mp3(file)
        wa.export(wavfile, format='wav')

        wa2 = AudioSegment.from_wav(wavfile)
        wa2.export(mp3file, format='mp3')
        os.remove(wavfile)
