import pygame
import pygame.midi
import time

from pyknon.music import Note, NoteSeq
from pyknon.genmidi import Midi


class PlayMidi:

    def __init__(self):
        # pick a midi music file you have ...
        # (if not in working folder use full path)
        freq = 44100  # audio CD quality
        bitsize = -16  # unsigned 16 bit
        channels = 2  # 1 is mono, 2 is stereo
        buffer = 1024  # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)
        # optional volume 0 to 1.0
        pygame.mixer.music.set_volume(0.8)

    def play(self, music_file):
        """
        stream music with mixer.music module in blocking manner
        this will stream the sound from disk while playing
        """
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(music_file)
            print ("Music file %s loaded!" % music_file)
        except pygame.error:
            print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # check if playback has finished
            clock.tick(30)

    def printSeq(self, seq):
        for n in seq:
            try:
                print( "{}({:2.2},O{}) ".format(n.name, n.dur, n.octave), end="")
            except AttributeError:
                # print( "{}({:2.2},O{}) ".format("R", n.dur, 0), end="")
                print(n,' ', end="")
        print("")

    def close(self):
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()


if __name__=="__main__":
    if False:
        pygame.midi.init()
        print ( pygame.midi.get_default_output_id())
        print( pygame.midi.get_device_info(0))
        player = pygame.midi.Output(0)
        # player.set_instrument(0)
        player.set_instrument(115, 0)
        player.note_on(64, 127)
        time.sleep(1)
        player.note_off(64, 127)

        del player
        pygame.midi.quit()

        exit()


    try:
        pm = PlayMidi()

        # make
        notes = "c d8 e f# r g4 r b c'' d d'' d''' c b' a"
        # r은 쉼표.
        # 음 길이는 숫자로 지정. 뒤에 나오는 음들의 디폴트 길이가 된다. 길이 지정시 변경됨. (쉼표길이도)
        # 반음 표시는 #, b 를 음뒤에 붙임.
        # 한 옥타브 올림은 '' 이후 계속 유지됨. ''를 더 써도 이미 올라가서 안됨. 더 올리려면 ''' 세 개.
        # 원래 옥타브로 돌아가려면 ' 한 개.
        seq = NoteSeq(notes)*2      # *로 반복 횟수 지정.
        pm.printSeq(seq)
        midi = Midi(number_tracks=1, tempo=120)
        midi.seq_notes(seq, track=0)
        midi.write("test.mid")

        notes21 = "c2 c g a#8  c2"
        notes22 = "e2 f b r8   e2"
        notes23 = "g2 a d gb8  g2"
        seq21 = NoteSeq(notes21)
        seq22 = NoteSeq(notes22)
        seq23 = NoteSeq(notes23)
        pm.printSeq(seq23)
        midi = Midi(number_tracks=3, tempo=120)
        midi.seq_notes(seq21, track=0)
        midi.seq_notes(seq22, track=1)
        midi.seq_notes(seq23, track=2)
        midi.write("test2.mid")

        n = Note("D", 4, 4)

        # play
        music_file = "test.mid"
        pm.play(music_file)
        print('end')
    except KeyboardInterrupt:
        pm.close()
        raise SystemExit

