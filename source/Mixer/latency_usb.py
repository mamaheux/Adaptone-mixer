import alsaaudio
import numpy as np
import matplotlib.pyplot as plt

def main():
    capture_device = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device='hw:CARD=x20,DEV=0')
    playback_device = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device='hw:CARD=x20,DEV=0')
    
    print capture_device.setchannels(10)
    print capture_device.setrate(44100)
    print capture_device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    print capture_device.setperiodsize(8)
    
    print playback_device.setchannels(10)
    print playback_device.setrate(44100)
    print playback_device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    print playback_device.setperiodsize(8)

    while True:
        length, data = capture_device.read()
        playback_device.write(data)

if __name__ == '__main__':
    main()
