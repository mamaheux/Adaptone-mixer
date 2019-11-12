#import alsaaudio
import time
from threading import Thread

# Room info
WITH_UNIFORMISATION_PATH = 'asser/'
WITHOUT_INIFORMISATION_PATH = 'no_uni/'
FILE_NAME_TEMPLATE = 'punk__millencolin-no-cigar_probe_'
FILE_NAME_EXTENSION = '.wav'

uniformization = True
#playback_device = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device='hw:CARD=x20,DEV=0')

def play_thread(thread_name):
  global playback_device
  data = 0
  while 1:
    playback_device.write(data)

def initialize():
  #global playback_device
  #playback_device.setchannels(2)
  #playback_device.setrate(44100)
  #playback_device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
  #playback_device.setperiodsize(8)
  
  #play = threading.Thread(target=play_thread, args=(1,))
  #play.start()
  print('initialized')

def select_probe(probe_id):
  print('selected probe #' + str(probe_id))

def unselect_probe():
  print('unselected probe')

def toggle_uniformization():
  global uniformization
  uniformization = not uniformization
  print('uniformization: ' + str(uniformization))