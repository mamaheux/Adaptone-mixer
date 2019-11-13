import alsaaudio
import time
import threading
import glob
import scipy.io.wavfile

# Room info
WITH_UNIFORMISATION_PATH = 'asserv/'
WITHOUT_INIFORMISATION_PATH = 'no_uni/'
FILE_NAME_TEMPLATE = 'punk__millencolin-no-cigar_probe_'
FILE_NAME_EXTENSION = '.wav'
WRITE_FRAME_COUNT = 256
VOLUME = 2**16

uniformization = True
selected_id = 1
uniform_audio_frames = {}
default_audio_frame = {}
current_frame_number = 0
total_frame_count = 0
playback_device = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device='hw:CARD=x20,DEV=0')

def play_thread(thread_name):
  global playback_device, current_frame_number, selected_id, uniform_audio_frames, default_audio_frame, uniformization, total_frame_count

  while True:
    # We don't want to swap the selected probe mid write
    probe_id = selected_id
    is_uniform = uniformization
    write_data = b""

    for _ in range(WRITE_FRAME_COUNT):
      if current_frame_number >= total_frame_count:
        current_frame_number = 0

      if is_uniform:
        current_frame = uniform_audio_frames[probe_id][current_frame_number]
      else:
        current_frame = default_audio_frame[probe_id][current_frame_number]

      frame_bytes = current_frame.tobytes()
      write_data += frame_bytes
      write_data += frame_bytes
      write_data += bytearray(4 * 8)

      current_frame_number += 1

    playback_device.write(write_data)

def initialize():
  global playback_device, uniform_audio_frames, default_audio_frame, total_frame_count
  playback_device.setchannels(10)
  playback_device.setrate(44100)
  playback_device.setformat(alsaaudio.PCM_FORMAT_S32_LE)
  playback_device.setperiodsize(WRITE_FRAME_COUNT)

  for filename in glob.glob(WITH_UNIFORMISATION_PATH + '*' + FILE_NAME_EXTENSION):
    fs, data = scipy.io.wavfile.read(filename)
    file_number = filename.split(WITH_UNIFORMISATION_PATH + FILE_NAME_TEMPLATE)[1].split(FILE_NAME_EXTENSION)[0]
    uniform_audio_frames[int(file_number)] = data.astype('int32') * VOLUME

  for filename in glob.glob(WITHOUT_INIFORMISATION_PATH + '*' + FILE_NAME_EXTENSION):
    fs, data = scipy.io.wavfile.read(filename)
    file_number = filename.split(WITHOUT_INIFORMISATION_PATH + FILE_NAME_TEMPLATE)[1].split(FILE_NAME_EXTENSION)[0]
    default_audio_frame[int(file_number)] = data.astype('int32') * VOLUME

  total_frame_count = len(uniform_audio_frames[1])

  play = threading.Thread(target=play_thread, args=(1,))
  play.start()
  print('initialized')

def select_probe(probe_id):
  global selected_id
  selected_id = probe_id
  print('selected probe #' + str(probe_id))

def unselect_probe():
  global selected_id
  selected_id = 1
  print('unselected probe')

def toggle_uniformization():
  global uniformization
  uniformization = not uniformization
  print('uniformization: ' + str(uniformization))
