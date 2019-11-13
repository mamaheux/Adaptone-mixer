import json
import random
import scipy.io as sio

CONFIRM_POSITIONS_SEQ_ID = 3
current_mic_index = 1

def create_speaker_item(speaker):
  speaker_json = {}
  speaker_json['x'] = speaker[0]
  speaker_json['y'] = speaker[1]
  speaker_json['type'] = 's'

  return speaker_json

def create_microphone_item(microphone):
  global current_mic_index

  microphone_json = {}
  microphone_json['id'] = current_mic_index
  microphone_json['errorRate'] = random.uniform(0, 0.12)
  microphone_json['x'] = microphone[0]
  microphone_json['y'] = microphone[1]
  microphone_json['type'] = 'm'

  current_mic_index = current_mic_index + 1

  return microphone_json

def create(filename):
  room_data = sio.loadmat(filename)
  room_json = {}
  data = {}

  room_json['seqId'] = CONFIRM_POSITIONS_SEQ_ID
  data['firstSymmetryPositions'] = list(map(create_speaker_item, room_data['speakers'])) + list(map(create_microphone_item, room_data['probes']))
  data['secondSymmetryPositions'] = data['firstSymmetryPositions']
  room_json['data'] = data

  return json.dumps(room_json)
