import asyncio
import websockets
import json
import positions_creator as pc
import sound_service as ss
import time

# Sequence ids
INIT_ROUTINE_START_SEQ_ID = 2
PROBE_LISTEN_MESSAGE_SEQ_ID = 25
STOP_PROBE_LISTEN_MESSAGE_SEQ_ID = 26
TOGGLE_UNIFORMIZATION_SEQ_ID = 27

# Websocket info
WEBSOCKET_ADDRESS = 'localhost'
WEBSOCKET_PORT = 8765

# Positions messaage
confirm_positions = pc.create('room.mat')
ss.initialize()

async def messageHandler(websocket, path):
  while True:
    async for data in websocket:
      json_message = json.loads(data)

      seq_id = json_message['seqId']

      if seq_id == INIT_ROUTINE_START_SEQ_ID:
        time.sleep(2)
        await websocket.send(confirm_positions)
      elif seq_id == PROBE_LISTEN_MESSAGE_SEQ_ID:
        ss.select_probe(json_message['data']['probeId'])
      elif seq_id == STOP_PROBE_LISTEN_MESSAGE_SEQ_ID:
        ss.unselect_probe()
      elif seq_id == TOGGLE_UNIFORMIZATION_SEQ_ID:
        ss.toggle_uniformization()

start_websockets_server = websockets.serve(messageHandler, WEBSOCKET_ADDRESS, WEBSOCKET_PORT)

asyncio.get_event_loop().run_until_complete(start_websockets_server)
asyncio.get_event_loop().run_forever()
