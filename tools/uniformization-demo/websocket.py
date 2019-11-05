import asyncio
import websockets
import json

PROBE_LISTEN_MESSAGE_SEQ_ID = 25
STOP_PROBE_LISTEN_MESSAGE_SEQ_ID = 26
WEBSOCKET_ADDRESS = "localhost"
WEBSOCKET_PORT = 8765

async def messageHandler(websocket, path):
  while True:
    async for data in websocket:
      json_message = json.loads(data)

      seq_id = json_message['seqId']

      if seq_id == PROBE_LISTEN_MESSAGE_SEQ_ID:
        # Handle listening to the corresponding probe
        probe_id = json_message['data']['probeId']
      elif seq_id == STOP_PROBE_LISTEN_MESSAGE_SEQ_ID:
        # Stop listening to the corresponding probe

start_websockets_server = websockets.serve(messageHandler, WEBSOCKET_ADDRESS, WEBSOCKET_PORT)

asyncio.get_event_loop().run_until_complete(start_websockets_server)
asyncio.get_event_loop().run_forever()
