import asyncio
import json
import websockets
from websockets.exceptions import ConnectionClosed

async def main():
    uri = "ws://127.0.0.1:8000/ngsi-ld/v1/subscriptions/urn:subscription:broker2-gtfs/ws"

    try:
        async with websockets.connect(uri) as ws:
            while True:
                msg = await ws.recv()
                print("Notification received:")
                print(msg)

                data = json.loads(msg)
                if data.get("status") == "stopped":
                    print("Subscription stopped cleanly.")
                    break

    except ConnectionClosed as e:
        print(f"WebSocket closed: code={e.code}, reason={e.reason}")

asyncio.run(main())
