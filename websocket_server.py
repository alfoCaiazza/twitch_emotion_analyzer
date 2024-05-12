import asyncio
import websockets

connected = set()  # Un set per tenere traccia dei client connessi

async def send_emotions(message):
    while True:
        await asyncio.sleep(1)  # Decidi ogni quanto tempo vuoi inviare l'emozione
        if connected:  # Controlla se ci sono client connessi
            for websocket in connected:
                await websocket.send(message)
                print(f"Sent to client: {message}")
    
async def server(websocket, path):
    # Aggiunge il nuovo websocket al set dei connessi
    connected.add(websocket)
    try:
        while True:
            message = await websocket.recv()
            print(f"Received from client: {message}")
            await send_emotions(message)
    except websockets.exceptions.ConnectionClosedOK:
        print("Connessione chiusa normalmente")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Errore di chiusura connessione: {e}")
    finally:
        # Rimuove il websocket dal set quando la connessione si chiude
        connected.remove(websocket)
        print("Pulizia delle risorse del server")

start_server = websockets.serve(server, 'localhost', 6789)

loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
loop.run_forever()
