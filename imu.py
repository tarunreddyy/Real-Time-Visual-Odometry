from queue import Queue
import threading
import json
import websocket

class Sensor:
    def __init__(self, address, sensor_type):
        self.address = address
        self.sensor_type = sensor_type
        self.stopped = False
        self.ws = None
        self.data_queue = Queue(maxsize=10) 

    def on_message(self, ws, message):
        data = json.loads(message)
        values = data['values']
        timestamp = data['timestamp']

        if not self.data_queue.full():
            self.data_queue.put((values, timestamp))
        else:
            self.data_queue.get() 
            self.data_queue.put((values, timestamp))

    def get_data(self):
        if not self.data_queue.empty():
            return self.data_queue.get()
        else:
            return None, None
        
    def on_error(self, ws, error):
        print("error occurred")
        print(error)

    def on_open(self, ws):
        print(f"connected to : {self.sensor_type}")

    def on_close(self, ws, close_code, reason):
        print(f"connection closed: {self.sensor_type}")
        self.stopped = True 

    def make_websocket_connection(self):
        self.ws = websocket.WebSocketApp(f"ws://{self.address}/sensor/connect?type={self.sensor_type}",
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)

        self.ws.run_forever()

    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start()

    def disconnect(self):
        if self.ws is not None:
            self.ws.close() 