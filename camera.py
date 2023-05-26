from queue import Queue
import threading
import traceback
import cv2

class Camera:
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.capture = None
        self.stopped = False
        self.display_frames_queue = Queue()
        self.odometry_frames_queue = Queue()
        self.capture_thread = None
        self.display_thread = None

    def connect(self):
        url = f"http://{self.address}:{self.port}/video"
        self.capture = cv2.VideoCapture(url)
        self.stopped = False

        self.capture_thread = threading.Thread(target=self.update, args=())
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.display_thread = threading.Thread(target=self.display, args=())
        self.display_thread.daemon = True
        self.display_thread.start()

    def update(self):
        while not self.stopped:
            try:
                ret, frame = self.capture.read()
                if ret:
                    self.display_frames_queue.put(frame.copy())
                    self.odometry_frames_queue.put(frame.copy())
            except Exception as e:
                print("Error in Camera.update():", e)
                self.stopped = True
                break

    def display(self):
        while not self.stopped:
            if not self.display_frames_queue.empty():
                frame = self.display_frames_queue.get()
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopped = True
                    break

    def get_frame_for_odometry(self):
        if not self.odometry_frames_queue.empty():
            return self.odometry_frames_queue.get()

    def disconnect(self):
        self.stopped = True
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.display_thread is not None:
            self.display_thread.join()