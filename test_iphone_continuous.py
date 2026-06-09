"""
Test PyAV en flux continu via thread reader + queue.
On vise 60 frames sans coupure.
"""
import time
import threading
import queue
import sys
try:
    import av
except ImportError:
    print("pip install av"); sys.exit(1)
import cv2

INDEX = 1


class PyAVCamera:
    """Wrapper PyAV qui imite cv2.VideoCapture (read/release)."""

    def __init__(self, device_index, size="1280x720", framerate=30):
        self.container = av.open(
            str(device_index),
            format='avfoundation',
            options={'framerate': str(framerate), 'video_size': size}
        )
        self.stream = self.container.streams.video[0]
        self.width, self.height = self.stream.width, self.stream.height
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            try:
                for frame in self.container.decode(video=0):
                    if self._stop.is_set():
                        return
                    img = frame.to_ndarray(format='bgr24')
                    # On garde la frame la plus fraîche (drop la vieille)
                    if self._queue.full():
                        try: self._queue.get_nowait()
                        except queue.Empty: pass
                    self._queue.put(img)
            except av.error.FFmpegError as e:
                if e.errno == 35:  # EAGAIN
                    time.sleep(0.003); continue
                time.sleep(0.05); continue
            except StopIteration:
                # decoder épuisé → on relance la boucle for
                time.sleep(0.005); continue
            except Exception as e:
                print(f"[reader] {type(e).__name__}: {e}")
                time.sleep(0.05)

    def read(self):
        try:
            return True, self._queue.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        self._stop.set()
        try: self.container.close()
        except Exception: pass
        self._thread.join(timeout=1.0)


print(f"Ouverture iPhone via PyAV (index {INDEX})...")
cam = PyAVCamera(INDEX)
print(f"OK : {cam.width}x{cam.height}\n")

count, fails, t0 = 0, 0, time.time()
while count < 60:
    ok, img = cam.read()
    if ok:
        count += 1
        if count % 10 == 0:
            fps = count / (time.time() - t0)
            print(f"  frame {count}  fps={fps:.1f}")
        if count == 30:
            cv2.imwrite('test_iphone_cont.jpg', img)
            print(f"  -> test_iphone_cont.jpg sauvegardée")
    else:
        fails += 1
        print(f"  read() False ({fails})")
        if fails > 5: break

cam.release()
print(f"\nTotal : {count} frames en {time.time()-t0:.1f}s ({count/(time.time()-t0):.1f} fps)")
