import time, sys
try:
    import av
except ImportError:
    print("pip install av"); sys.exit(1)
import cv2

INDEX = 1
print(f"Ouverture device AVFoundation index {INDEX}...")
container = av.open(
    str(INDEX),
    format='avfoundation',
    options={'framerate': '30', 'video_size': '1280x720'}
)
stream = container.streams.video[0]
print(f"OK : {stream.width}x{stream.height}\n")

decoder = container.decode(video=0)
count, eagain = 0, 0
saved = False

while count < 60 and eagain < 500:
    try:
        frame = next(decoder)
        img = frame.to_ndarray(format='bgr24')
        count += 1
        if not saved:
            cv2.imwrite('test_iphone_frame.jpg', img)
            print(f"  frame {count} -> test_iphone_frame.jpg SAVED")
            saved = True
        else:
            print(f"  frame {count} OK")
        eagain = 0
    except av.error.FFmpegError as e:
        if e.errno == 35:
            eagain += 1; time.sleep(0.005); continue
        print(f"FFmpeg error: {e}"); break
    except StopIteration:
        print(f"  StopIteration apres {count} frames"); break

container.close()
print(f"\nTotal: {count} frames lues. saved={saved}")
print("Verifie : ls -la test_iphone_frame.jpg")
