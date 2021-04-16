import time

import cv2

from src.getDensity import getFilteredDensity
from src.tracker import Tracker

if __name__ == "__main__":
    tracker = Tracker(adaptive=True, cuda=True)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 40)
    frame_num = 0.0
    start_time = time.time()
    capture = False
    while True:
        success, img = cap.read()
        if capture:
            frame_num += 1.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[72:400, 175:480]
            flow = tracker.track(img)
            density = getFilteredDensity(flow, use_cuda=True)
            density = ((density + 1.2) / 1.2 * 255.0).astype('uint8')
            density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
            cv2.imshow('img', img)
            cv2.imshow('density', density)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time > 1.0:
            capture = True
            fps = frame_num / (time.time() - start_time)
            print("{:.2f}".format(fps),
                  "FPS, current frame queue length: ",
                  tracker.getFrameQueueLength(),
                  end='\r')
            frame_num = 0.0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
