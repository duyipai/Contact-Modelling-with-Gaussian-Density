import math
import time

import cv2
import numpy as np
from scipy import ndimage
import src.utils as utils
from src.getDensity import getFilteredDensity
from src.tracker import Tracker

if __name__ == "__main__":
    tracker = Tracker(adaptive=True,
                      cuda=False)  # cuda=True if using opencv cuda
    cap = cv2.VideoCapture(0)
    frame_num = 0.0
    start_time = time.time()
    capture = False
    while True:
        success, img = cap.read()
        if capture:
            frame_num += 1.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[72:400, 175:480].copy()
            flow = tracker.track(img)
            density = getFilteredDensity(flow, use_cuda=True)
            # print(density.max(), density.min())
            density = (density + 1.0) / 1.0
            density = (density * 255.0).astype('uint8')
            threshold, contact_boundary = utils.getContactBoundary(density)
            density[np.where(density < threshold)] = 0
            center_of_mass = ndimage.measurements.center_of_mass(density)
            density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
            cv2.circle(density,
                       (int(center_of_mass[1]), int(center_of_mass[0])), 5,
                       (255, 0, 0), 2)
            arrows = utils.put_optical_flow_arrows_on_image(
                np.zeros_like(density), flow[15:-15, 15:-15, :])
            cv2.imshow('img', img)
            cv2.imshow('boundary', contact_boundary)
            cv2.imshow('arrows', arrows)
            cv2.imshow('density', density)
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        elif k == 0:
            tracker.reset()
            print("Tracker reset")
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
