import math
import time

import cv2
import numpy as np
from scipy import ndimage

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
    kernel = np.ones((5, 5), np.uint8)
    while True:
        success, img = cap.read()
        if capture:
            frame_num += 1.0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[72:400, 175:480]
            flow = tracker.track(img)
            density = getFilteredDensity(flow, use_cuda=True)
            # print(density.max(), density.min())
            density = (density + 1.0) / 1.0
            # density_grad = np.gradient(density_grad_mag)
            # density_grad_mag = np.hypot(density_grad[0], density_grad[1])
            center_of_mass = ndimage.measurements.center_of_mass(density)
            density = (density * 255.0).astype('uint8')
            contact_area = cv2.threshold(density,
                                         0,
                                         255,
                                         type=cv2.THRESH_BINARY +
                                         cv2.THRESH_OTSU)
            # print(density_grad_mag[0])
            if contact_area[0] > 5:
                contact_area = contact_area[1]
                contact_area = cv2.morphologyEx(contact_area,
                                                cv2.MORPH_OPEN,
                                                kernel,
                                                iterations=5)
            else:
                contact_area = np.zeros_like(contact_area[1])
            contact_boundary = np.gradient(contact_area.astype('float32'))
            contact_boundary = np.hypot(contact_boundary[0],
                                        contact_boundary[1]).astype('uint8')
            lines = cv2.HoughLines(contact_boundary, 8, np.pi / 20, 30)
            contact_boundary = cv2.cvtColor(contact_boundary,
                                            cv2.COLOR_GRAY2BGR)
            if lines is not None and len(lines) > 1:
                # print(len(lines))
                lines = lines[:2]
                if np.abs(lines[0][0][1] - lines[1][0][1]) < np.pi / 4.0:
                    if np.abs(lines[0][0][1] - lines[1][0][1]) < 1e-2:
                        a = -1
                        b = -1
                    else:
                        Theta = np.array([[
                            math.cos(lines[0][0][1]),
                            math.sin(lines[0][0][1])
                        ], [
                            math.cos(lines[1][0][1]),
                            math.sin(lines[1][0][1])
                        ]])
                        Rho = np.array([[lines[0][0][0]], [lines[1][0][0]]])
                        intersection = np.matmul(np.linalg.inv(Theta), Rho)
                        a = intersection[0].item()
                        b = intersection[1].item()
                    if not (a > 0 and a < density.shape[1] and b > 0
                            and b < density.shape[0]):
                        for i in range(0, len(lines)):
                            rho = lines[i][0][0]
                            theta = lines[i][0][1]
                            a = math.cos(theta)
                            b = math.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                            cv2.line(contact_boundary, pt1, pt2, (0, 0, 255),
                                     3, cv2.LINE_AA)
            density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
            cv2.circle(density,
                       (int(center_of_mass[1]), int(center_of_mass[0])), 5,
                       (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.imshow('boundary', contact_boundary)
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
