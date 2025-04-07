import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# video = cv2.VideoCapture("C:/VBHIT.mp4")
video = cv2.VideoCapture("C:/VNLSERVE.mp4")

# tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerCSRT_create()  # 創建追蹤器
tracking = False                    # 設定 False 表示尚未開始追蹤
track_points = deque(maxlen=10)

while True:
    ret, frame = video.read()
    if not ret:
        print("Cannot receive frame or END")
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame,(540,300))  # 縮小尺寸，加快速度
    keyName = cv2.waitKey(1)

    if keyName == ord('q'):
        break 
    if keyName == ord('a'):
        area = cv2.selectROI(frame, showCrosshair=False, fromCenter=False)
        tracker = cv2.TrackerCSRT_create()
        # tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, area)    # 初始化追蹤器
        tracking = True              # 設定可以開始追蹤
    if tracking:
        success, point = tracker.update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
        if success:
            p1 = [int(point[0]), int(point[1])]
            p2 = [int(point[0] + point[2]), int(point[1] + point[3])]
            cv2.rectangle(frame, p1, p2, (0,0,255), 3)   # 根據座標，繪製四邊形，框住要追蹤的物件
            # 計算中心點座標
            center_x = int((p1[0] + p2[0]) / 2)
            center_y = int((p1[1] + p2[1]) / 2)
            center = (center_x, center_y)
            track_points.append(center)
            for j in range(1, len(track_points)):
                if track_points[j - 1] is None or track_points[j] is None:
                    continue
                cv2.line(frame, track_points[j - 1], track_points[j], (255, 0, 255), 2)


    cv2.imshow('Detected Volleyball', frame)

video.release()
cv2.destroyAllWindows()