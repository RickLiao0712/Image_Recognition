import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 讀取影片
video = cv2.VideoCapture('C:/VNLSERVE.mp4')
# video = cv2.VideoCapture("C:/VBHIT.mp4")
# 影片輸出設置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定 MP4 格式
out = cv2.VideoWriter('/mnt/data/output.mp4', fourcc, int(video.get(5)), 
                      (int(video.get(3)), int(video.get(4))))

# HSV 顏色範圍（黃色區域）
lower_yellow = np.array([20, 90, 90])
upper_yellow = np.array([35, 240, 240])

# 記錄軌跡的 deque
track_points = deque(maxlen=10)

while True:
    ret, frame = video.read()
    if not ret:
        break  # 影片讀取結束

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 過濾黃色
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 轉為灰階並模糊化（優化 Hough 圓偵測）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    blurred = cv2.bilateralFilter(gray, 50, 300, 1000)
    # Hough 圓偵測
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, param1=50, param2=30, minRadius=15, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            track_points.appendleft(center)
            cv2.circle(frame, center, i[2], (0, 255, 0), 2)  # 畫出球的輪廓

    # 繪製球的運動軌跡
    for j in range(1, len(track_points)):
        if track_points[j - 1] is None or track_points[j] is None:
            continue
        cv2.line(frame, track_points[j - 1], track_points[j], (255, 0, 255), 2)

    # 顯示結果
    cv2.imshow("Processed Video", frame)

    # 寫入輸出影片
    out.write(frame)

    # 按 'q' 退出
    if cv2.waitKey(30) & 0xFF == ord(' '):
        break

# 分解軌跡點
x_vals = [p[0] for p in track_points]
y_vals = [p[1] for p in track_points]

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Ball Trajectory')
plt.gca().invert_yaxis()  # 影像座標 (0,0) 在左上角，所以要反轉 y 軸
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Volleyball Serve Trajectory')
plt.legend()
plt.show()
# 釋放資源
video.release()
out.release()
cv2.destroyAllWindows()
