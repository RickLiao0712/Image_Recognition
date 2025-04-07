# import cv2

# video = cv2.VideoCapture("C:\VNL.mp4")

# if not video.isOpened():
#     print("影片無法讀取！請確認檔案是否成功上傳。")
# else:
#     print("影片讀取成功！")

# video.release()
import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 讀取影片
video = cv2.VideoCapture("C:/VNLSERVE.mp4")
# video = cv2.VideoCapture("C:/VBHIT.mp4")
# 影片輸出設置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定 MP4 格式
out = cv2.VideoWriter('/mnt/data/output.mp4', fourcc, int(video.get(5)), 
                      (int(video.get(3)), int(video.get(4))))

# 設定黃色範圍（可以調整以適應排球顏色）
lower_yellow = np.array([25, 100,100])
upper_yellow = np.array([45, 255, 255])

track_points = deque(maxlen=10)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 過濾出黃色區域
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 去除雜訊（先侵蝕 erode 再膨脹 dilate）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 過濾掉太小的區域
        if cv2.contourArea(cnt) > 300 and cv2.contourArea(cnt)< 700:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 18 and radius < 45:  # 過濾掉過小、大的圓
                cv2.circle(frame, center, radius, (0, 255, 0), 2)  # 畫出圓形
            # 畫出球的運動軌跡
        for j in range(1, len(track_points)):
            if track_points[j - 1] is None or track_points[j] is None:
                continue
            cv2.line(frame, track_points[j - 1], track_points[j], (0, 0, 255), 2)

    # 顯示影像
    cv2.imshow("Detected Volleyball", frame)

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

video.release()
cv2.destroyAllWindows()
