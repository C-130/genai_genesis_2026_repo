"""
Camera diagnostic — run this to see all available camera indices
and what resolution each one actually delivers.
"""

import cv2

print("Testing camera indices 0-4...\n")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"  Index {i}: not available")
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    actual_h, actual_w = frame.shape[:2] if ret else (0, 0)

    print(f"  Index {i}: reported {w}x{h}, actual frame {actual_w}x{actual_h}")
    cap.release()

print("\nNow showing live feed — press a number key to switch camera, Q to quit.")
print("This lets you see which index gives the best image.\n")

current = 0
cap = cv2.VideoCapture(current)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test", 2560, 1440)      # 2560 x 1440      # 1280 x 720

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Camera index: {current}  |  {w}x{h}  |  press 0-4 to switch, Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
        idx = int(chr(key))
        cap.release()
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        current = idx
        print(f"Switched to camera {idx}")

cap.release()
cv2.destroyAllWindows()
