import cv2
import time
import  PoseEstimationModule as pem

cap = cv2.VideoCapture('../videos/video12.mp4')
pTime = 0
detector = pem.PoseDetector()
success, img = cap.read()
h, w, c = img.shape
scale_percent = 20  # percent of original size
width = int(w * scale_percent / 100)
height = int(h * scale_percent / 100)
dim = (width, height)

while True:
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img = detector.findPose(img)
    lmList = detector.findPosition(img, height, width, draw=False)
    if len(lmList):
        cv2.circle(img, (lmList[25][1], lmList[25][2]), 5, (255, 0, 0), cv2.FILLED)
    # print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps: " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
    success, img = cap.read()