import cv2
import time
import numpy as np
import math
import os
from HandTracking import HandTrackingModule as htm

'''
###########################
        CONTROLS
Single index finger up - draw
index and ring finger up - select color
thumb and index up - move brush without painting 
index, ring and thumb up - select brush thickness
all fingers up - steady state 
###########################
'''

####################################
wCam, hCam = 1280, 720
####################################

folderPath = "../Images/Header-Files"
mylist = os.listdir(folderPath)
print(mylist)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Declarations
header = overlayList[0]
drawColor = (255, 0, 255)
pTime = 0
xp, yp = 0, 0
detector = htm.handDetector(detectionCon=0.8, maxHands=1)
brushThickness = 25
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
smoothening = 5
clocX, clocY = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[4][1:]
        fingerStatus = detector.fingersUp()
        clocX = int(xp + (x1 - xp) / smoothening)
        clocY = int(yp + (y1 - yp) / smoothening)
        print(clocX, clocY)

        # Steady state
        if all(x == 1 for x in fingerStatus):
            pass

        # Thickness adjustment
        elif fingerStatus[0] and fingerStatus[1] and fingerStatus[2]:
            print("Thickness Mode")
            cx, cy = (x1 + x3) // 2, (y1 + y3) // 2
            distance = int(math.hypot(x3 - x1, y3 - y1))
            brushThickness = int(np.interp(distance, [35, 160], [1, 120]))
            print(distance, brushThickness)
            cv2.line(img, (x1, y1), (x3, y3), (0, 250, 0), 2)
            cv2.circle(img, (cx, cy), brushThickness//2, drawColor, cv2.FILLED)

        # Select mode if two fingers(index and ring) up
        elif fingerStatus[1] and fingerStatus[2] and not fingerStatus[0]:
            # xp, yp = 0, 0
            print("Selection Mode")

            # # Checking for the click
            if y1 < 125:
                if 100 < x1 < 300:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 400 < x1 < 600:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 650 < x1 < 800:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 900 < x1 < 1050:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # # Draw mode if index finger up
        elif fingerStatus[1] and not fingerStatus[2] and not fingerStatus[0]:
            cv2.circle(img, (x1, y1), brushThickness//2, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = clocX, clocY

            cv2.line(img, (xp, yp), (clocX, clocY), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (clocX, clocY), drawColor, brushThickness)
        # Erase everything
        # elif all(x == 0 for x in fingerStatus):
        #     imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        xp, yp = clocX, clocY

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:125, 100:1130] = header[:, 250:]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.rectangle(img, (100,125), (wCam-170, hCam-150), (0,0,0), 2, cv2.FILLED)
    cv2.putText(img, "FPS: " + str(int(fps)), (6, hCam//2-13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 250, 250), 1)
    cv2.putText(img, "BRUSH : " + str(brushThickness), (6, hCam//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 250, 250), 1)
    cv2.imshow("Web cam", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break