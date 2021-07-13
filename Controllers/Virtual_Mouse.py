import math
import cv2
import numpy as np
from HandTracking import HandTrackingModule as htm
import time
import autopy

'''
###########################
        CONTROLS
Single index finger up - move the cursor
index and ring finger up and distant - get ready for a left click
close gap between index and ring finger up - make a left click
index, ring and thumb up and distant - get ready for a right click
close gap between index and ring finger up along with thumb up - make a right click
only index and thumb up - make a selection by holding the left click
index up and thumbs down - release the hold on the left click 
###########################
'''
##########################
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 3
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
pressed = False
# print(wScr, hScr)

while True:

    # 1. Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingerStatus = detector.fingersUp()

        # 4. Only Index Finger : Moving Mode
        if fingerStatus[1] == 1 and fingerStatus[2] == 0:
            cv2.circle(img, (lmList[8][1:]), 10, (255, 255, 255), cv2.FILLED)

            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR - 50, hCam - frameR - 50), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            if fingerStatus[0] and not pressed:
                print("pressed")
                pressed = True
                autopy.mouse.toggle(down=pressed)
            elif fingerStatus[0] == 0 and pressed:
                print("not pressed")
                pressed = False
                autopy.mouse.toggle(down=pressed)

            # 7. Move Mouse
            autopy.mouse.move(clocX, clocY)

            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up : Clicking Mode
        elif fingerStatus[1] == 1 and fingerStatus[2] == 1:
            cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            distance = math.hypot(x1-x2, y1-y2)
            if distance<20:
                cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
                if fingerStatus[0]:
                    cv2.circle(img, lmList[4][1:], 10, (255, 255, 255), cv2.FILLED)
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)
                else:
                    autopy.mouse.click()

    cv2.rectangle(img, (frameR, frameR-50), (wCam-frameR, hCam-frameR-50), (0, 0, 0), 2, cv2.FILLED)
    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break