import cv2
import time
import numpy as np
import math
from HandTracking import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

'''
###########################
        CONTROLS
thumb, index and pinky finger up other fingers down - volume control, move thumb and index closer and apart to change volume
###########################
'''

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detectionCon=0.9)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 0
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) > 0:
        fingerStatus = detector.fingersUp()

    if len(lmList)>0 and fingerStatus[2] == 0 and fingerStatus[3] == 0 and fingerStatus[4]:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 7, (0, 0, 250), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (0, 0, 250), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 250, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 250), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        if length <= 20:
            cv2.circle(img, (cx, cy), 5, (0, 250, 0), cv2.FILLED)

        # Hand range = 15 to 150
        # Volume range = -63.5 to 0
        vol = int(np.interp(length, [15, 150], [minVol, maxVol]))
        volBar = int(np.interp(length, [15, 150], [400, 150]))
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
    vol = volume.GetMasterVolumeLevel()
    volBar = np.interp(vol, [minVol, maxVol], [400, 150])
    volPer = int(np.interp(vol, [minVol, maxVol], [0, 100]))
    cv2.rectangle(img, (50, 150), (65, 400), (0, 100, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (65, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # print(vol, volBar)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 250, 250), 2)
    cv2.imshow("Web cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
