import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        h, w, c = img.shape
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255,255,255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingerTips = [8, 12, 16, 20]
        fingerStatus = [1]*5
        minRange = min(self.lmList[3][1], self.lmList[17][1]) - 10
        maxRange = max(self.lmList[3][1], self.lmList[17][1]) + 10
        if self.lmList[4][1] in range(minRange, maxRange):
            fingerStatus[0] = 0

        for i in range(len(fingerTips)):
            if self.lmList[fingerTips[i]][2] > self.lmList[fingerTips[i]-2][2]:
                fingerStatus[i+1] = 0
        return fingerStatus



def main():
    pTime = cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList):
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        cv2.imshow("Web cam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    main()