import cv2
import mediapipe as mp
import time


class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, height, width, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('../videos/video12.mp4')
    pTime = 0
    detector = PoseDetector()
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
            cv2.circle(img, (lmList[10][1], lmList[10][2]), 5, (255, 0, 0), cv2.FILLED)
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


if __name__ == '__main__':
    main()
