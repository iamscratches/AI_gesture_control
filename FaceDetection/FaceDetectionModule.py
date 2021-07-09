import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * self.width), int(bboxC.ymin * self.height), \
                       int(bboxC.width * self.width), int(bboxC. height * self.height)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return img, bboxs

    def resize(self, img, scale_percent):
        h, w, c = img.shape
        # scale_percent = 30  # percent of original size
        self.width = int(w * scale_percent / 100)
        self.height = int(h * scale_percent / 100)
        dim = (self.width, self.height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def fancyDraw(self, img, bbox, l=30, t=5, rt=2):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 255, 0), rt)
        cv2.line(img, (x,y), (x+l,y), (255,255,0), t)
        cv2.line(img, (x, y), (x, y+l), (255, 255, 0), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 255, 0), t)
        return img

def main():
    # cap = cv2.VideoCapture("../videos/video11.mp4")
    cap = cv2.VideoCapture(0)

    success, img = cap.read()
    pTime = 0

    detector = FaceDetector()

    while True:

        img = detector.resize(img, 100)
        img, bboxs = detector.findFaces(img)
        # print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "fps: " + str(int(fps)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 255, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

        success, img = cap.read()


if __name__ == '__main__':
    main()