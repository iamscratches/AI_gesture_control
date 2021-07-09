import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = 0.5, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findMeshMesh(self, img, width, height, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs, self.drawSpecs)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x*width), int(lm.y*height)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 210, 210), 1)
                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces



def main():
    # cap = cv2.VideoCapture("../videos/video9.mkv")
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    pTime = 0
    h, w, c = img.shape
    detector = FaceMeshDetector()

    while True:
        scale_percent = 130  # percent of original size
        width = int(w * scale_percent / 100)
        height = int(h * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img, faces = detector.findMeshMesh(img, width, height)

        if len(faces)!=0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "fps: " + str(int(fps)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (110, 210, 210), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

        success, img = cap.read()




if __name__ == '__main__':
    main()
