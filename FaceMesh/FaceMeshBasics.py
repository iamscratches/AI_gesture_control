import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# cap = cv2.VideoCapture("../videos/video9.mkv")
cap = cv2.VideoCapture(0)
success, img = cap.read()
pTime = 0
h, w, c = img.shape

while True:
    scale_percent = 130  # percent of original size
    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpecs, drawSpecs)
            for id, lm in enumerate(faceLms.landmark):
                x, y = int(lm.x*width), int(lm.y*height)
                print(id, x, y)

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