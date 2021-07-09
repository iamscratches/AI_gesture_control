import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# cap = cv2.VideoCapture("../videos/video11.mp4")
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
    results = faceDetection.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score, detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                   int(bboxC.width * width), int(bboxC. height * height)
            cv2.rectangle(img, bbox, (255, 255, 0), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps: " + str(int(fps)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

    success, img = cap.read()
