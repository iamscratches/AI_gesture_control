import cv2
import uuid
import os
import time

labels = ["thumbsup", "thumbsdown", "thankyou", "livelong"]
number_imgs = 5

IMAGES_PATH = os.path.join('.', 'Tensorflow', 'workspace', 'images', 'collected_images')

for label in labels:
    cap = cv2.VideoCapture(0)
    print("Collecting images for {}".format(label))
    start = time.time()
    while True:
        if time.time() - start > 5:
            break
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for imgnum in range(number_imgs):
        print('collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame2', frame)
        start = time.time()
        while True:
            if time.time() - start > 2:
                break
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


