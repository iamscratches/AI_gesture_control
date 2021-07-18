# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import cv2
cap = cv2.VideoCapture(0)
import time

start = time.time()

while True:
    diff = 5 - time.time() + start
    ret, frame = cap.read()
    cv2.putText(frame, f'{int(diff)}', (240, 340), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10)
    cv2.imshow('frame', frame)
    if diff <= 0:
        start = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break