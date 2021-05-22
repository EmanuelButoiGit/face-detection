# face clasifier sunt antrenate cu imagii ce contin fete

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


cap = cv2.VideoCapture('video.mp4')

# pt mp4
while cap.isOpened():
    _, img = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # viteza de redare 1.2
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)  # 4=vecini (mean)
    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)



    roi_gray = gray[y:y + height, x:x + width]
    roi_color = img[y:y + height, x:x + width]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    nose = nose_cascade.detectMultiScale(roi_gray)
    for (nx, ny, nw, nh) in nose:
        if nw > 50:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

    cv2.imshow('Detector', img)
    # cv2.waitKey() #fara asta la video

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cap.release
