import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture("IGVC 2015 UNSW Advanced Course GoPro - Speed Record.mp4")




while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray,9,75,75)
    edges = cv2.Canny(blur,400,500)
    # Display the resulting frame
    #cv2.imshow('frame', blur)
    cv2.imshow('frame2', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()