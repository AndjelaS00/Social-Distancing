# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:46:58 2024

@author: Andjela
"""

# Import opencv library
import cv2
 
# Read video frames
cap = cv2.VideoCapture('C:/Users/Andjela/Desktop/SocialDistancing/data/cctv.mp4')

 
while True:
    # Read video frame by frame
    read_ok, img = cap.read()
    if not read_ok:
       print("Cannot read frame, exiting...")
       break
    cv2.imshow("Image", img)
 
    # Save last frame of the video as image using OpenCV
    cv2.imwrite("last_frame.jpg", img)
 
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
