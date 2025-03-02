# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:57:23 2024

@author: Andjela
"""

import cv2
import numpy as np
import math
 
# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((2,2),np.int32)
 
counter = 0
# Function to store coordinate of mouse click in image
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1
 
# Function for Euclidean Distance between two points
def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
 
# Loading image which was saved in previous step
img = cv2.imread("last_frame.jpg")
 
while True:
    for x in range (0,2):
        cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,(0,0,255),cv2.FILLED)
 
    # Stop when we have two mouse click
    if counter == 2:
        starting_x = point_matrix[0][0]
        starting_y = point_matrix[0][1]
 
        ending_x = point_matrix[1][0]
        ending_y = point_matrix[1][1]
 
        # Draw line between two mouse clicked points
        cv2.line(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), thickness=2)
 
        # Show length of the line on image
        line_length = calculateDistance(starting_x, starting_y, ending_x, ending_y)
 
        # Original width of a tile is 2 feet
        # So width of three tiles will be 6 feet in total
        font = cv2.FONT_HERSHEY_DUPLEX
        blue_color = (255,0,0)
        cv2.putText(img, f'{"Pixel Distance: ", round(line_length, 2)}', (starting_x - 25, starting_y + 70), font, 1, blue_color, 2)
        cv2.putText(img, f'{"Original: ", "6 ft"}', (starting_x - 25, starting_y + 100), font, 1, blue_color, 2)
 
    # Showing original image
    cv2.imshow("Original Image ", img)
    # Mouse click event on original image
    cv2.setMouseCallback("Original Image ", mousePoints)
    # Refreshing window all time
    cv2.waitKey(1)