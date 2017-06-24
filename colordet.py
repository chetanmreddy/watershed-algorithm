# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:22:34 2017

@author: chetan
"""

import numpy as np
import cv2

image = cv2.imread('test.jpg')
lower = [17, 15, 100]
upper = [50, 56, 200]
lower=np.array(lower)
upper=np.array(upper)
 
	# find the colors within the specified boundaries and apply
	# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)
 
	# show the images
cv2.imshow("images", output)
cv2.waitKey(0)