from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2



image = cv2.imread('test.jpeg')
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.imshow("Input", image)


gray_image = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray_image, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)


D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
    labels=thresh)


markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))


for label in np.unique(labels):
    if label == 0:
        continue

'''
here the background is made black
so we are comparing it with the '0'pixel intensity
since it represents black in grayscale
'''
    mask = np.zeros(gray_image.shape, dtype="uint8")
    mask[labels == label] = '#000080'


    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]



    cv2.drawContours(img, cnts, -1, (0,255,0), 3,
        cv2.FONT_HERSHEY_SIMPLEX,)