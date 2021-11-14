import cv2
import matplotlib.pyplot as plt

from IrisLocalization import *
from IrisNormalization import *
from IrisEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from PerformanceEvaluation import *
import glob
imgs_train= [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('datasets/CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]

img_in = imgs_train[0]
# imgsub = img_in[60:220, 80:240]
# h_projection = np.mean(imgsub, axis=0)
# v_projection = np.mean(imgsub, axis=1)
# xp, yp = np.argmin(h_projection) + 60, np.argmin(v_projection) + 80
# if xp < 60:
#     xp = 60
# if yp < 60:
#     yp = 60
#
# img_sub = img_in[(yp-60):(yp+60), (xp-60):(xp+60)]
# img_blur = cv2.medianBlur(img_in,11)
# img_mask = cv2.inRange(img_blur, 0, 60)
# img_edge = cv2.Canny(img_mask, 100, 200)
# pupils = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 10, 100)
#
# # img_sub = img_blur[(yp-60):(yp+60), (xp-60):(xp+60)]
# # _, thres = cv2.threshold(img_sub,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # tight_p = cv2.Canny(thres, 30,20)
# # pupil = cv2.HoughCircles(tight_p, cv2.HOUGH_GRADIENT, 4, 300, minRadius=20, maxRadius=60)
#
# pupil = pupils[0]
# circle_pupil = pupil.astype(int)
# x_p_o,y_p_o,r_p = circle_pupil[0]
#
# output = img_in.copy()
# img = cv2.circle(output, (x_p_o, y_p_o), r_p, (255, 255, 255), 1)
#
# top = np.max([0, (y_p_o - 130)])
# bottom = np.min([280, (y_p_o + 130)])
# left = np.max([0, (x_p_o - 130)])
# right = np.min([320, (x_p_o + 130)])
# img_sub2 = img_in[top:bottom, left:right]
#
# blur = cv2.GaussianBlur(img_sub2, (7, 7), 0)
# # Use Edge detection (Canny Operator) to binarize the image
# # Reference: https://docs.opencv.org/3.4.15/d7/de1/tutorial_js_canny.html
# tight = cv2.Canny(blur, 30, 20)
# subx = int(x_p_o - left)
# suby = int(y_p_o - top)
# subr = int(r_p)
# img_sub3 = tight.copy()
# img_sub3[0:(suby + subr + 20), (subx - subr - 20):(subx + subr + 20)] = 0
# plt.imshow(img_sub3,cmap='gray')
# plt.show()

x_i, y_i, r_i, x_p, y_p, r_p = irisLocalization(img_in)
normalized = irisNormalization(x_i, y_i, r_i, x_p, y_p, r_p, img_in)
M, N = 64, 512
# approximate intensity variation

img_enhanced = irisEnhancement(normalized)
v = featureExtraction(img_enhanced,9)
plt.imshow(img_enhanced, cmap='gray')
plt.show()
normalized_rotated = imgRotate(normalized,6)
plt.imshow(normalized, cmap='gray')
plt.show()