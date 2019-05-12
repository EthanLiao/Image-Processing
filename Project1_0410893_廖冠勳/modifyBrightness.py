from __future__ import print_function
#from builtins import input
import cv2 as cv
import numpy as np
import argparse
import sys
from matplotlib import pyplot as plt

import numpy as np
#normalize histogram
def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape),cdf

# Read image given by user
imagename=sys.argv[1]
image = cv.imread(imagename)
if image is None:
    print('Could not open or find the image: ',imagename)
    exit(0)
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0 # Simple contrast control
beta = 0    # Simple brightness control
# Initialize values
print(' Basic Linear Transforms ')
print('-------------------------')
try:
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
except ValueError:
    print('Error, not a number')
# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
normalize_image,cdf2=histeq(image,256)
cv.imshow('Original Image', image)
cv.imshow('New Image', new_image)
cv.imshow('Normalize Image', normalize_image)
plt.hist(new_image.ravel(),256,[0,256])
plt.show()
plt.hist(image.ravel(),256,[0,256])
plt.show()
plt.hist(normalize_image.ravel(),256,[0,256])
plt.show()
# Wait until usesr press some key
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
