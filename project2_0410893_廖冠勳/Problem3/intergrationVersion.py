import cv2
import numpy as np
from scipy import ndimage
import scipy
import sys
import math
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image


path = "problem3_4a.bmp"
img = Image.open(path)
width, height = img.size
newimg = img.copy()


def AdaptiveMedianFilter(sMax):
	filterSize = 3
	borderSize = sMax // 2
	imgMax = img.getpixel((0,0))
	mid = (filterSize*filterSize)//2
	for i in range(width):
	    for j in range(height):
	        if(imgMax < img.getpixel((i,j))):
	            imgMax = img.getpixel((i,j))

	for i in range(borderSize,width-borderSize):
	    for j in range(borderSize,height-borderSize):
	        members = [imgMax] * (sMax*sMax)
	        filterSize = 3
	        zxy = img.getpixel((i,j))
	        result = zxy
	        while(filterSize<=sMax):
	            borderS = filterSize // 2
	            for k in range(filterSize):
	                for t in range(filterSize):
	                    members[k*filterSize+t] = img.getpixel((i+k-borderS,j+t-borderS))
	            members.sort()
	            med  = (filterSize*filterSize)//2
	            zmin = members[0]
	            zmax = members[(filterSize-1)*(filterSize+1)]
	            zmed = members[med]
	            if(zmed<zmax and zmed > zmin):
	                if(zxy>zmin and zxy<zmax):
	                    result = zxy
	                else:
	                    result = zmed
	                break
	            else:
	                filterSize += 2

	        newimg.putpixel((i,j),(result))

def renoiseInBorder(borderSize):
	for i in range(1,width):
	    for j in range(borderSize):
	        newimg.putpixel((i,j),newimg.getpixel((i,borderSize)))
	        newimg.putpixel((i,height-j-1),newimg.getpixel((i,height-borderSize-1)))

	for j in range(height):
	    for i in range(borderSize):
	        newimg.putpixel((i,j),newimg.getpixel((borderSize,j)))
	        newimg.putpixel((width -i-1,j),newimg.getpixel((width-borderSize-1,j)))


def getVarAndMean(img,win_row,win_cols):
  sum=0
  for i in range(win_row) :
    for j in range(win_cols) :
        sum+=img[i][j]
  #win_mean=sum/(win_row*win_cols)
  win_mean=sum/3800
  win_var=math.sqrt(ndimage.variance(img))

  return win_mean,win_var

def my_imfilter(image, imfilter):

    output = image.copy()
    im_dim=image.shape
    flt_dim=imfilter.shape
    img_dim1=im_dim[0]
    img_dim2=im_dim[1]
    flt_dim1=flt_dim[0]
    flt_dim2=flt_dim[1]
    pad_dim1=int((flt_dim1-1)/2)
    pad_dim2=int((flt_dim2-1)/2)
    pad_mat=np.zeros((img_dim1+2*pad_dim1,img_dim2+2*pad_dim2,3))
    pad_mat[pad_dim1: img_dim1 + pad_dim1, pad_dim2: img_dim2 + pad_dim2] = image
    for d in range(len(image[0][0])):
        for i in range(len(image)):
            for j in range(len(image[0])):
                output[i][j][d] = sum(sum(np.multiply(imfilter,pad_mat[i:i+flt_dim1,j:j+flt_dim2,d])))

    return output


img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#global_var,global_mean=getVarAndMean(gray,gray.shape[0],gray.shape[1])
#print(gray.shape)
# create image mask
mask = np.zeros(gray.shape, np.uint8)
#mask[120:170, 390:450] = 255
#mask[60:80, 150:200] = 255
#mask[260:280, 170:360] = 255
mask[200:220, 60:250] = 255
masked_gray = cv2.bitwise_and(gray, gray, mask = mask)
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

mask_var,mask_mean=getVarAndMean(masked_gray,masked_gray.shape[0],masked_gray.shape[1])

print (mask_var,mask_mean)
#print (masked_gray.shape[0],masked_gray.shape[1])

plt.subplot(221), plt.imshow(gray, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_gray, 'gray')
plt.subplot(224), plt.plot(hist_mask)#plt.plot(hist_full)
plt.xlim([0,256])
plt.show()



for i in range(9,1,-2):
	img = newimg.copy()
	AdaptiveMedianFilter(i)
	if i==5:
		renoiseInBorder(2)


file_out="_result.bmp"
renoiseInBorder(1)
if len(newimg.split()) == 4:
    # prevent IOError: cannot write mode RGBA as BMP
    r, g, b, a = img.split()
    newimg = Image.merge("RGB", (r, g, b))
    newimg.save(path+file_out)
else:
    newimg.save(path+file_out)
