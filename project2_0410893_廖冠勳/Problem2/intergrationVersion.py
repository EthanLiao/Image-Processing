import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import csv
from PIL import Image
import scipy.misc
import os
from scipy.ndimage import imread



def transtxtTobmp():
    blank_img = np.zeros((600,800,3), np.float64)
    file=open('periodic_noise.txt',"r").readlines()
    N=len(file)-1
    row=0
    col=0
    for i in range(0,N):
     set=file[i].split()
     for element in set:
        tmp=np.float64(element)
        if tmp > 255.0:
           tmp=255.0
        if tmp < 0:
           tmp=0
        blank_img[row,col]=tmp
        col+=1
        if col==800:
           row+=1
           col=0
    scipy.misc.imsave('periodic_noise_result.bmp', blank_img)

transtxtTobmp()
path = "periodic_noise_result.bmp"
img = Image.open(path)

def normalize(img):
    ''' Function to normalize an input array to 0-1 '''
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)



def gauss2D(shape=(3,3),sigma=0.1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


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


class Image() :
    def __init__(self) :
        self.image = None
        self.type = None
    def open(self,path) :
        img= cv2.imread(path)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.type = path.split(".")[-1]
    def set(self, image) :
        if len(image.shape) == 2 :
            self.image = image
    def show(self,name) :
        mode='Greys_r'
        plt.imshow(self.image,cmap=mode)
        scipy.misc.imsave(name, self.image)

class FourierTransform() :
    def __init__(self) :
        self.f = None
        self.F = None
        self.magnitude = None
        self.phase = None
        self.M = None
        self.N = None
        self.image = None
    def setImage(self, image) :
        if len(image.shape) == 2 :
           self.image = image
    def forwardTransform(self) :
        M = self.image.shape[0]
        N = self.image.shape[1]
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, -2j*np.pi*u*x/M)
        exp_2 = pow(np.e, -2j*np.pi*v*y/N)
        self.F = np.dot(exp_2, np.dot(exp_1,self.image).transpose())/(M*N)
        return self.F
    def inverseTransform(self) :
        M = self.F.shape[0]
        N = self.F.shape[1]
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, 2j*np.pi*u*x/M)
        exp_2 = pow(np.e, 2j*np.pi*v*y/N)
        self.f = np.dot(exp_2, np.dot(exp_1,self.F).transpose())
        return self.f
    def shift(self, image) :
        M = image.shape[0]
        N = image.shape[1]
        m = M/2
        n = N/2
        temp = np.zeros((M,N))
        temp[-m:,-n:] = np.abs(np.copy(image[:m,:n]))
        temp[-m:,:-n] = np.abs(np.copy(image[:m,n:]))
        temp[:-m,-n:] = np.abs(np.copy(image[m:,:n]))
        temp[:-m,:-n] = np.abs(np.copy(image[m:,n:]))
        return temp
    def error(self) :
        E = (self.image - self.f)**2
        M = E.shape[0]
        N = E.shape[1]
        I = np.ones((1,N))
        J = np.ones((M,1))

class ImageProcessing() :
    def __init__(self) :
        self.image = Image()
        self.fourierTransform = FourierTransform()
        self.laplacianLevels = []
    def readImage(self,path) :
        self.image.open(path)
    #def showImage(self) :
        #self.image.show()
    def computeFourierTransforms(self,resultbmp) :
        self.fourierTransform.setImage(self.image.image)
        fimg = Image()
        fimg.set(np.log(np.abs(self.fourierTransform.shift(self.fourierTransform.forwardTransform()))**2))
        fimg.show(resultbmp)




def getName():
	name = ''
	for i in range(6,len(path)):
		name = name+path[i]
	#print(name)
	return name
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




'''
Transfer the txt file to bmp file
'''

txtfile = 'periodic_noise.txt'
periodic_noise_img='periodic_noise_result.bmp'


'''
Plot the image in frequency domain
'''
original_freq_domain='origin_frequency.bmp'

imageProcessing = ImageProcessing()
imageProcessing.readImage(periodic_noise_img)
imageProcessing.computeFourierTransforms(original_freq_domain)


'''
Apply adaptive filter
'''

width, height = img.size
newimg = img.copy()

for i in range(5,1,-2):
	img = newimg.copy()
	#print(i)
	AdaptiveMedianFilter(i)
	if i==5:
		renoiseInBorder(2)

renoiseInBorder(1)
file_out = "after_adaptive.bmp"
if len(newimg.split()) == 4:
    r, g, b, a = newimg.split()
    newimg = Image.merge("RGB", (r, g, b))
    newimg.save(file_out)
else:
    newimg.save(file_out)


'''
Apply Gaussion filter to the image
'''
filter_result='filter_result.bmp'
test_image = mpimg.imread(file_out)
test_image = test_image.astype(np.single)/255

large_2d_blur_filter = gauss2D(shape=(7,7), sigma = 2.3)
large_blur_image = my_imfilter(test_image, large_2d_blur_filter);
scipy.misc.imsave(filter_result, normalize(large_blur_image + 0.5))


'''
Plot the image after filtering in frequency domain
'''
after_freq_domain='after_frequency.bmp'
rstimageProcessing = ImageProcessing()
rstimageProcessing.readImage(filter_result)
rstimageProcessing.computeFourierTransforms(after_freq_domain)
