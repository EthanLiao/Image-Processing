import cv2
import numpy as np
from scipy import ndimage
import scipy
import sys
import math
import scipy.misc
zero=0
def getVarAndMean(img,win_row,win_cols):
  '''Identifies & highlights areas of image fname with high pixel intensity variance (focus)'''
  sum=0
  for i in range(win_row) :
    for j in range(win_cols) :
        sum+=img[i][j]
  win_mean=sum/(win_row*win_cols)
  win_var=math.sqrt(ndimage.variance(img))

  return win_mean,win_var




def Statistic_filter(picture,kernel_shape):
      kernel_middle_len=(kernel_shape-1)/2
      row_index=kernel_middle_len
      col_index=kernel_middle_len
      #print picture.shape[0],picture.shape[1]
      while (row_index+kernel_middle_len)<=picture.shape[0]:
       col_index=kernel_middle_len
       while (col_index+kernel_middle_len)<=picture.shape[1]:
          window=np.empty((kernel_shape,kernel_shape))
          window.fill(zero)

          for i in range(row_index-kernel_middle_len,row_index+kernel_middle_len,1):
            for j in range(col_index-kernel_middle_len,col_index+kernel_middle_len,1):
              window[i-(row_index-kernel_middle_len)][j-(col_index-kernel_middle_len)]=picture[i][j]
          E=20.0
          K0=0.8
          K1=0.02
          K2=0.7
          win_mean,win_variance=getVarAndMean(window,kernel_shape,kernel_shape)
          print win_mean,win_variance
          img_mean,img_variance=getVarAndMean(picture,picture.shape[0],picture.shape[1])
          if win_mean<=K0*img_mean and K1*img_variance<=win_variance and win_variance<=K2*img_variance :
           for i in range(kernel_shape) :
            for j in range(kernel_shape) :
             #picture[row_index][col_index]=picture[row_index][col_index]*E
             picture[i+(row_index-kernel_middle_len)][j+(row_index-kernel_middle_len)]=window[i][j]*E

          col_index=col_index+1
          if col_index+kernel_middle_len>picture.shape[1] :
           row_index=row_index+1
           print row_index


      return picture


#imagename=sys.argv[1]
img = cv2.imread('problem3.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows,cols = gray.shape
kernel=331
result=Statistic_filter(gray,kernel)
scipy.misc.imsave('result_kernel331_10_0.8_0.6_0.5.bmp', result)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
