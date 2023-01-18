import cv2
from matplotlib.style import library
import numpy as np
import math
from matplotlib import pyplot as plt
from copy import deepcopy
import random

def G_noise(img, s):
    m, n = img.shape
    noise = np.random.normal(0, s**2, m * n)
    noise = noise.reshape(m, n) # make Gaussian noise ( m * n ) 
    return noise
    
def Psnr(img1, img2):
    m, n = img1.shape
    
    mse = 0

    for i in range(m):
        for j in range(n):
            mse = mse + ((img1[i][j] - img2[i][j]) ** 2.0) # sum of square errors 
    mse = mse / (m * n) # definition of mean square error
	return 20 * math.log10(255 / math.sqrt(mse)) # scaling mse to log scale

def G_filter(k, sigma, v):

    assert k % 2 == 1, 'kernel_size must be odd' # filter must be odd_size
    
    n = int((k - 1) / 2) # number to get range of filter
    l = []
    
    for i in range(-1 * n, n + 1):
        for j in range(-1 * n, n + 1):
            c = -1 * ( ( i**2 + j**2 ) / (2 * sigma**2) ) 
            # definition of Gaussian filter
            l.append(c)
               
    gaussian = np.exp(l).reshape(-1, k) # take exponential
    sum = gaussian.sum() # normalization factor

    if v == 1:
        return gaussian / sum # version #1
    if v == 2:
        return gaussian # version #2

def padding(img, n):

    x, y = img.shape # original image size

    padding_img = np.zeros((x+2*n, y+2*n)) # consider up,down,left,right

    padding_img[n:x+n, n:y+n] = img # zero padding

    return padding_img

def conv(img, filter):
   
    k = len(filter) # kernel size
    n = int((k - 1) / 2) # padding number
    x, y = img.shape # original image size

    a = padding(img, n)

    for i in range(n, x+n):
        for j in range(n, y+n):
            a[i][j] = np.multiply(a[i-n:i+n+1, j-n:j+n+1], filter).sum() # convolution operation
     
    return a[n:x+n, n:y+n] # return result image ,except the padding part

def Bi_filtered(img, k, sigma_s, sigma_r):
    
    n = int((k - 1) / 2) # padding number
    x, y = img.shape

    G = G_filter(k, sigma_s, 2) # get Gaussian filter

    a = padding(img, n) # padding

    for i in range(n, x+n):
        for j in range(n, y+n):

            b = Bi(a, k, sigma_r, i, j) # get Bilateral filter at present position

            filter = np.multiply(G, b) # multiply Gaussian and Bilateral
            filter = filter / filter.sum() # normalization

            a[i][j] = np.multiply(a[i-n:i+n+1, j-n:j+n+1], filter).sum() # convolution operation
    
    return a[n:x+n, n:y+n]

def Bi(img, k, sigma_r, x, y):

    n = int((k - 1) / 2) # number to get range of filter    

    l = []

    for i in range(-1 * n, n + 1):
        for j in range(-1 * n, n + 1):
            c = -1 * ( ((img[x][y] - img[x+i][y+j])**2) / (2 * (sigma_r)**2) ) 
            # definition of Bilateral
            l.append(c)

    bilateral = np.exp(l).reshape(-1, k)
    
    return bilateral

def conv(img, filter):
   
    k = len(filter) # kernel size
    n = int((k - 1) / 2) # padding number
    x, y = img.shape # original image size

    a = padding(img, n)

    for i in range(n, x+n):
        for j in range(n, y+n):
            a[i][j] = np.multiply(a[i-n:i+n+1, j-n:j+n+1], filter).sum() # convolution operation
     
    return a[n:x+n, n:y+n] # return result image ,except the padding part


imageFile = '/home/youraddress/image_project1/lena.bmp'
img1 = cv2.imread(imageFile, 0)

noise = G_noise(img1, 5)

img2 = img1 + noise

img3 = conv(img2, G_filter(7,5,1))

img4 = Bi_filtered(img2,7,50,50)



plt.subplot(2,2,1)
plt.imshow(img1, vmin=0, vmax=255, cmap='gray')


plt.subplot(2,2,2)
plt.imshow(img2, vmin=0, vmax=255, cmap='gray')


plt.subplot(2,2,3)
plt.imshow(img3, vmin=0, vmax=255, cmap='gray')

plt.subplot(2,2,4)
plt.imshow(img4, vmin=0, vmax=255, cmap='gray')

plt.show()

print('Gaussian psnr:', Psnr(img1,img3))
print('Bilateral psnr:', Psnr(img1,img4))
