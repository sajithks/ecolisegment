# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:56:22 2013

@author: root
"""
import numpy as np
from criticalpoints import *
from matplotlib import pyplot as plt

def gauss_kern(Img):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h2,h1 = Img.shape    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 5.
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
    return g / g.sum()

#%%
img = drawCircle(np.ones((100,100))*200,(45,45),20,5)
img = img*np.random.randn(img.shape[0],img.shape[1])
gau = gauss_kern(img)
#Img_smooth = signal.convolve(Img,g,mode='same')
Imgfft = np.fft.rfft2(img)
gfft = np.fft.rfft2(gau)
fftimage = np.multiply(Imgfft, gfft)
img_smooth = np.real(np.fft.ifftshift( np.fft.irfft2(fftimage)))
Iy, Ix = np.abs( np.gradient(img_smooth))
x = 45 + 8*np.cos(np.arange(0, 2*np.pi, 0.1))
x = np.int32(np.round(x))
y =  45 + 5*np.sin(np.arange(0, 2*np.pi,0.1))
y = np.int32(np.round(y))

myshow(img)
plt.hold(True)
plt.plot(x, y)
#plt.hold(False)

#plt.contour(x,y,img),plt.show()

alpha = 0.001
beta = 0.4
gamma = 100
iterations = 50

N = len(x)
a = gamma * (2*alpha + 6 * beta) + 1
b = gamma * (-alpha - 4 * beta) 
c = gamma * beta

P = np.diag(np.tile(a,N),0)
P = P + np.diag(np.tile(b,N-1),1) 
P = P + np.diag(np.tile(b,N-1),-1)
P = P +  np.diag(np.tile(c,N-2),2)
P = P +  np.diag(np.tile(c,N-2),-2)
P[0, N-1] = b
P[N-1, 0] = b

P[N-2, 0] = c
P[N-1, 1] = c
P[0, N-2] = c
P[1, N-1] = c

P[0, N-3:N-1] = c

P = np.linalg.inv(P)

for ii in range(iterations):
    fex = Ix[x,y]
    fey = Iy[x,y]
    x = np.int32(np.round(np.dot(P, x + gamma * fex)))
    y = np.int32(np.round(np.dot(P, y + gamma * fey)))

plt.plot(x, y)














