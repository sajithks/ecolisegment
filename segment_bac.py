# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:44:08 2013

@author: root
"""

import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal as signal
import time
#from evolution import *
import cv2
from scipy.ndimage import label
import skimage.morphology as skmorph
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import *
from scipy.spatial import ConvexHull
import scipy as sp
from matplotlib.mlab import PCA as pca
import gaussianRectangle
##########
    
def myshow(img):
    
    def onClick(event):
        print(img[event.ydata,event.xdata])     
        #    plt.close('all')
    plt.figure()    
    plt.ion() 
    plt.imshow(img,cmap='gray'),plt.show()
    fig = plt.gcf()
    #    on mouse click the value at the image location is displayed in output screen
    _ = fig.canvas.mpl_connect('button_press_event', onClick)

def myshow2(img):
    
    def onClick(event):
        print(img[event.ydata,event.xdata])     
        #    plt.close('all')
    plt.figure()    
    plt.ion() 
    plt.imshow(img),plt.show()
    fig = plt.gcf()
    #    on mouse click the value at the image location is displayed in output screen
    _ = fig.canvas.mpl_connect('button_press_event', onClick)
    
def gauss_kern(Img):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h2,h1 = Img.shape    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2
    sigma = 17.5
    g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) )
    return g / g.sum()
    
def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov     


def plot_bars(x_bar, y_bar, cov, ax):
    """Plot bars with a length of 2 stddev along the principal axes."""
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle =np.pi/2
    #    eigvecs[0] = np.dot(np.array([[np.cos(angle), np.sin(angle)],[np.sin(angle),np.cos(angle)]]), eigvecs[0])
    #    eigvecs[1] = np.dot(np.array([[np.cos(angle), np.sin(angle)],[np.sin(angle),np.cos(angle)]]), eigvecs[1])
    print(eigvals)
    print(eigvecs)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, 1), marker='o', color='red')
    ax.axis('image')

def principalComponents(matrix):
    # Columns of matrix correspond to data points, rows to dimensions.

    deviationMatrix = (matrix.T - np.mean(matrix, axis=1)).T
    covarianceMatrix = np.cov(deviationMatrix)
    eigenvalues, principalComponents = np.linalg.eig(covarianceMatrix)

    # sort the principal components in decreasing order of corresponding eigenvalue
    #    indexList = np.argsort(-eigenvalues)
    #    eigenvalues = eigenvalues[indexList]
    #    principalComponents = principalComponents[:, indexList]

    return eigenvalues, principalComponents                                       
#%%
#take output from ilastik
                                    
orimg = plt.imread('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/img__000000000_125_000.tiff')                                    
img = plt.imread('img__000000000_125_000_processed_segmentation.png')
img = img == img.max()

img = sp.ndimage.filters.median_filter(img,(3,3))
img = skmorph.remove_small_objects(img, 500)
#%% creating structuring element
#strel = np.ones((3, 3))
#diamond = strel
#diamond[0, 0] = diamond[0, 2] = diamond[2, 0] = diamond[2, 2] = 0
##diamond = strel
##diamond[3,:] = diamond[:,3] = 1
##diamond[1:6, 2:5] = diamond[2:5, 1:6] = 1
##disk = strel
##disk[3,:] = disk[:,3] = 1
##disk[1:6, 1:6] = 1 
#
##%%
#disk = np.ones((3,3))
#img = binary_closing(img, diamond)
#img = binary_closing(img, disk)
#img = binary_opening(img, disk)
#img = binary_opening(img, diamond)
#img = binary_fill_holes(img)
labelimg, ncc = label(img)
#myshow2(labelimg)

index = np.arange(1,ncc+1)
np.random.shuffle(index)
randlabel = np.zeros((labelimg.shape), labelimg.dtype)
for ii in np.arange(1, ncc):
    randlabel[labelimg == ii] = index[ii]
#hull = ConvexHull(labelimg==164)
temprandlabel = np.copy(randlabel)
winsize = 1
row, col = randlabel.shape
for iteration in range(3):
    for ii in np.arange(0 + winsize, row - winsize):
        for jj in np.arange(0 + winsize, col - winsize):
            window = randlabel[ii - winsize: ii + winsize +1, jj - winsize: jj + winsize + 1]
            if (window[winsize, winsize] == 0):
                temprandlabel[ii, jj] = window[np.unravel_index(np.argmax(window),(window.shape))]
    randlabel = np.copy(temprandlabel)            


#myshow2(temprandlabel)       
#myshow2(randlabel)

#savimg = np.zeros((row, col, 3), labelimg.dtype)
#for iteration in range(3):
#    np.random.shuffle(index)
#    for ii in np.arange(1, ncc):
#        randlabel[temprandlabel == ii] = index[ii]
#    savimg[:, :, iteration] = np.copy(randlabel)


#x_bar, y_bar, cov = intertial_axis(data)      
##
#
#coord = np.argwhere(data == True)
#plt.ion()
#plt.figure()
#plt.scatter(coord[:, 1]-coord[:, 1].min(), coord[:, 0]-coord[:, 0].min())
##val = np.max([coord[:,1].max()-coord[:,1].min(),coord[:,0].max()-coord[:,0].min()])
#plt.axis([0, 200, 200, 0]),plt.show()
#
#ncoord = np.zeros((coord.shape),'int32')
#ncoord[:, 0] = coord[:, 0] - coord[:, 0].min()
#ncoord[:, 1] = coord[:, 1] - coord[:, 1].min()
#cropreg = np.zeros((200, 200), 'float32')
#cropreg[ncoord[:, 0],ncoord[:, 1]] = 1
#u, s, v = np.linalg.svd(cropreg)
#
#s[s<10.]=0
#myshow2(np.abs(np.dot(np.dot(u,np.diag(s)),v)))

#%% Hough transform of data
#boundaryimg = data - binary_erosion(data)
#accmatrix = np.zeros((np.sqrt(row**2+col**2), 90))
#rowcolval =  np.argwhere(boundaryimg)
#
#for ii in range(rowcolval.shape[0]):
#    for jj in np.arange(0, 90):
#        rval = np.sqrt(np.int32(rowcolval[ii, 0]*np.cos(jj*np.pi/180))**2 + (rowcolval[ii, 1]*np.sin(jj*np.pi/180))**2)  
#        accmatrix[rval, jj] = accmatrix[rval, jj] + 1
#    
#maxpoints = np.argwhere(accmatrix>=.9*accmatrix.max())
#for jj in range(maxpoints.shape[0]):
#    yval = []
#    for ii in range(col):
#        yval.append((maxpoints[jj,0]- ii*np.cos(maxpoints[jj,1]*(np.pi/180)))/np.sin(maxpoints[0,1]*(np.pi/180)))
#yval = np.array(yval)
#xval = np.argwhere(np.logical_and(yval>0.0, yval<row))
#yval = np.int32(yval[np.logical_and(yval>0.0, yval<row)])
#data[xval, yval.T] = 1
#    data[]
data = temprandlabel == 66
endpt =[]
skeldata = skmorph.skeletonize(data)
skelcoord = np.argwhere(skeldata)
for ii in range(skelcoord.shape[0]):
    #    print(skeldata[skelcoord[ii][0]-1:skelcoord[ii][0] + 2, skelcoord[ii][1] - 1 : skelcoord[ii][1] + 2].sum() )
    if(skeldata[skelcoord[ii][0]-1:skelcoord[ii][0] + 2, skelcoord[ii][1] - 1 : skelcoord[ii][1] + 2].sum() == 2):
        endpt.append([skelcoord[ii][0], skelcoord[ii][1]])
endpt = np.array(endpt)    
#dicescore = gaussianRectangle.compare(endpt[2], endpt[3], data)

accsize = gaussianRectangle.findSum(endpt.shape[0])
dicescore = np.zeros((accsize))
imgregion = np.zeros((200, 200, accsize))
coverage = np.zeros((accsize))
areabyconvexarea = np.zeros((accsize))
imgarea = np.zeros((accsize))
imgcoordinate = []
tempcounter = 0
for ii in range(endpt.shape[0] - 1):
    for jj in np.arange(ii + 1, endpt.shape[0]):
        print(ii, jj)
        dicescore[tempcounter], imgregion[:, :, tempcounter], coverage[tempcounter], tempval = gaussianRectangle.compare(endpt[ii], endpt[jj], data)
        imgcoordinate.append(tempval)        
        tempcounter += 1

# filtering the object parts

for ii in range(tempcounter):
    labelimg, nums = label(imgregion[:,:,ii] > 0)
    imgarea[ii] = np.argwhere(imgregion[:,:,ii]>0).shape[0]
    if(nums > 1 or imgarea[ii] < 500):
        dicescore[ii] = 0
    convarea = np.argwhere(skmorph.convex_hull_image(imgregion[:,:,ii]>0)).shape[0]
    areabyconvexarea[ii] = np.float64(imgarea[ii])/convarea

#imgdatasegment = np.zeros((img.shape),np.uint8)
#imgdatasegment = np.copy(data)
#for ii in range(accsize):
#    imgdatasegment[imgcoordinate[ii][:,0], imgcoordinate[ii][:,1]] = imgdatasegment[imgcoordinate[ii][:,0], imgcoordinate[ii][:,1]] +1

#eigval, eigvec  = principalComponents(np.argwhere(imgpart1))
#print(eigval[0:5])

#dt = cv2.distanceTransform(data, 2, 3)


#
#min0 = coord[:,0].min()
#min1 = coord[:,1].min()
#for ii in range(coord.shape[0]):
#    coord[ii,0] = coord[ii,0] - min0
#    coord[ii,1] = coord[ii,1] - min1
#plt.plot(coord,'r . '),plt.show()
#covmat = np.cov([coord[:,0],coord[:,1]])
#eigvals, eigvecs = np.linalg.eig(covmat)
#xval0 = []
#yval0 = []
#xval1 = []
#yval1 = []
#
#for ii in range(np.int32(np.sqrt(eigvals[0]))):
#    xval0.append(x_bar + ii)
#    yval0.append(y_bar + ii * (eigvecs[0][0]/eigvecs[0][1]))
#
#for ii in range(np.int32(np.sqrt(eigvals[1]))):
#    xval1.append( x_bar + ii)
#    yval1.append(y_bar + ii * (eigvecs[1][0]/eigvecs[1][1]))
#
#plt.plot(xval0,yval0,'g-',xval1,yval1,'r-')

#plt.ion()
#fig, ax = plt.subplots()
#ax.imshow(data)
#plot_bars(x_bar, y_bar, covmat, ax)
#plt.show()




