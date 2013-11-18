    
def segbac(img):    
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
#    from scipy.ndimage.morphology import *
    from scipy.spatial import ConvexHull
    import scipy as sp
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
                                                
    #%%
    #take output from ilastik
                                                
    #img = plt.imread('img__000000000_125_000_processed_segmentation.png')
    img = img == img.max()
    
    img = sp.ndimage.filters.median_filter(img,(3,3))
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
    
    savimg = np.zeros((row, col, 3), labelimg.dtype)
    for iteration in range(3):
        np.random.shuffle(index)
        for ii in np.arange(1, ncc):
            randlabel[temprandlabel == ii] = index[ii]
        savimg[:, :, iteration] = np.copy(randlabel)
    return(savimg)        
    
    
    
    
    
    
    
    
    
    
    
