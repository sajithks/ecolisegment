import os
import segbac
from matplotlib import pyplot as plt
import numpy as np

filelist = os.listdir('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/classified/ilastikout')

for ii in np.arange(1, np.size(filelist)):
#for ii in np.arange(1, 2):    
    savimg = segbac.segbac(plt.imread(filelist[ii]))
    savimg = np.uint8(savimg)
#    myshow(savimg)
    savname = 'labeled_'+ filelist[ii]
    plt.imsave( savname, savimg)