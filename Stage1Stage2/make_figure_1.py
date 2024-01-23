#Load in astropy, numpy and matplotlib 
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
#import lognorm
from matplotlib.colors import LogNorm, SymLogNorm

# Import CMasher to register colormaps
import cmasher as cmr

import glob

#######################################################
########## Read in the two first roll angles ##########
#######################################################

datadir_f300m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"
datadir_f360m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/"


sci1_filename_f300m = datadir_f300m+"jw01386117001_03106_00001_nrcalong_calints.fits"
sci1_hdul_f300m = fits.open(sci1_filename_f300m)
#Median collapse the first dimension
sci1_data_f300m = np.median(sci1_hdul_f300m[1].data,axis=0)


sci1_filename_f360m = datadir_f360m+"jw01386117001_03107_00001_nrcalong_calints.fits"
sci1_hdul_f360m = fits.open(sci1_filename_f360m)
#Median collapse the first dimension
sci1_data_f360m = np.median(sci1_hdul_f360m[1].data,axis=0)

#######################################################
# From the headers of each of these JWST image files grab the roll angle and the image center
#######################################################

#F300M
roll_angle_f300m = sci1_hdul_f300m[1].header['ROLL_REF']
x_center_f300m = sci1_hdul_f300m[1].header['CRPIX1']
y_center_f300m = sci1_hdul_f300m[1].header['CRPIX2']

#F360M
roll_angle_f360m = sci1_hdul_f360m[1].header['ROLL_REF']
x_center_f360m = sci1_hdul_f360m[1].header['CRPIX1']
y_center_f360m = sci1_hdul_f360m[1].header['CRPIX2']

#Read the pixel scale from the header
pixel_scale_f300m = np.sqrt(sci1_hdul_f300m[1].header['PIXAR_A2'])
pixel_scale_f360m = np.sqrt(sci1_hdul_f360m[1].header['PIXAR_A2'])

##################################
# Read in a reference star image #
##################################
datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"
ref_filenames = sorted(glob.glob(datadir+"jw01386116001_03106_0000?_nrcalong_calints.fits"))

#Read in the first reference star image - F300M
ref_hdul_f300M = fits.open(ref_filenames[0])
ref_data_f300M = np.median(ref_hdul_f300M[1].data,axis=0)

datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/"
ref_filenames = sorted(glob.glob(datadir+"jw01386116001_03108_0000?_nrcalong_calints.fits"))

#Read in the first reference star image - F360M
ref_hdul_f360M = fits.open(ref_filenames[0])
ref_data_f360M = np.median(ref_hdul_f360M[1].data,axis=0)



#### Convert all the data to uJy/arcsec^2 ####
#What are the units? 
#Convert from MJy/sr to uJy/arcsecond^2
conversion_factor = ref_hdul_f300M[1].header['PHOTUJA2']/ref_hdul_f300M[1].header['PHOTMJSR']

sci1_data_f300m = sci1_data_f300m*conversion_factor
sci1_data_f360m = sci1_data_f360m*conversion_factor
ref_data_f300M = ref_data_f300M*conversion_factor
ref_data_f360M = ref_data_f360M*conversion_factor

#######################################################
# Make a two-panel plot of the two images, where the x and y axes are 
# in units of delta arcseconds from the center of the image. Include a colorbar. 
# Use matplotlib axes objects. Use the pixel scale to get the pixel size in milliarcseconds.

#Coronagraphic center: 
coron_center = [173.4,149.1]

#F300M
x_size_f300m = sci1_data_f300m.shape[1]
y_size_f300m = sci1_data_f300m.shape[0]
x_f300m = np.arange(x_size_f300m)
y_f300m = np.arange(y_size_f300m)
x_f300m = (x_f300m-coron_center[1])*pixel_scale_f300m
y_f300m = (y_f300m-coron_center[0])*pixel_scale_f300m
xx_f300m,yy_f300m = np.meshgrid(x_f300m,y_f300m)

#F360M
x_size_f360m = sci1_data_f360m.shape[1]
y_size_f360m = sci1_data_f360m.shape[0]
x_f360m = np.arange(x_size_f360m)
y_f360m = np.arange(y_size_f360m)
x_f360m = (x_f360m-coron_center[1])*pixel_scale_f360m
y_f360m = (y_f360m-coron_center[0])*pixel_scale_f360m
xx_f360m,yy_f360m = np.meshgrid(x_f360m,y_f360m)

###########################################################
############ Plot the two images in a 2x2 grid ############
###########################################################


#### Plot Setup ####

#Use a SymLogNorm for the color scale
#Decide on a good linthresh for SymLogNorm, based on the data
linthresh = 1e-1

cmap = plt.get_cmap('cmr.freeze') 
# # cmap = plt.get_cmap('cmr.sapphire') #Worse than freeze
# cmap = plt.get_cmap('cmr.ember') #Worse than freeze
# cmap = plt.get_cmap('cmr.amber') #Worse than freeze
# cmap = plt.get_cmap('cmr.amethyst') #Worse than freeze
# cmap = plt.get_cmap('cmr.gothic') #Worse than freeze
# cmap = plt.get_cmap('cmr.sunburst') #Worse than freeze
# cmap = plt.get_cmap('cmr.voltage') #Worse than freeze


f300m_vmax = 1000*conversion_factor
f360m_vmax = 1000*conversion_factor
f300m_vmin = -10*conversion_factor
f360m_vmin = -10*conversion_factor

f300m_vmax_ref = 15000*conversion_factor
f300m_vmin_ref = -10*conversion_factor
f360m_vmax_ref = 15000*conversion_factor
f360m_vmin_ref = -10*conversion_factor

fig,ax = plt.subplots(2,2,figsize=(10,16))
ax = ax.flatten()


################
#### F300M #####
im0 = ax[0].imshow(sci1_data_f300m,
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   cmap=cmap,
                   norm=SymLogNorm(linthresh=linthresh,
                                   vmax = f300m_vmax,vmin=f300m_vmin),
                   origin='lower')
# im0 = ax[0].imshow(sci1_data_f300m,extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],)
ax[0].set_xlabel('x (mas)')
ax[0].set_ylabel('y (mas)')
ax[0].set_title('F300M - HD 141569A')
#Add colorbar - force it to be the same height as the image
from matplotlib import ticker
cbar = plt.colorbar(im0,ax=ax[0],label='uJy/arcsec$^2$',fraction=0.046)
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
# tick_locator = ticker.MaxNLocator(nbins=10)
# cbar.locator = tick_locator
# cbar.update_ticks()

# cbar = plt.colorbar(im0,ax=ax[0],label='uJy/arcsec$^2$')

###############
#### F360M ####

im1 = ax[1].imshow(sci1_data_f360m,
                   extent=[x_f360m[0],x_f360m[-1],y_f360m[0],y_f360m[-1]],
                   cmap=cmap,
                   norm=SymLogNorm(linthresh=linthresh,vmax = f360m_vmax,
                                   vmin = f360m_vmin),
                   origin='lower')
ax[1].set_xlabel('x (mas)')
ax[1].set_ylabel('y (mas)')
ax[1].set_title('F360M - HD 141569A')
#Add colorbar
cbar = plt.colorbar(im1,ax=ax[1],label='uJy/arcsec$^2$',fraction=0.046)
for label in cbar.ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
######################
#### F300M - Ref #####

im2 = ax[2].imshow(ref_data_f300M,
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   cmap=cmap,
                   norm=SymLogNorm(linthresh,
                                   vmax = f300m_vmax_ref,
                                   vmin = f300m_vmin_ref),
                   origin='lower')
ax[2].set_xlabel('x (mas)')
ax[2].set_ylabel('y (mas)')
ax[2].set_title('F300M - PSF Reference')
#Add colorbar
cbar = plt.colorbar(im2,ax=ax[2],label='uJy/arcsec$^2$',fraction=0.046)
for label in cbar.ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)

######################
#### F360M - Ref #####

im3 = ax[3].imshow(ref_data_f360M,
                   extent=[x_f360m[0],x_f360m[-1],y_f360m[0],y_f360m[-1]],
                   cmap=cmap,
                   norm=SymLogNorm(linthresh,
                                   vmax = f360m_vmax_ref,
                                   vmin = f360m_vmin_ref),
                   origin='lower')
ax[3].set_xlabel('x (mas)')
ax[3].set_ylabel('y (mas)')
ax[3].set_title('F360M - PSF Reference')
#Add colorbar
cbar = plt.colorbar(im3,ax=ax[3],label='uJy/arcsec$^2$',fraction=0.046)
for label in cbar.ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)

plt.tight_layout()
plt.show()











