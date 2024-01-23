from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import cmasher as cmr # Import CMasher to register colormaps
import glob

#######################################################
########## Read in the two first roll angles ##########
#######################################################

datadir_f300m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"

sci1_filename_f300m = datadir_f300m+"jw01386117001_03106_00001_nrcalong_calints.fits"
sci1_hdul_f300m = fits.open(sci1_filename_f300m)
#Median collapse the first dimension
sci1_data_f300m = np.median(sci1_hdul_f300m[1].data,axis=0)

#######################################################
# From the headers of each of these JWST image files grab the roll angle and the image center
#######################################################

#F300M
roll_angle_f300m = sci1_hdul_f300m[1].header['ROLL_REF']
x_center_f300m = sci1_hdul_f300m[1].header['CRPIX1']
y_center_f300m = sci1_hdul_f300m[1].header['CRPIX2']

#Read the pixel scale from the header
pixel_scale_f300m = np.sqrt(sci1_hdul_f300m[1].header['PIXAR_A2'])

##################################
# Read in a reference star image #
##################################
datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"
ref_filenames = sorted(glob.glob(datadir+"jw01386116001_03106_0000?_nrcalong_calints.fits"))

#Read in the first reference star image - F300M
ref_hdul_f300M = fits.open(ref_filenames[0])
ref_data_f300M = np.median(ref_hdul_f300M[1].data,axis=0)

#### Convert all the data to uJy/arcsec^2 ####
#What are the units? 
#Convert from MJy/sr to uJy/arcsecond^2
conversion_factor = ref_hdul_f300M[1].header['PHOTUJA2']/ref_hdul_f300M[1].header['PHOTMJSR']

sci1_data_f300m = sci1_data_f300m*conversion_factor
ref_data_f300M = ref_data_f300M*conversion_factor

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

###########################################################
############ Plot the two images in a 2x2 grid ############
###########################################################

#### Plot Setup ####
#Use a SymLogNorm for the color scale
linthresh = 10

cmap = plt.get_cmap('cmr.freeze') 
# # cmap = plt.get_cmap('cmr.sapphire') #Worse than freeze
# cmap = plt.get_cmap('cmr.ember') #Worse than freeze
# cmap = plt.get_cmap('cmr.amber') #Worse than freeze
# cmap = plt.get_cmap('cmr.amethyst') #Worse than freeze
# cmap = plt.get_cmap('cmr.gothic') #Worse than freeze
# cmap = plt.get_cmap('cmr.sunburst') #Worse than freeze
# cmap = plt.get_cmap('cmr.voltage') #Worse than freeze


f300m_vmax = 100*conversion_factor
f300m_vmin = -10*conversion_factor

f300m_vmax_ref = 15000*conversion_factor
f300m_vmin_ref = -10*conversion_factor

fig,ax = plt.subplots(1,3,figsize=(12,4))
ax = ax.flatten()


################
#### F300M #####
im0 = ax[0].imshow(sci1_data_f300m,
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   cmap=cmap,
                   vmax = f300m_vmax,vmin=f300m_vmin,
                   origin='lower')
ax[0].set_xlabel('x (mas)')
ax[0].set_ylabel('y (mas)')
ax[0].set_title('F300M - HD 141569A')
#Add colorbar - force it to be the same height as the image
cbar = plt.colorbar(im0,ax=ax[0],label='uJy/arcsec$^2$',fraction=0.046)

# In Figure 1 let's add a compass rose that points to the north and east. 
# Here east is counterclockwise of north. The roll_angle_f300m variable defines the angle to point north. 
# Assuming roll_angle_f300m is defined and in degrees
north_angle = roll_angle_f300m + 270
east_angle = (roll_angle_f300m + 180) % 360  # East is counterclockwise of North

# Assuming ax is your axis object
arrow_length = 2.5
compass_center = [7.6,-6]
# Draw North arrow
ax[0].arrow(compass_center[0], compass_center[1], arrow_length * np.cos(np.radians(north_angle)), -arrow_length * np.sin(np.radians(north_angle)), head_width=0.5, head_length=0.5, fc='w', ec='w')
# Draw North label
ax[0].text(arrow_length * np.cos(np.radians(north_angle))+compass_center[0], -arrow_length * np.sin(np.radians(north_angle))+compass_center[1]+1, 'N', color='w', ha='center', va='center')

# Draw East arrow
ax[0].arrow(compass_center[0], compass_center[1], arrow_length * np.cos(np.radians(east_angle)), -arrow_length * np.sin(np.radians(east_angle)), head_width=0.5, head_length=0.5, fc='w', ec='w')
# Draw East label
ax[0].text(arrow_length * np.cos(np.radians(east_angle))+compass_center[0]-1, -arrow_length * np.sin(np.radians(east_angle))+compass_center[1], 'E', color='w', ha='center', va='center')

################
#### F300M #####
f300m_psf_max= np.max(sci1_data_f300m[155:185,135:165])
im0 = ax[1].imshow(sci1_data_f300m,
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   cmap=cmap,
                    norm=SymLogNorm(linthresh,
                   vmax = f300m_psf_max,vmin=f300m_vmin),
                   origin='lower')

ax[1].set_xlim(-5,5)
ax[1].set_ylim(-5,5)
ax[1].set_xlabel('x (mas)')
ax[1].set_ylabel('y (mas)')
ax[1].set_title('F300M - HD 141569A - Zoom')
#Add colorbar - force it to be the same height as the image
from matplotlib import ticker
cbar = plt.colorbar(im0,ax=ax[1],label='uJy/arcsec$^2$',fraction=0.046)

######################
#### F300M - Ref #####
f300m_ref_psf_max= np.max(ref_data_f300M[155:185,135:165])
im2 = ax[2].imshow(ref_data_f300M,
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   cmap=cmap,
                   norm=SymLogNorm(linthresh,
                                   vmax = f300m_ref_psf_max,
                                   vmin = f300m_vmin),
                   origin='lower')
ax[2].set_xlabel('x (mas)')
ax[2].set_ylabel('y (mas)')
ax[2].set_title('F300M - PSF Reference')
ax[2].set_xlim(-5,5)
ax[2].set_ylim(-5,5)
#Add colorbar
cbar = plt.colorbar(im2,ax=ax[2],label='uJy/arcsec$^2$',fraction=0.046)

plt.tight_layout()
plt.savefig("raw_data_f300m.png",dpi=300,bbox_inches="tight")
plt.show()










