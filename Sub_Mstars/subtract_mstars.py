from datetime import datetime
import glob 

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec
import cmasher as cmr # Import CMasher to register colormaps

# from nircam_disk_utils import * 

import jax.numpy as jnp
import numpy as np

from astropy.io import fits
#import minimize
from scipy.optimize import minimize

def read_fits(filename, sub_median = True,
              median_frames=True,
              median_region_x0=50,median_region_x1=270,
              median_region_y0=10,median_region_y1=75,
              return_error=True):
    '''
    Read in a fits file and clean it up a bit. 
    '''

    hdul = fits.open(filename)
    data = hdul[1].data
    data[~np.isfinite(data)]=0 #Clean up nans - jax does not like these
    data = data.astype(jnp.float32)

    if median_frames: #Median collapse all the frames
        data = jnp.median(data,axis=0) 

    if sub_median: #Subtract median background level
        data = data-jnp.median(data[median_region_y0:median_region_y1,
                                    median_region_x0:median_region_x1]) 
    

    if return_error: 
        err_data = hdul[2].data
        err_data[~np.isfinite(err_data)]=0 #Clean up nans
        if median_frames:
            err_data = jnp.sqrt(jnp.average(err_data.astype(jnp.float32)**2,axis=0))

        return data, err_data
    else: 
        return data

#### New data or old data
new=True
if new: 
    suffix = ""
    datadir_f300m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"
    datadir_f360m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/"
else: 
    suffix = "_older"
    datadir_f300m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/older/"
    datadir_f360m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/older/"

##############################################################
########## Read in the two F300M science data files ##########
##############################################################
sci1_filename = datadir_f300m+"jw01386117001_03106_00001_nrcalong_calints.fits"
sci2_filename = datadir_f300m+"jw01386118001_03106_00001_nrcalong_calints.fits"

sci1_hdul_f300m = fits.open(sci1_filename)
sci2_hdul_f300m = fits.open(sci2_filename)

roll_angle_f300m = sci1_hdul_f300m[1].header['ROLL_REF']

sci1_data_f300m,sci1_err_f300m = read_fits(sci1_filename)
sci2_data_f300m,sci2_err_f300m = read_fits(sci2_filename)

conversion_factor = sci1_hdul_f300m[1].header['PHOTUJA2']/sci1_hdul_f300m[1].header['PHOTMJSR']

sci1_data_f300m *= conversion_factor
sci2_data_f300m *= conversion_factor

data_shape = sci1_data_f300m.shape

#Read the pixel scale from the header
pixel_scale_f300m = np.sqrt(sci1_hdul_f300m[1].header['PIXAR_A2'])

##############################################################
########## Read in the two F360M science data files ##########
##############################################################
sci1_filename = datadir_f360m+"jw01386117001_03107_00001_nrcalong_calints.fits"
sci2_filename = datadir_f360m+"jw01386118001_03107_00001_nrcalong_calints.fits"

sci1_hdul_f360m = fits.open(sci1_filename)
sci2_hdul_f360m = fits.open(sci2_filename)

conversion_factor = sci1_hdul_f360m[1].header['PHOTUJA2']/sci1_hdul_f360m[1].header['PHOTMJSR']

sci1_data_f360m,sci1_err_f360m = read_fits(sci1_filename)
sci2_data_f360m,sci2_err_f360m = read_fits(sci2_filename)

sci1_data_f360m *= conversion_factor
sci2_data_f360m *= conversion_factor

#########################################
####### Read in the M-star Models #######
#########################################
m_star_dir = "/home/maxmb/Library/jwst_hd141569a_lib/mstar_models/"

print("Fitting F300M M-star models, Roll 1")
m_star_roll1_filename = "bcmodel_jw01386117001_03106_00001_nrcalong_calints.fits"
m_star_roll1_data_f300m = fits.open(m_star_dir+m_star_roll1_filename)[0].data.astype(jnp.float32)
m_star_roll1_data_f300m *= conversion_factor

print("Fitting F300M M-star models, Roll 2")
m_star_roll2_filename = "bcmodel_jw01386118001_03106_00001_nrcalong_calints.fits"
m_star_roll2_data_f300m = fits.open(m_star_dir+m_star_roll2_filename)[0].data.astype(jnp.float32)
m_star_roll2_data_f300m *= conversion_factor

print("Fitting F360M M-star models, Roll 1")
#Now the F360M M-stars Models
m_star_roll1_filename = "bcmodel_jw01386117001_03107_00001_nrcalong_calints.fits"
m_star_roll1_data_f360m = fits.open(m_star_dir+m_star_roll1_filename)[0].data.astype(jnp.float32)
m_star_roll1_data_f360m *= conversion_factor

print("Fitting F360M M-star models, Roll 2")
m_star_roll2_filename = "bcmodel_jw01386118001_03107_00001_nrcalong_calints.fits"
m_star_roll2_data_f360m = fits.open(m_star_dir+m_star_roll2_filename)[0].data.astype(jnp.float32)
m_star_roll2_data_f360m *= conversion_factor

### Make a mast out the saturated parts of the M-stars
# mask = sci1_data_f300m < 5000
mask = sci1_data_f300m < 5000

#######################################################################
####### Plot the starting guess - not scaling the M-star models #######
#######################################################################

plot_start = False
if plot_start: 
    f300m_vmax = 100*conversion_factor
    f300m_vmin = -10*conversion_factor
    linthresh = 10

    f360m_vmax = 100*conversion_factor
    f360m_vmin = -10*conversion_factor
    

    cmap = plt.get_cmap('cmr.freeze') 

    fig,ax = plt.subplots(2,4,figsize=(10,10))
    ax[0,0].imshow(sci1_data_f300m,norm=SymLogNorm(linthresh,
                   vmax = f300m_vmax,vmin=f300m_vmin),
                   origin='lower',cmap=cmap)
    ax[0,1].imshow(sci2_data_f300m,norm=SymLogNorm(linthresh,
                   vmax = f300m_vmax,vmin=f300m_vmin),
                   origin='lower',cmap=cmap)
    ax[0,2].imshow(sci1_data_f360m,norm=SymLogNorm(linthresh,
                   vmax = f360m_vmax,vmin=f360m_vmin),
                   origin='lower',cmap=cmap)
    ax[0,3].imshow(sci2_data_f360m,norm=SymLogNorm(linthresh,
                   vmax = f360m_vmax,vmin=f360m_vmin),
                   origin='lower',cmap=cmap)

    ax[1,0].imshow(sci1_data_f300m-m_star_roll1_data_f300m,norm=SymLogNorm(linthresh,
                   vmax = f300m_vmax,vmin=f300m_vmin),
                   origin='lower',cmap=cmap)
    ax[1,1].imshow(sci2_data_f300m-m_star_roll2_data_f300m,norm=SymLogNorm(linthresh,
                     vmax = f300m_vmax,vmin=f300m_vmin),
                     origin='lower',cmap=cmap)
    ax[1,2].imshow(sci1_data_f360m-m_star_roll1_data_f360m,norm=SymLogNorm(linthresh,
                        vmax = f360m_vmax,vmin=f360m_vmin),
                        origin='lower',cmap=cmap)
    ax[1,3].imshow(sci2_data_f360m-m_star_roll2_data_f360m,norm=SymLogNorm(linthresh,
                        vmax = f360m_vmax,vmin=f360m_vmin),
                        origin='lower',cmap=cmap)    

    

################################################################
####### Subtract the M-star models from the science data #######
################################################################
## Here we fit for a scaling factor because the new data appears to be brighter than the old data
## We could probably do this with a linear inversion in fewer lines...

def rms_residuals(x,data,mstars,mask): 
    resids = np.sum(jnp.sqrt((data-x[0]*mstars)**2)[mask])
    return resids


results_f300m_roll1 = minimize(rms_residuals,[1.],args=(sci1_data_f300m,m_star_roll1_data_f300m,mask),method='Nelder-Mead',tol=1e-6)
sci1_data_f300m_sub = sci1_data_f300m - results_f300m_roll1.x[0]*m_star_roll1_data_f300m

results_f300m_roll2 = minimize(rms_residuals,[1.],args=(sci2_data_f300m,m_star_roll2_data_f300m,mask),method='Nelder-Mead',tol=1e-6)
sci2_data_f300m_sub = sci2_data_f300m - results_f300m_roll2.x[0]*m_star_roll2_data_f300m

results_f360m_roll1 = minimize(rms_residuals,[1.],args=(sci1_data_f360m,m_star_roll1_data_f360m,mask),method='Nelder-Mead',tol=1e-6)
sci1_data_f360m_sub = sci1_data_f360m - results_f360m_roll1.x[0]*m_star_roll1_data_f360m

results_f360m_roll2 = minimize(rms_residuals,[1.],args=(sci2_data_f360m,m_star_roll2_data_f360m,mask),method='Nelder-Mead',tol=1e-6)
sci2_data_f360m_sub = sci2_data_f360m - results_f360m_roll2.x[0]*m_star_roll2_data_f360m


######################################
####### Plot the final results #######
######################################

plot_results = True
if plot_results: 

    #Coronagraphic center: 
    coron_center = [173.4,149.1]

    #F300M - pixel sizes and plotting range. 
    x_size_f300m = sci1_data_f300m.shape[1]
    y_size_f300m = sci1_data_f300m.shape[0]
    x_f300m = np.arange(x_size_f300m)
    y_f300m = np.arange(y_size_f300m)
    x_f300m = (x_f300m-coron_center[1])*pixel_scale_f300m
    y_f300m = (y_f300m-coron_center[0])*pixel_scale_f300m
    xx_f300m,yy_f300m = np.meshgrid(x_f300m,y_f300m)

    ## Setting up the minimum and maximum values for the color scale
    f300m_vmax = 100*conversion_factor
    f300m_vmin = -10*conversion_factor
    linthresh = 10

    f360m_vmax = 100*conversion_factor
    f360m_vmin = -10*conversion_factor
    
    ## Pick a colomap. Many have been tested. Freeze is the best. 
    cmap = plt.get_cmap('cmr.freeze') 
    

    #################################
    ######## 2-panel version ########
    #################################
    '''
    #We'll just show roll 1 from each image. 
    fig,ax = plt.subplots(2,1,figsize=(5.5,9))

    plt.subplots_adjust(hspace=0.1)

    ## The original Data
    im0 = ax[0].imshow(sci1_data_f300m,norm=SymLogNorm(linthresh,
                   vmax = f300m_vmax,vmin=f300m_vmin),
                   extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                   origin='lower',cmap=cmap)
    cbar = plt.colorbar(im0,ax=ax[0],label='uJy/arcsec$^2$',fraction=0.046, pad=0.04)
    #Adjust the ticks
    ax[0].tick_params(axis='both', which='both', labelsize=10, direction='in', pad=2)
    ax[0].set_title("Stage 2 data",fontsize=16)
    
    ## The data after the M-star model has been subtracted
    im2 = ax[1].imshow(sci1_data_f300m_sub,norm=SymLogNorm(linthresh,
                     vmax = f300m_vmax,vmin=f300m_vmin),
                     extent=[x_f300m[0],x_f300m[-1],y_f300m[0],y_f300m[-1]],
                     origin='lower',cmap=cmap)
    ax[1].tick_params(axis='both', which='both', labelsize=10, direction='in', pad=2)
    cbar = plt.colorbar(im2,ax=ax[1],label='uJy/arcsec$^2$',fraction=0.046, pad=0.04)
    ax[1].set_title("M-star Model Subtracted",fontsize=16)
    
    #Set up some axis labels
    ax[0].set_ylabel('y (mas)')
    ax[1].set_xlabel('x (mas)')
    ax[1].set_ylabel('y (mas)')

    #Save it and show it.     
    plt.tight_layout()
    
    fig.savefig("M_star_subtracted_F300M.png",dpi=300,bbox_inches='tight')
    '''
    
    #################################
    ######## 3-panel version ########
    #################################
    #Now we'll make a 3-panel plot where the middle panel is the m-star model 
    fig = plt.figure(figsize=(15, 5))

    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Add an extra column for the colorbar

    ax = [plt.subplot(gs[i]) for i in range(4)]

    plt.subplots_adjust(wspace=0.)

    # The original Data
    im0 = ax[0].imshow(sci1_data_f300m, norm=SymLogNorm(linthresh, vmax=f300m_vmax, vmin=f300m_vmin),
                    extent=[x_f300m[0], x_f300m[-1], y_f300m[0], y_f300m[-1]],
                    origin='lower', cmap=cmap)
    ax[0].tick_params(axis='both', which='both', labelsize=12, direction='in', pad=2, right=True, top=True, width=1.5, length=5, color='white')
    ax[0].set_title("Stage 2 data", fontsize=18)

    # The M-star model
    im1 = ax[1].imshow(m_star_roll1_data_f300m, norm=SymLogNorm(linthresh, vmax=f300m_vmax, vmin=f300m_vmin),
                    extent=[x_f300m[0], x_f300m[-1], y_f300m[0], y_f300m[-1]],
                    origin='lower', cmap=cmap)
    ax[1].tick_params(axis='both', which='both', labelsize=12, direction='in', pad=2, right=True, top=True, width=1.5, length=5, color='white')
    ax[1].set_title("M-star Model", fontsize=18)
    ax[1].set_yticklabels([]) 

    # The data after the M-star model has been subtracted
    im2 = ax[2].imshow(sci1_data_f300m_sub, norm=SymLogNorm(linthresh, vmax=f300m_vmax, vmin=f300m_vmin),
                    extent=[x_f300m[0], x_f300m[-1], y_f300m[0], y_f300m[-1]],
                    origin='lower', cmap=cmap)
    ax[2].tick_params(axis='both', which='both', labelsize=12, direction='in', pad=2, right=True, top=True, width=1.5, length=5, color='white')
    cbar = plt.colorbar(im2, cax=ax[3], label='uJy/arcsec$^2$', fraction=0.036, pad=0.04)
    ax[2].set_title("Stage 2 Model Subtracted", fontsize=18)
    ax[2].set_yticklabels([]) 

    # Set up some axis labels
    ax[0].set_ylabel('$\Delta$y ($\prime\prime$)', fontsize=14)
    ax[0].set_xlabel('$\Delta$x ($\prime\prime$)', fontsize=14)
    ax[1].set_xlabel('$\Delta$x ($\prime\prime$)', fontsize=14)
    ax[2].set_xlabel('$\Delta$x ($\prime\prime$)', fontsize=14)



    ## Make a Compass rose. 
    head_width = 0.5
    head_length = 0.5
    north_angle = roll_angle_f300m + 270
    east_angle = (roll_angle_f300m + 180) % 360  # East is counterclockwise of North

    # Assuming ax is your axis object
    arrow_length = 2.5
    compass_center = [8.5,-7]
    # Draw North arrow
    ax[0].arrow(compass_center[0], compass_center[1], arrow_length * np.cos(np.radians(north_angle)), -arrow_length * np.sin(np.radians(north_angle)), head_width=head_width, head_length=head_length, fc='w', ec='w')
    # Draw North label
    ax[0].text(arrow_length * np.cos(np.radians(north_angle))+compass_center[0], -arrow_length * np.sin(np.radians(north_angle))+compass_center[1]+1, 'N', color='w', ha='center', va='center')

    # Draw East arrow
    ax[0].arrow(compass_center[0], compass_center[1], arrow_length * np.cos(np.radians(east_angle)), -arrow_length * np.sin(np.radians(east_angle)), head_width=head_width, head_length=head_length, fc='w', ec='w')
    # Draw East label
    ax[0].text(arrow_length * np.cos(np.radians(east_angle))+compass_center[0]-1, -arrow_length * np.sin(np.radians(east_angle))+compass_center[1], 'E', color='w', ha='center', va='center')




    # Save it and show it.
    plt.tight_layout()
    fig.savefig("M_star_subtracted_F300M_3panel.png", dpi=300, bbox_inches='tight')
    plt.show()





#Save the subtracted data - adding a suffix to the original filename
# sci1_hdul_f300m[1].data = sci1_data_f300m_sub
# sci2_hdul_f300m[1].data = sci2_data_f300m_sub

# sci1_hdul_f300m.writeto(sci1_filename[:-5]+"_mstar_subtracted.fits",overwrite=True)
# sci2_hdul_f300m.writeto(sci2_filename[:-5]+"_mstar_subtracted.fits",overwrite=True)


# #Now the F360M data
# sci1_data_f360m_data = sci1_hdul_f360m[1].data
# sci2_data_f360m_data = sci2_hdul_f360m[1].data