from datetime import datetime
import glob 
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

from nircam_disk_utils import * 

import jax.numpy as jnp

################################################################
########## Read in the two rolls from F300M and F360M ##########
################################################################

f300m_roll1_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/jw01386117001_03106_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"
f300m_roll2_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/jw01386118001_03106_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"
f360m_roll1_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/jw01386117001_03107_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"
f360m_roll2_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/jw01386118001_03107_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"


f300m_roll1_hdul = fits.open(f300m_roll1_filename,overwrite=True)
f300m_roll2_hdul = fits.open(f300m_roll2_filename,overwrite=True)

f300m_roll1_data = f300m_roll1_hdul[0].data
f300m_roll2_data = f300m_roll2_hdul[0].data
f300m_roll1_err = f300m_roll1_hdul[1].data
f300m_roll2_err = f300m_roll2_hdul[1].data

f360m_roll1_hdul = fits.open(f360m_roll1_filename,overwrite=True)
f360m_roll2_hdul = fits.open(f360m_roll2_filename,overwrite=True)

f360m_roll1_data = f360m_roll1_hdul[0].data
f360m_roll2_data = f360m_roll2_hdul[0].data
f360m_roll1_err = f360m_roll1_hdul[1].data
f360m_roll2_err = f360m_roll2_hdul[1].data

##############################################
############## Crop the Data #################
##############################################
y1 = 100
y2 = 240
x1 = 80
x2 = 220
sci1_data_f300m = f300m_roll1_data[y1:y2,x1:x2].astype(jnp.float64)
sci2_data_f300m = f300m_roll2_data[y1:y2,x1:x2].astype(jnp.float64)
#Crop the errors too
sci1_err_f300m = f300m_roll1_err[y1:y2,x1:x2].astype(jnp.float64)
sci2_err_f300m = f300m_roll2_err[y1:y2,x1:x2].astype(jnp.float64)

sci1_data_f360m = f360m_roll1_data[y1:y2,x1:x2].astype(jnp.float64)
sci2_data_f360m = f360m_roll2_data[y1:y2,x1:x2].astype(jnp.float64)
#Crop the errors too
sci1_err_f360m = f360m_roll1_err[y1:y2,x1:x2].astype(jnp.float64)
sci2_err_f360m = f360m_roll2_err[y1:y2,x1:x2].astype(jnp.float64)

data_crop_shape = sci1_data_f300m.shape

######################################
######### Simulation Setup ###########
######################################
pixel_scale=0.063 # nircam pixel scale in arcsec/px
nx = x2-x1 # number of pixels of your image in X
ny = y2-y1 # number of pixels of your image in Y
cent = [173.4-y1,149.1-x1] #[y_cent,x_cent] #Cropped cent
n_nodes = 6 # A legacy term that defines the number of parameters used for 
osamp = 2

#############################################
######### Read in NIRCam model PSFs #########
#############################################
psf_dir = "/home/maxmb/Library/jwst_hd141569a_lib/PSFs/F300M/"
psf_suffixes = "_F300M.npy"

#Read in the pre-generated PSFs
im_mask_rolls = jnp.load(psf_dir+"im_mask_rolls"+psf_suffixes)
im_mask = im_mask_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
psf_inds_rolls = jnp.load(psf_dir+"psf_inds_rolls"+psf_suffixes)
psf_inds = psf_inds_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
psf_offsets = jnp.load(psf_dir+"psf_offsets"+psf_suffixes)
psfs = jnp.load(psf_dir+"psfs"+psf_suffixes)

unique_inds = jnp.unique(psf_inds_rolls)
n_unique_inds = len(unique_inds)

nircam_psf_list_f300m = [psfs,psf_inds,im_mask,unique_inds]

########################################################
######### Scaling values for some parameters ###########
########################################################

############# SMA ##############
#Parameters from June
# a0_1_f300m = 40. # semimajoraxis of the inner disk in au
# a0_2_f300m = 220. # semimajoraxis of the inner disk in au
# a0_3_f300m = 410. # semimajoraxis of the inner disk in au
# a0_4_f300m = 300.

#Parameters from October
a0_1_f300m = 40.0  # semimajoraxis of the inner disk in au
a0_2_f300m = 220.0  # semimajoraxis of the inner disk in au
a0_3_f300m = 510.0  # semimajoraxis of the inner disk in au
a0_4_f300m = 330.0

################# Relative flux #####################
#Parameters from June
# f0_1_f300m = 500*120./2
# f0_2_f300m = 50*250./2*1.2
# f0_3_f300m = 5000./20*0.3*2.2
# f0_4_f300m = 50*150./20*1.

#Parameters from October
f0_1_f300m = 3e4
f0_2_f300m = 7.5e3
f0_3_f300m = 140
f0_4_f300m = 410

# A scaling factor because the new data slightly brighter than the old data
new_flux_scaling = 1.23
f0_1_f300m *= new_flux_scaling
f0_2_f300m *= new_flux_scaling
f0_3_f300m *= new_flux_scaling
f0_4_f300m *= new_flux_scaling

############################################
########## Plotting extent details ################
############################################
data_shape = f300m_roll1_data.shape
pixel_scale_f300m = np.sqrt(f300m_roll1_hdul[1].header['PIXAR_A2'])

#Coronagraphic center: 
coron_center = [173.4,149.1]

#F300M - pixel sizes and plotting range. 
x_size_f300m = f300m_roll1_data.shape[1]
y_size_f300m = f300m_roll1_data.shape[0]
x_f300m = np.arange(x_size_f300m)
y_f300m = np.arange(y_size_f300m)
x_f300m = (x_f300m-coron_center[1])*pixel_scale_f300m
y_f300m = (y_f300m-coron_center[0])*pixel_scale_f300m
xx_f300m,yy_f300m = np.meshgrid(x_f300m,y_f300m)

#Now for the cropped data
x_size_crop = data_crop_shape[1]
y_size_crop = data_crop_shape[0]
x_crop = np.arange(x_size_crop)
y_crop = np.arange(y_size_crop)
x_crop_f300m = (x_crop-cent[1])*pixel_scale_f300m
y_crop_f300m = (y_crop-cent[0])*pixel_scale_f300m

new_extent = [x_crop_f300m[0],x_crop_f300m[-1],y_crop_f300m[0],y_crop_f300m[-1]]

###############################################
######## Generate some initial models #########
###############################################
print("Generating Base Model")
# x0_f300m = jnp.load("../230613/hg3fit_F300M_m_stars_bounded.npz.npy")
x0_f300m = jnp.load("../Disk_Modelling/hg3fit_F300M_m_stars_bounded_quad_new.npz.npy")

model2_f300m, model1_f300m = gen_roll_images_hg3(x0_f300m,cent,nircam_psf_list_f300m,
                                                 f0_1=f0_1_f300m,f0_2=f0_2_f300m,f0_3=f0_3_f300m,f0_4=f0_4_f300m,
                                                 a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

#Create four pairs of disk images, by zerouting out the other three disks
disk1_roll2_f300m, disk1_roll1_f300m = gen_roll_images_hg3(x0_f300m,cent,nircam_psf_list_f300m,
                                                              f0_1=f0_1_f300m,f0_2=0,f0_3=0,f0_4=0,
                                                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

disk2_roll2_f300m, disk2_roll1_f300m = gen_roll_images_hg3(x0_f300m,cent,nircam_psf_list_f300m,
                                                                f0_1=0,f0_2=f0_2_f300m,f0_3=0,f0_4=0,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)


disk3_roll2_f300m, disk3_roll1_f300m = gen_roll_images_hg3(x0_f300m,cent,nircam_psf_list_f300m,
                                                                f0_1=0,f0_2=0,f0_3=f0_3_f300m,f0_4=0,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

disk4_roll2_f300m, disk4_roll1_f300m = gen_roll_images_hg3(x0_f300m,cent,nircam_psf_list_f300m,
                                                                f0_1=0,f0_2=0,f0_3=0,f0_4=f0_4_f300m,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)    

###############################################
######## Define the fitting functions #########
###############################################

def model_images_f300m(params):
    f1 = params[0]
    f2 = params[1]
    f3 = params[2]
    f4 = params[3]

    model_image1 = f1*disk1_roll1_f300m + f2*disk2_roll1_f300m + f3*disk3_roll1_f300m + f4*disk4_roll1_f300m
    model_image2 = f1*disk1_roll2_f300m + f2*disk2_roll2_f300m + f3*disk3_roll2_f300m + f4*disk4_roll2_f300m

    return model_image1, model_image2

def log_likelihood_f300m(params, image1, image1_errors, image2, image2_errors):

    model_image1, model_image2 = model_images_f300m(params)

    #In this version the noise model is a fraction of 
    error_boost = params[4]
    new_errors1 = np.sqrt(image1_errors**2+(error_boost*model_image1)**2)
    new_errors2 = np.sqrt(image2_errors**2+(error_boost*model_image2)**2)

    logl_pt1 = np.sum((model_image1 - image1)**2/new_errors1**2+np.log(2*np.pi*new_errors1**2))
    logl_pt2 = np.sum((model_image2 - image2)**2/new_errors2**2+np.log(2*np.pi*new_errors2**2))

    return -0.5*(logl_pt1 + logl_pt2)

def log_prior(params):
    f1 = params[0]
    f2 = params[1]
    f3 = params[2]
    f4 = params[3]

    #Enforce positive fluxes
    if f1 < 0 or f2 < 0 or f3 < 0 or f4 < 0 or params[4] < 0:
        return -np.inf
    else: 
        return 0.0

def log_probability_f300m(params, image1, image1_errors, image2, image2_errors):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_f300m(params, image1, image1_errors, image2, image2_errors)


#######################################
######## Plot the First Guess #########
#######################################


import emcee
first_guess = np.array([0.995,1.01,1.,1.025,0.025])
ndim = len(first_guess)
nwalkers = 32

plot_first_guess = False
if plot_first_guess: 
    scaled_models = model_images_f300m(first_guess)
    error_boost = first_guess[4]
    new_errors2 = np.sqrt(sci2_err_f300m**2+error_boost**2)
    #With Colormars
    fig, axes = plt.subplots(1,4,figsize=(10,4))
    im0 = axes[0].imshow(sci2_data_f300m,vmin=-1,vmax=80)
    axes[0].set_title("Data")
    fig.colorbar(im0,ax=axes[0])
    im1 = axes[1].imshow(scaled_models[1],vmin=-1,vmax=80)
    axes[1].set_title("Model")
    fig.colorbar(im1,ax=axes[1])
    im2 = axes[2].imshow(sci2_data_f300m-scaled_models[1],vmin=-3,vmax=3,cmap='RdBu')
    axes[2].set_title("Data - Model")
    fig.colorbar(im2,ax=axes[2])
    im3 = axes[3].imshow((sci2_data_f300m-scaled_models[1])/new_errors2,vmin=-3,vmax=3,cmap='PuOr')
    axes[3].set_title("Data - Model / Error")
    fig.colorbar(im3,ax=axes[3])
    plt.show()

############################
######## Run Emcee #########
############################

#Positions of the first walkers
pos = first_guess + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_f300m, args=(sci1_data_f300m, sci1_err_f300m, sci2_data_f300m, sci2_err_f300m)
)

#For Testing
# burnin=10
# nsteps = 100

#For run-time
burnin=300
nsteps = 3000
# nsteps = 1000
print("Starting MCMC Burn-in with {} steps".format(burnin))
state = sampler.run_mcmc(pos, burnin,progress=True)
sampler.reset()
print("Starting main MCMC run with {} steps".format(nsteps))
sampler.run_mcmc(state, nsteps, progress=True)

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Ring 1 Scale", "Ring 2 Scale", "Ring 3 Scale", "Ring 4 Scale", "Fractional Variance\nIncrease"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
# plt.savefig("F300M_MCMC_chains.png",dpi=300)


tau = sampler.get_autocorr_time()
print("Autocorrelation times: {}".format(tau))
max_tau = np.max(tau).astype(int)

flat_samples_f300m = sampler.get_chain(discard=int(2.5*max_tau), thin=max_tau//2, flat=True)

#Convert to uJ/arcsecond^2
conversion_factor_f300m = f300m_roll1_hdul[1].header["PHOTUJA2"]/f300m_roll1_hdul[1].header["PHOTMJSR"]

plot_best_fit = False
if plot_best_fit: 
    median_params = np.median(flat_samples_f300m,axis=0)
    scaled_models = model_images_f300m(median_params)
    error_boost = median_params[4]
    new_errors2 = np.sqrt(sci2_err_f300m**2+(error_boost*scaled_models[1])**2)

    #Convert to uJ/arcsecond^2
    # conversion_factor_f300m = f300m_roll1_hdul[1].header["PHOTUJA2"]/f300m_roll1_hdul[1].header["PHOTMJSR"]

    #Plotting setup stuff
    f300m_vmax=1400
    f300m_vmin = 0
    linthresh = 0.1

    cmap = plt.get_cmap('cmr.freeze')

    gamma = 0.3
    #Make a Linear LogNorm, SymLogNorm and PowerNorm dictionary: 
    norms_f300m = {'linear': mpl.colors.Normalize(vmin=f300m_vmin,vmax=f300m_vmax),
            'log': mpl.colors.LogNorm(vmin=f300m_vmin,vmax=f300m_vmax),
            'symlog': mpl.colors.SymLogNorm(linthresh=linthresh,vmin=f300m_vmin,vmax=f300m_vmax),
            'power': mpl.colors.PowerNorm(gamma=gamma,vmin=f300m_vmin,vmax=f300m_vmax)}

    norm_key = "power"

    #With Colormars
    fig, axes = plt.subplots(1,4,figsize=(12,3))
    im0 = axes[0].imshow(sci2_data_f300m*conversion_factor_f300m,origin='lower',
                cmap=cmap,norm = norms_f300m[norm_key],
                extent=new_extent)
    axes[0].set_title("Data")
    cbar0 = fig.colorbar(im0,ax=axes[0])
    cbar0.set_label(label=r'$\mu $Jy arcsec$^{-2}$',size=12)
    
    im1 = axes[1].imshow(scaled_models[1]*conversion_factor_f300m,
                cmap=cmap,norm = norms_f300m[norm_key],
                extent=new_extent)
    axes[1].set_title("Model")
    cbar1 = fig.colorbar(im1,ax=axes[1])
    cbar1.set_label(label=r'$\mu $Jy arcsec$^{-2}$',size=12)

    im2 = axes[2].imshow((sci2_data_f300m-scaled_models[1])**conversion_factor_f300m,vmin=-3,vmax=3,cmap='RdBu',extent=new_extent)
    axes[2].set_title("Data - Model")
    fig.colorbar(im2,ax=axes[2],)
    
    im3 = axes[3].imshow((sci2_data_f300m-scaled_models[1])/new_errors2,vmin=-3,vmax=3,cmap='Spectral',extent=new_extent)
    axes[3].set_title("Data - Model / Error")
    fig.colorbar(im3,ax=axes[3])
    fig.suptitle("F300M")
    # fig.savefig("F300M_best_fit_flux_model.png")

    axes[0].set_ylabel(r"$\Delta y ('')$",fontsize=12)
    
    axes[0].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[1].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[2].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[3].set_xlabel(r"$\Delta x ('')$",fontsize=12)

    #Just the normalized residuals
    fig, axes = plt.subplots(1,1,figsize=(5,5))
    axes.imshow((sci2_data_f300m-scaled_models[1])/new_errors2,vmin=-3,vmax=3,cmap='Spectral',extent=new_extent)
    axes.set_title("F300M Roll 1\n(Data - Model) / Error")
    cbar = fig.colorbar(im3,ax=axes)
    cbar.set_label(label=r'$\sigma$',size=12)
    # fig.suptitle("F300M")
    axes.set_ylabel(r"$\Delta y ('')$",fontsize=12)
    axes.set_xlabel(r"$\Delta x ('')$",fontsize=12)
    # plt.savefig("F300M_residuals_sigma.png",dpi=300)

plot_chains_corner = False
if plot_chains_corner: 
    import corner
    figure = corner.corner(flat_samples_f300m,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt='.3f')
    # plt.savefig("F300M_corner.png",dpi=300)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")


plt.show()

np.save("F300M_emcee_samples.npy",flat_samples_f300m)

##############################################################
##############################################################
######### Now let's do all the same things for F360M #########
##############################################################
##############################################################
n_refs = 9
x0_f360m = jnp.load("../Disk_Modelling/hg3fit_F360M_full_params.npy")

f0_1_f360m = 1.5e5
f0_2_f360m = 2.5e3
f0_3_f360m = 140
f0_4_f360m = 410

############################################
########## Plotting details ################
############################################
data_shape = f360m_roll1_data.shape
pixel_scale_f360m = np.sqrt(f360m_roll1_hdul[1].header['PIXAR_A2'])

#Coronagraphic center: 
coron_center = [173.4,149.1]

#f360m - pixel sizes and plotting range. 
x_size_f360m = f360m_roll1_data.shape[1]
y_size_f360m = f360m_roll1_data.shape[0]
x_f360m = np.arange(x_size_f360m)
y_f360m = np.arange(y_size_f360m)
x_f360m = (x_f360m-coron_center[1])*pixel_scale_f360m
y_f360m = (y_f360m-coron_center[0])*pixel_scale_f360m
xx_f360m,yy_f360m = np.meshgrid(x_f360m,y_f360m)

#Now for the cropped data
x_size_crop = data_crop_shape[1]
y_size_crop = data_crop_shape[0]
x_crop = np.arange(x_size_crop)
y_crop = np.arange(y_size_crop)
x_crop_f360m = (x_crop-cent[1])*pixel_scale_f360m
y_crop_f360m = (y_crop-cent[0])*pixel_scale_f360m

new_extent = [x_crop_f360m[0],x_crop_f360m[-1],y_crop_f360m[0],y_crop_f360m[-1]]

############################################
######### Read in NIRCam model PSFs #########
############################################
psf_dir = "/home/maxmb/Library/jwst_hd141569a_lib/PSFs/F360M/"
psf_suffixes = "_F360M.npy"

#Read in the pre-generated PSFs
im_mask_rolls = jnp.load(psf_dir+"im_mask_rolls"+psf_suffixes)
im_mask = im_mask_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
psf_inds_rolls = jnp.load(psf_dir+"psf_inds_rolls"+psf_suffixes)
psf_inds = psf_inds_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
psf_offsets = jnp.load(psf_dir+"psf_offsets"+psf_suffixes)
psfs = jnp.load(psf_dir+"psfs"+psf_suffixes)

unique_inds = jnp.unique(psf_inds_rolls)
n_unique_inds = len(unique_inds)

nircam_psf_list_f360m = [psfs,psf_inds,im_mask,unique_inds]

###############################################
######## Generate some initial models #########
###############################################

model2_f360m, model1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                    f0_1=f0_1_f360m,f0_2=f0_2_f360m,f0_3=f0_3_f360m,f0_4=f0_4_f360m,
                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

#Create four pairs of disk images, by zerouting out the other three disks
disk1_roll2_f360m, disk1_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                              f0_1=f0_1_f360m,f0_2=0,f0_3=0,f0_4=0,
                                                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

disk2_roll2_f360m, disk2_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=f0_2_f360m,f0_3=0,f0_4=0,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

disk3_roll2_f360m, disk3_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=0,f0_3=f0_3_f360m,f0_4=0,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

disk4_roll2_f360m, disk4_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=0,f0_3=0,f0_4=f0_4_f360m,
                                                                    a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

###############################################
######## Define the fitting functions #########
###############################################

def model_images_f360m(params):
    f1 = params[0]
    f2 = params[1]
    f3 = params[2]
    f4 = params[3]

    model_image1 = f1*disk1_roll1_f360m + f2*disk2_roll1_f360m + f3*disk3_roll1_f360m + f4*disk4_roll1_f360m
    model_image2 = f1*disk1_roll2_f360m + f2*disk2_roll2_f360m + f3*disk3_roll2_f360m + f4*disk4_roll2_f360m

    return model_image1, model_image2

def log_likelihood_f360m(params, image1, image1_errors, image2, image2_errors):
    
        model_image1, model_image2 = model_images_f360m(params)
    
        error_boost = params[4]
        new_errors1 = np.sqrt(image1_errors**2+(error_boost*model_image1)**2)
        new_errors2 = np.sqrt(image2_errors**2+(error_boost*model_image2)**2)
    
        logl_pt1 = np.sum((model_image1 - image1)**2/(new_errors1**2)+np.log(2*np.pi*(new_errors1**2)))
        logl_pt2 = np.sum((model_image2 - image2)**2/(new_errors2**2)+np.log(2*np.pi*(new_errors2**2)))
    
        return -0.5*(logl_pt1 + logl_pt2)

def log_probability_f360m(params, image1, image1_errors, image2, image2_errors):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_f360m(params, image1, image1_errors, image2, image2_errors)

#######################################
######## Plot the First Guess #########
#######################################

import emcee
first_guess = np.array([1.02,0.98,1.,1.2,0.10]) #Based on some initial trials
ndim = len(first_guess)
nwalkers = 32

plot_first_guess = False
if plot_first_guess: 
    scaled_models = model_images_f360m(first_guess)
    error_boost = first_guess[4]
    new_errors2 = np.sqrt(sci2_err_f360m**2+(error_boost*scaled_models[1])**2)
    #With Colormars
    fig, axes = plt.subplots(1,4,figsize=(10,4))
    im0 = axes[0].imshow(sci2_data_f360m,vmin=-1,vmax=80)
    axes[0].set_title("Data")
    fig.colorbar(im0,ax=axes[0])
    im1 = axes[1].imshow(scaled_models[1],vmin=-1,vmax=80)
    axes[1].set_title("Model")
    fig.colorbar(im1,ax=axes[1])
    im2 = axes[2].imshow(sci2_data_f360m-scaled_models[1],vmin=-3,vmax=3,cmap='RdBu')
    axes[2].set_title("Data - Model")
    fig.colorbar(im2,ax=axes[2])
    im3 = axes[3].imshow((sci2_data_f360m-scaled_models[1])/new_errors2,vmin=-1,vmax=1,cmap='PuOr')
    axes[3].set_title("Data - Model / Error")
    fig.colorbar(im3,ax=axes[3])
    plt.show()

############################
######## Run Emcee #########
############################

pos = first_guess + 1e-4 * np.random.randn(nwalkers, ndim)
sampler_f360m = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_f360m, args=(sci1_data_f360m, sci1_err_f360m, sci2_data_f360m, sci2_err_f360m)
)

#For testing
# burnin=10
# nsteps = 100

burnin=400
nsteps = 3000

print("Starting MCMC Burn-in with {} steps".format(burnin))
state = sampler_f360m.run_mcmc(pos, burnin,progress=True)
sampler_f360m.reset()
print("Starting main MCMC run with {} steps".format(nsteps))
sampler_f360m.run_mcmc(state, nsteps, progress=True)

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler_f360m.get_chain()
# labels = ["disk1_scale", "disk2_scale", "disk3_scale","disk4_scale", "error_boost"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
# plt.savefig("F360M_MCMC_chains.png",dpi=300)
# plt.show()

tau = sampler_f360m.get_autocorr_time()
print("Autocorrelation times: {}".format(tau))
# max_tau = np.max(tau).astype(int)
max_tau = np.max([sampler_f360m.get_autocorr_time(),sampler.get_autocorr_time()]).astype(int)

flat_samples_f360m = sampler_f360m.get_chain(discard=int(2.5*max_tau), thin=max_tau//2, flat=True)

plot_best_fit = True
if plot_best_fit: 
    median_params = np.median(flat_samples_f360m,axis=0)
    scaled_models = model_images_f360m(median_params)
    error_boost = median_params[4]
    new_errors2 = np.sqrt(sci2_err_f360m**2+(error_boost*scaled_models[1])**2)

    #Convert to uJ/arcsecond^2
    conversion_factor_f360m = f360m_roll1_hdul[1].header["PHOTUJA2"]/f360m_roll1_hdul[1].header["PHOTMJSR"]

    #Plotting setup stuff
    f360m_vmax=2200
    f360m_vmin = 0
    linthresh = 0.1

    import cmasher as cmr
    cmap = plt.get_cmap('cmr.freeze')

    gamma = 0.3
    #Make a Linear LogNorm, SymLogNorm and PowerNorm dictionary: 
    norms_f360m = {'linear': mpl.colors.Normalize(vmin=f360m_vmin,vmax=f360m_vmax),
         'log': mpl.colors.LogNorm(vmin=f360m_vmin,vmax=f360m_vmax),
         'symlog': mpl.colors.SymLogNorm(linthresh=linthresh,vmin=f360m_vmin,vmax=f360m_vmax),
         'power': mpl.colors.PowerNorm(gamma=gamma,vmin=f360m_vmin,vmax=f360m_vmax)}

    norm_key = "power"

    #With Colormars
    fig, axes = plt.subplots(1,4,figsize=(12,3))
    im0 = axes[0].imshow(sci2_data_f360m*conversion_factor_f360m,origin='lower',
                cmap=cmap,norm = norms_f360m[norm_key],
                extent=new_extent)
    axes[0].set_title("Data")
    cbar0 = fig.colorbar(im0,ax=axes[0])
    cbar0.set_label(label=r'$\mu $Jy arcsec$^{-2}$',size=12)
    
    im1 = axes[1].imshow(scaled_models[1]*conversion_factor_f360m,
                cmap=cmap,norm = norms_f360m[norm_key],
                extent=new_extent)
    axes[1].set_title("Model")
    cbar1 = fig.colorbar(im1,ax=axes[1])
    cbar1.set_label(label=r'$\mu $Jy arcsec$^{-2}$',size=12)

    im2 = axes[2].imshow((sci2_data_f360m-scaled_models[1])**conversion_factor_f360m,vmin=-3,vmax=3,cmap='RdBu',extent=new_extent)
    axes[2].set_title("Data - Model")
    fig.colorbar(im2,ax=axes[2],)
    
    im3 = axes[3].imshow((sci2_data_f360m-scaled_models[1])/new_errors2,vmin=-3,vmax=3,cmap='Spectral',extent=new_extent)
    axes[3].set_title("Data - Model / Error")
    fig.colorbar(im3,ax=axes[3])
    fig.suptitle("F360M")
    # fig.savefig("f360m_best_fit_flux_model.png")

    axes[0].set_ylabel(r"$\Delta y ('')$",fontsize=12)
    
    axes[0].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[1].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[2].set_xlabel(r"$\Delta x ('')$",fontsize=12)
    axes[3].set_xlabel(r"$\Delta x ('')$",fontsize=12)

    #Just the normalized residuals
    fig, axes = plt.subplots(1,1,figsize=(5,5))
    axes.imshow((sci2_data_f360m-scaled_models[1])/new_errors2,vmin=-3,vmax=3,cmap='Spectral',extent=new_extent)
    axes.set_title("F360M Roll 1\n(Data - Model) / Error")
    cbar = fig.colorbar(im3,ax=axes)
    cbar.set_label(label=r'$\sigma$',size=12)
    # fig.suptitle("f360m")
    axes.set_ylabel(r"$\Delta y ('')$",fontsize=12)
    axes.set_xlabel(r"$\Delta x ('')$",fontsize=12)
    # plt.savefig("F360M_residuals_sigma.png",dpi=300)
    
plot_chains_corner = True
if plot_chains_corner: 
    import corner
    figure = corner.corner(flat_samples_f360m,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt='.3f')
    # plt.savefig("f360m_corner.png",dpi=300)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

plt.show()

np.save("F360M_emcee_samples.npy",flat_samples_f360m)

#############################
#############################
### Make Flux Ratio Plots ###
#############################
#############################

n_samples = 100
#pick n_samples random samples from the chain
rand_inds = np.random.randint(0,np.min([len(flat_samples_f300m),len(flat_samples_f360m)]),n_samples)

def elliptical_distance_fn(center, shape, inclination, rotation):
    """
    Calculate the elliptical distance from a center point in a 2D space.

    Parameters:
    center (tuple): The (x, y) coordinates of the center point.
    shape (tuple): The shape of the 2D space as (height, width).
    inclination (float): The inclination of the ellipse.
    rotation (float): The rotation of the ellipse in radians.

    Returns:
    np.array: A 2D array representing the elliptical distances.
    """
    # Create a 2D grid of (x, y) coordinates
    y, x = np.ogrid[:shape[0], :shape[1]]

    # Shift the origin to the center
    x = x - center[0]
    y = y - center[1]

    # Apply the rotation
    x_new = x * np.cos(rotation) - y * np.sin(rotation)
    y_new = x * np.sin(rotation) + y * np.cos(rotation)

    # Calculate the elliptical distance
    distance = np.sqrt((x_new)**2 + (y_new / np.sin(inclination))**2)

    return distance

#Inclination 0 is face on, 90 is edge on

###########################################
### Measure the profile from the models ###
###########################################

#Adjust to the roll 2 center
elliptical_distance = elliptical_distance_fn(cent[::-1]+np.array([x0_f300m[20+4*n_nodes],x0_f300m[21+4*n_nodes]]), data_crop_shape, np.pi/2-x0_f300m[0], np.pi/2-x0_f300m[1])

# star_flux_f300m = 684 #mJy - From here: https://docs.google.com/document/d/11nynSfSsUnl0N80X9CAwg-nHhaa-dC1l/edit?usp=sharing&ouid=106103012443201268330&rtpof=true&sd=true
# star_flux_f360m = 491 #mJy

star_flux_f300m = 712 #mJy - Updated by Elodie 07/2024 to include full bandpass
star_flux_f360m = 533 #mJy
 
ratio_profiles = []
normalized_surface_brightness_f300m_model = []
normalized_surface_brightness_f360m_model = []
for i in range(n_samples):
    #Generate a pair of model images for both filters
    model1_f300m, model2_f300m = model_images_f300m(flat_samples_f300m[rand_inds[i]])
    model1_f360m, model2_f360m = model_images_f360m(flat_samples_f360m[rand_inds[i]])

    width = 2
    n_bins = 35
    this_ratio_profile = np.zeros(n_bins)
    this_normalized_surface_brightness_f300m = np.zeros(n_bins)
    this_normalized_surface_brightness_f360m = np.zeros(n_bins)
    for j in range(n_bins):
        #Create an annulus
        annulus = np.where((elliptical_distance > j*width) & (elliptical_distance < (j+1)*width))
        #Calculate the flux ratio
        # ratio = np.median(model2_f300m[annulus])/np.median(model2_f360m[annulus])*star_flux_f360m/star_flux_f300m
        ratio = np.median((model2_f300m/model2_f360m)[annulus])*star_flux_f360m/star_flux_f300m
        this_ratio_profile[j] = ratio
        this_normalized_surface_brightness_f300m[j] = np.median(model2_f300m[annulus])/star_flux_f300m
        this_normalized_surface_brightness_f360m[j] = np.median(model2_f360m[annulus])/star_flux_f360m
    ratio_profiles.append(this_ratio_profile)
    normalized_surface_brightness_f300m_model.append(this_normalized_surface_brightness_f300m)
    normalized_surface_brightness_f360m_model.append(this_normalized_surface_brightness_f360m)

#########################################
### Measure the profile from the data ###
#########################################

#Measure this ratio in the data too for one of the rolls
data_ratio_profile = np.zeros(n_bins)
normalized_surface_brightness_f300m = np.zeros(n_bins)
normalized_surface_brightness_f360m = np.zeros(n_bins)
for j in range(n_bins):
    #Create an annulus
    annulus = np.where((elliptical_distance > j*width) & (elliptical_distance < (j+1)*width))
    #Calculate the flux ratio
    # ratio = np.median(sci2_data_f300m[annulus])/np.median(sci2_data_f360m[annulus])*star_flux_f360m/star_flux_f300m
    ratio = np.median((sci2_data_f300m/sci2_data_f360m)[annulus])*star_flux_f360m/star_flux_f300m
    data_ratio_profile[j] = ratio
    normalized_surface_brightness_f300m[j] = np.median(sci2_data_f300m[annulus])/star_flux_f300m
    normalized_surface_brightness_f360m[j] = np.median(sci2_data_f360m[annulus])/star_flux_f360m


#####################################################
### Measure the profile from the deconvolved data ###
#####################################################

deoconvolved_data_Dir = "/home/maxmb/Library/jwst_ers_NIRCAM_HD141569A/Deconvolution/decon_data/"

f300m_filename = "JWST_NIRCAM_NRCALONG_F300M_MASKRND_MASKA335R_SUB320A335R_mcrdi_psfsub_deconv.fits"
f360m_filename = "JWST_NIRCAM_NRCALONG_F360M_MASKRND_MASKA335R_SUB320A335R_mcrdi_psfsub_deconv.fits"

deconvolved_f300m_data = fits.open(deoconvolved_data_Dir+f300m_filename)[1].data.astype(jnp.float64)
deconvolved_f360m_data = fits.open(deoconvolved_data_Dir+f360m_filename)[1].data.astype(jnp.float64)

#Subtract a background median
deconvolved_f300m_data = deconvolved_f300m_data - np.nanmedian(deconvolved_f300m_data[:100,:100])
deconvolved_f360m_data = deconvolved_f360m_data - np.nanmedian(deconvolved_f360m_data[:100,:100])

roll1_f300m = f300m_roll1_hdul[0].header['GS_V3_PA']
deconvolved_cent = jnp.array([173.4,149.1])
deconvolved_f300m_data = pyklip_rotate(deconvolved_f300m_data, -roll1_f300m, deconvolved_cent[::-1])
deconvolved_f360m_data = pyklip_rotate(deconvolved_f360m_data, -roll1_f300m, deconvolved_cent[::-1])

#Measure this ratio in the data too for one of the rolls
deconvolved_ratio_profile = np.zeros(n_bins)
deconvolved_ratio_profile_test = np.zeros(n_bins)
deconvolved_surface_brightness_f300m = np.zeros(n_bins)
deconvolved_surface_brightness_f360m = np.zeros(n_bins)
for j in range(n_bins):

    deconvolved_elliptical_distance = elliptical_distance_fn(deconvolved_cent[::-1], deconvolved_f300m_data.shape, np.pi/2-x0_f300m[0], np.pi/2-x0_f300m[1])
    #Create an annulus
    annulus = np.where((deconvolved_elliptical_distance > j*width) & (deconvolved_elliptical_distance < (j+1)*width))
    #Calculate the flux ratio
    ratio = np.nanmedian(deconvolved_f300m_data[annulus])/np.nanmean(deconvolved_f360m_data[annulus])*star_flux_f360m/star_flux_f300m
    deconvolved_ratio_profile[j] = ratio
    deconvolved_ratio_profile_test[j] = np.nanmedian((deconvolved_f300m_data/deconvolved_f360m_data)[annulus])*star_flux_f360m/star_flux_f300m
    deconvolved_surface_brightness_f300m[j] = np.nanmedian(deconvolved_f300m_data[annulus])/star_flux_f300m
    deconvolved_surface_brightness_f360m[j] = np.nanmedian(deconvolved_f360m_data[annulus])/star_flux_f360m


dstar= 111.61 # distance to the star in pc
distances = np.arange(n_bins)*width+width/2
distances_au = distances*pixel_scale*dstar


fig, axes = plt.subplots(1,3,figsize=(16,4))
im = axes[0].imshow(deconvolved_f300m_data/deconvolved_f360m_data*star_flux_f360m/star_flux_f300m,vmax=1.)
plt.colorbar(im, ax = axes[0])

im2 = axes[1].imshow((deconvolved_elliptical_distance > j*width) & (deconvolved_elliptical_distance < (j+1)*width))

axes[2].plot(distances_au,deconvolved_ratio_profile,label="Deconvolved Data")
axes[2].plot(distances_au,deconvolved_ratio_profile_test)
plt.show()
#########################################################################
######## Plot the flux ratio profiles for each sample (OLD PLOT) ########
#########################################################################

# fig,axes = plt.subplots(1,2,figsize=(12,4))
# ax=axes[0]
# ax.set_xlabel("Elliptical Distance (px)")
# ax.set_ylabel("Flux Ratio")
# ax.set_title("Flux Ratio Profiles")
# ax.plot(distances_au,ratio_profiles[i],alpha=0.05,color='k', label="MCMC Samples")
# for i in range(1,n_samples):
#     ax.plot(distances_au,ratio_profiles[i],alpha=0.05,color='k')
# ax.plot(distances_au, data_ratio_profile, color='r', label="Data")
# ax.set_ylim(0,1)
# ax.legend()
# # fig.savefig("flux_ratio_profiles.png")
# #Convert the ratios to astronomical delta magnitude, and convert the x-axis to astronomical units using the pixel scale and distance

# ######## Plot the delta magnitude profiles for each sample ###

# delta_mags = -2.5*np.log10(np.array(ratio_profiles))
# delta_mag_data = -2.5*np.log10(data_ratio_profile)
# # fig = plt.figure(figsize=(6,4))
# ax2 = axes[1]
# ax2.set_xlabel("Distance (au)")
# ax2.set_ylabel("Delta Magnitude")
# ax2.set_title("Delta Magnitude Profiles")
# for i in range(n_samples):
#     ax2.plot(distances_au,delta_mags[i],alpha=0.1,color='k')
# ax2.plot(distances_au,delta_mag_data,color='r')
# fig.savefig("flux_ratio_delta_mag_profiles.png")


################################################
######## Display the flux ratio images  ########
################################################

#Use the elliptical distance to create contour lines at intervals of 100au from the center and overplot it on a residuals image
fig, axes = plt.subplots(1,2,figsize=(8,3.7))
linestyle = "--"
alpha = 0.7
vmin=0
vmax=1.0

#Now the contour lines
# levels = np.arange(0,1000,200)
levels = np.array([45,245,406]) #Rings from Singh et al. 2021, Mazoyer et al. 2016 and Biller 2015

im2 = axes[0].imshow(sci2_data_f300m/sci2_data_f360m*star_flux_f360m/star_flux_f300m,vmin=vmin,vmax=vmax,extent=new_extent)
axes[0].set_title("Data")
axes[0].contour(elliptical_distance*pixel_scale,levels/dstar,colors='w',linestyle=linestyle,alpha=alpha,extent=new_extent)

im1 = axes[1].imshow(model1_f300m/model1_f360m*star_flux_f360m/star_flux_f300m,vmin=vmin,vmax=vmax,extent=new_extent)
axes[1].set_title("Model")
cbar = fig.colorbar(im1,ax=axes[1], fraction=0.096, pad=0.04,label="F300M/F360M Scattering Efficiency Ratio")
#Now the contour lines
axes[1].contour(elliptical_distance*pixel_scale,levels/dstar,colors='w',linestyle=linestyle,alpha=alpha,extent=new_extent)

axes[1].set_yticks([])
axes[1].set_yticklabels([])

axes[0].set_xlabel("$\Delta$ x ('')")
axes[1].set_xlabel("$\Delta$ x ('')")
axes[0].set_ylabel("$\Delta$ y ('')")


plt.tight_layout()

# plt.savefig("F300M_F360M_flux_ratio_images.png",dpi=300)
# plt.show()

#########################################################################
######## Display the flux ratio images with the deconvolved data ########
#########################################################################

#Use the elliptical distance to create contour lines at intervals of 100au from the center and overplot it on a residuals image
fig, axes = plt.subplots(1,3,figsize=(12,3.7))
linestyle = "--"
alpha = 0.7
vmin=0
vmax=1.0

#Now the contour lines
# levels = np.arange(0,1000,200)
levels = np.array([45,245,406]) #Rings from Singh et al. 2021, Mazoyer et al. 2016 and Biller 2015

im2 = axes[0].imshow(sci2_data_f300m/sci2_data_f360m*star_flux_f360m/star_flux_f300m,vmin=vmin,vmax=vmax,extent=new_extent)
axes[0].set_title("Data")
axes[0].contour(elliptical_distance*pixel_scale,levels/dstar,colors='w',linestyle=linestyle,alpha=alpha,extent=new_extent)

im1 = axes[1].imshow(model1_f300m/model1_f360m*star_flux_f360m/star_flux_f300m,vmin=vmin,vmax=vmax,extent=new_extent)
axes[1].set_title("Model")
cbar = fig.colorbar(im1,ax=axes[1], fraction=0.096, pad=0.04,label="F300M/F360M Scattering Efficiency Ratio")
#Now the contour lines
axes[1].contour(elliptical_distance*pixel_scale,levels/dstar,colors='w',linestyle=linestyle,alpha=alpha,extent=new_extent)

axes[1].set_yticks([])
axes[1].set_yticklabels([])

axes[0].set_xlabel("$\Delta$ x ('')")
axes[1].set_xlabel("$\Delta$ x ('')")
axes[0].set_ylabel("$\Delta$ y ('')")

im3 = axes[2].imshow(deconvolved_f300m_data/deconvolved_f360m_data*star_flux_f360m/star_flux_f300m,vmin=vmin,vmax=vmax,extent=new_extent)
axes[2].set_title("Deconvolved Data")
cbar = fig.colorbar(im3,ax=axes[2], fraction=0.096, pad=0.04,label="F300M/F360M Scattering Efficiency Ratio")
#Now the contour lines
axes[2].contour(deconvolved_elliptical_distance*pixel_scale,levels/dstar,colors='r',linestyle=linestyle,alpha=alpha,extent=new_extent)

plt.tight_layout()

# plt.savefig("F300M_F360M_flux_ratio_images_alt.png",dpi=300)
# plt.show()

################################################
######## Display the delta mag images  #########
################################################
# fig, axes = plt.subplots(1,3,figsize=(12,4))
# linestyle = "--"
# alpha = 0.7
# vmin=-1
# vmax=50
# im0 = axes[0].imshow(sci2_data_f300m,vmin=vmin,vmax=vmax)
# axes[0].set_title("F300M Science; Roll 2")
# fig.colorbar(im0,ax=axes[0])
# #Now the contour lines
# levels = np.arange(0,1000,200)
# axes[0].contour(elliptical_distance*pixel_scale*dstar,levels,colors='w',linestyle=linestyle,alpha=alpha)

# im2 = axes[1].imshow(-2.5*np.log10(sci2_data_f300m/sci2_data_f360m*star_flux_f360m/star_flux_f300m),vmin=-2,vmax=2,cmap='RdBu_r')
# axes[1].set_title("Data Delta Mag")
# fig.colorbar(im2,ax=axes[1])
# #Now the contour lines

# im1 = axes[2].imshow(-2.5*np.log10(model1_f300m/model1_f360m*star_flux_f360m/star_flux_f300m),vmin=-2,vmax=2,cmap='RdBu_r')
# axes[2].set_title("Model Delta Mag")
# fig.colorbar(im1,ax=axes[2])
# #Now the contour lines
# axes[1].contour(elliptical_distance*pixel_scale*dstar,levels,colors='w',linestyle=linestyle,alpha=alpha)
# axes[2].contour(elliptical_distance*pixel_scale*dstar,levels,colors='w',linestyle=linestyle,alpha=alpha)
# plt.savefig("F300M_F360M_delta_mag_images.png")
# plt.show()

###################################################
######## Make unconvolved model profiles  #########
###################################################

raw1_f300m = quad_disk_model_hg3(x0_f300m,cent,
                                f0_1=f0_1_f300m,f0_2=0,f0_3=0,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw2_f300m = quad_disk_model_hg3(x0_f300m,cent,
                                f0_1=0,f0_2=f0_2_f300m,f0_3=0,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw3_f300m = quad_disk_model_hg3(x0_f300m,cent,
                                f0_1=0,f0_2=0,f0_3=f0_3_f300m,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw4_f300m = quad_disk_model_hg3(x0_f300m,cent,
                                f0_1=0,f0_2=0,f0_3=0,f0_4=f0_4_f300m,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

raw1_f360m = quad_disk_model_hg3(x0_f360m,cent,
                                f0_1=f0_1_f360m,f0_2=0,f0_3=0,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw2_f360m = quad_disk_model_hg3(x0_f360m,cent,
                                f0_1=0,f0_2=f0_2_f360m,f0_3=0,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw3_f360m = quad_disk_model_hg3(x0_f360m,cent,
                                f0_1=0,f0_2=0,f0_3=f0_3_f360m,f0_4=0,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)
raw4_f360m = quad_disk_model_hg3(x0_f360m,cent,
                                f0_1=0,f0_2=0,f0_3=0,f0_4=f0_4_f360m,
                                a0_1=a0_1_f300m,a0_2=a0_2_f300m,a0_3=a0_3_f300m,a0_4=a0_4_f300m)

median_samples_f300m = np.median(flat_samples_f300m,axis=0)
median_samples_f360m = np.median(flat_samples_f360m,axis=0)

raw_quad_model_f300m = median_samples_f300m[0]*raw1_f300m+\
                          median_samples_f300m[1]*raw2_f300m+\
                             median_samples_f300m[2]*raw3_f300m+\
                                median_samples_f300m[3]*raw4_f300m

raw_quad_model_f300m /=4 ##Account for oversampling sum when forward modeling

raw_quad_model_f360m = median_samples_f360m[0]*raw1_f360m+\
                          median_samples_f360m[1]*raw2_f360m+\
                            median_samples_f360m[2]*raw3_f360m+\
                                median_samples_f360m[3]*raw4_f360m

raw_quad_model_f360m /=4 ##Account for oversampling sum when forward modeling

#Measure the flux ratio profile of the unconvolved models
raw_profile_f300m = np.zeros(n_bins)
raw_profile_f360m = np.zeros(n_bins)

for j in range(n_bins):
    #Create an annulus
    annulus = np.where((elliptical_distance > j*width) & (elliptical_distance < (j+1)*width))
    #Calculate the flux ratio
    ratio_f300m = np.median(raw_quad_model_f300m[annulus])
    ratio_f360m = np.median(raw_quad_model_f360m[annulus])
    raw_profile_f300m[j] = ratio_f300m
    raw_profile_f360m[j] = ratio_f360m

raw_ratio_profile = raw_profile_f300m/raw_profile_f360m*star_flux_f360m/star_flux_f300m

#Read in Kellen's delta mag_profile
kellen_delta_mag = np.genfromtxt("kellen_delta_mags.csv",delimiter=',')
#convert it to a flux ratio profile
kellen_flux_ratio = 10**(-kellen_delta_mag[:,1]/2.5)

fig,axis = plt.subplots(1,1,figsize=(6,4))
# ax=axes[0]
# ax.set_xlabel("Deprojected Radial Distance (au)")
# ax.set_ylabel("Normalized Surface Brightness")

# #From the best fit unconvolved model
# ax.semilogy(distances_au,raw_profile_f300m/star_flux_f300m/1e3*conversion_factor_f300m,color='C0',label="F300M - Unconvolved Model") #1e-3 to convert from mjy (star_flux) to uJy (image units from conversion_factor)
# ax.plot(distances_au,raw_profile_f360m/star_flux_f360m/1e3*conversion_factor_f300m,color='C0',label="F360M - Unvonvolved Model",linestyle='--')
# # The convolved MCMC Models
import matplotlib.lines as mlines
legend_line = mlines.Line2D([], [], color='k', label='F300M MCMC Samples')
# for i in range(0,n_samples):
#     ax.plot(distances_au,normalized_surface_brightness_f300m_model[i]/1e3*conversion_factor_f300m,alpha=0.1,color='k')
#     ax.plot(distances_au,normalized_surface_brightness_f360m_model[i]/1e3*conversion_factor_f300m,alpha=0.1,color='k',linestyle='--')

# # The Deconvolved Data
# ax.plot(distances_au,deconvolved_surface_brightness_f300m/1e3,color='C3',label="F300M - Deconvolved Data")
# ax.plot(distances_au,deconvolved_surface_brightness_f360m/1e3,color='C3',label="F360M - Deconvolved Data",linestyle='--')

# # The Data
# ax.plot(distances_au,normalized_surface_brightness_f300m/1e3*conversion_factor_f300m,color='C1',label="F300M - Data")
# ax.plot(distances_au,normalized_surface_brightness_f360m/1e3*conversion_factor_f300m,color='C1',label="F360M - Data",linestyle='--')

# legend_line_f300m = mlines.Line2D([], [], color='k', label='F300M MCMC Samples')
# legend_line_f360m = mlines.Line2D([], [], color='k', label='F360M MCMC Samples',linestyle='--')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles + [legend_line_f300m,legend_line_f360m], labels + ['F300M MCMC Samples', "F360M MCMC Samples"],ncol=2)

# #Plot the Rings
# for level in levels: 
#     axes[0].axvline(level,linestyle='--',color='k',alpha=0.5)
#     axes[1].axvline(level,linestyle='--',color='k',alpha=0.5)
# axes[0].text(levels[0]-20,4e-4/1e3*conversion_factor_f300m,"C",alpha=0.5)
# axes[0].text(levels[1]-20,4e-4/1e3*conversion_factor_f300m,"B",alpha=0.5)
# axes[0].text(levels[2]-20,4e-4/1e3*conversion_factor_f300m,"A",alpha=0.5)

# axes[0].text(levels[0]-20,2e-2,"C",alpha=0.5)
# axes[0].text(levels[1]-20,2e-2,"B",alpha=0.5)
# axes[0].text(levels[2]-20,2e-2,"A",alpha=0.5)


# #The Coronagraph IWA = 0.64" for MASK335R
# axes[0].axvline(0.64*dstar,linestyle='--',color='r',alpha=0.3)

# axes[0].text(0.64*dstar+10,2e-2,"50% throughput",color='r',alpha=0.3)

# axes[0].set_ylim(1e-8,4e-2,)

ax=axis
axes[1].axvline(0.64*dstar,linestyle='--',color='r',alpha=0.3)
ax.set_xlabel("Deprojected Radial Distance (au)")
# ax.set_ylabel("Flux Ratio")
ax.set_ylabel("F300M/F360M Scattering Efficiency Ratio")
# ax.plot(distances_au,ratio_profiles[i],alpha=0.05,color='k', label="MCMC Samples")
# for i in range(0,n_samples):
    # ax.plot(distances_au,ratio_profiles[i],alpha=0.05,color='k')
# Create a separate line without alpha for the legend

# ax.plot(distances_au, raw_ratio_profile, color='C2', label="Unconvolved Best Fit Model",linestyle='--')
# ax.plot(kellen_delta_mag[:,0], kellen_flux_ratio, color='C3', label="Deconvolution",linestyle='--')
ax.plot(distances_au, deconvolved_ratio_profile_test, color='C3', label="Deconvolution",linestyle='--')
# ax.plot(distances_au, data_ratio_profile, color='C1', label="Data")
ax.set_ylim(0,1)

# Add legend with all handles
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles + [legend_line], labels + ['MCMC Samples'], loc='lower right')

ax.fill_between([0,150],0,1,color='grey',alpha=0.3)

plt.tight_layout()
plt.show()
fig.savefig("flux_ratio_surface_brightness_profiles_raw_4chen.png",dpi=300)

# ax.legend()
# fig.savefig("flux_ratio_profiles.png")
#Convert the ratios to astronomical delta magnitude, and convert the x-axis to astronomical units using the pixel scale and distance

##############################################################
# ######## Plot the delta magnitude profiles for each sample ###
# ##############################################################
# #Read in kellen_delta_mag.csv with numpy
# kellen_delta_mag = np.genfromtxt("kellen_delta_mags.csv",delimiter=',')



# delta_mags = -2.5*np.log10(np.array(ratio_profiles))
# delta_mag_data = -2.5*np.log10(data_ratio_profile)
# delta_mag_raw = -2.5*np.log10(raw_ratio_profile)
# # fig = plt.figure(figsize=(6,4))
# ax2 = axes[1]
# ax2.set_xlabel("Distance (au)")
# ax2.set_ylabel("Delta Magnitude")
# ax2.set_title("Delta Magnitude Profiles")
# for i in range(n_samples):
#     ax2.plot(distances_au,delta_mags[i],alpha=0.1,color='k')
# ax2.plot(distances_au,delta_mag_data,color='C1')
# ax2.plot(distances_au,delta_mag_raw,color='C2',linestyle='--')
# ax2.plot(kellen_delta_mag[:,0],kellen_delta_mag[:,1],color='C3',linestyle='--')
# ax2.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig("flux_ratio_surface_brightness_profiles_raw.png")



# fig, axes = plt.subplots(1,3,figsize=(10,4))
# vmin=-3
# vmax=3
# im0 = axes[0].imshow(sci2_data_f300m-sci1_data_f300m,vmin=vmin,vmax=vmax,cmap='RdBu')
# axes[0].set_title("Roll 2 - Roll 1")

# im1 = axes[1].imshow((sci2_data_f300m-model2_f300m)-(sci1_data_f300m-model1_f300m),vmin=vmin,vmax=vmax,cmap='RdBu')
# axes[1].set_title("Roll 2 - Roll 1: Model Subtracted")

# new_cent = (jnp.array(cent[::-1])-jnp.array([2.75,13.2])/1000*pixel_scale)*osamp #Offsets from Jens
# roll1_shifted = pyklip_rotate((sci1_data_f300m-model1_f300m), 0, jnp.array(cent[::-1])*osamp,new_center = new_cent)
# im1 = axes[2].imshow((sci2_data_f300m-model2_f300m)-roll1_shifted,vmin=vmin,vmax=vmax,cmap='RdBu')
# axes[2].set_title("Roll 2 - Shifted Roll 1: Model Subtracted")
# plt.show()

