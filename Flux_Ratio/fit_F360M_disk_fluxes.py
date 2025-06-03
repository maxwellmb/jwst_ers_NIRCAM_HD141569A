import numpy as np

import cmasher as cmr # Import CMasher to register colormaps

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

from nircam_disk_utils import * 

import jax.numpy as jnp

################################################################
########## Read in the two rolls from F360M ##########
################################################################

f360m_roll1_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/jw01386117001_03107_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"
f360m_roll2_filename = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/jw01386118001_03107_00001_nrcalong_calints_mstar_subtracted_MCRDI_psf_subtracted.fits"

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
sci1_data_f360m = f360m_roll1_data[y1:y2,x1:x2].astype(jnp.float64)
sci2_data_f360m = f360m_roll2_data[y1:y2,x1:x2].astype(jnp.float64)
#Crop the errors too
sci1_err_f360m = f360m_roll1_err[y1:y2,x1:x2].astype(jnp.float64)
sci2_err_f360m = f360m_roll2_err[y1:y2,x1:x2].astype(jnp.float64)

data_crop_shape = sci1_data_f360m.shape

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

########################################################
######### Scaling values for some parameters ###########
########################################################

############# SMA ##############
#Parameters from October 2024
a0_1_f360m = 40.0  # semimajoraxis of the inner disk in au
a0_2_f360m = 220.0  # semimajoraxis of the inner disk in au
a0_3_f360m = 510.0  # semimajoraxis of the inner disk in au
a0_4_f360m = 330.0

################# Relative flux #####################
#Parameters from October 2024
f0_1_f360m = 1.5e5
f0_2_f360m = 2.5e3
f0_3_f360m = 140
f0_4_f360m = 410


###############################################
######## Generate some initial models #########
###############################################
print("Generating Base Model")
# x0_f300m = jnp.load("../230613/hg3fit_F300M_m_stars_bounded.npz.npy")
x0_f360m = jnp.load("../Disk_Modelling/hg3fit_F360M_full_params.npy")

model2_f360m, model1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                    f0_1=f0_1_f360m,f0_2=f0_2_f360m,f0_3=f0_3_f360m,f0_4=f0_4_f360m,
                                                    a0_1=a0_1_f360m,a0_2=a0_2_f360m,a0_3=a0_3_f360m,a0_4=a0_4_f360m)

disk1_roll2_f360m, disk1_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                              f0_1=f0_1_f360m,f0_2=0,f0_3=0,f0_4=0,
                                                                a0_1=a0_1_f360m,a0_2=a0_2_f360m,a0_3=a0_3_f360m,a0_4=a0_4_f360m)

disk2_roll2_f360m, disk2_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=f0_2_f360m,f0_3=0,f0_4=0,
                                                                    a0_1=a0_1_f360m,a0_2=a0_2_f360m,a0_3=a0_3_f360m,a0_4=a0_4_f360m)

disk3_roll2_f360m, disk3_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=0,f0_3=f0_3_f360m,f0_4=0,
                                                                    a0_1=a0_1_f360m,a0_2=a0_2_f360m,a0_3=a0_3_f360m,a0_4=a0_4_f360m)

disk4_roll2_f360m, disk4_roll1_f360m = gen_roll_images_hg3(x0_f360m,cent,nircam_psf_list_f360m,
                                                                f0_1=0,f0_2=0,f0_3=0,f0_4=f0_4_f360m,
                                                                    a0_1=a0_1_f360m,a0_2=a0_2_f360m,a0_3=a0_3_f360m,a0_4=a0_4_f360m)

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
# xx_crop,yy_crop = np.meshgrid(x_crop,y_crop)



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

plot_first_guess = True
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


#Positions of the first walkers
pos = first_guess + 1e-4 * np.random.randn(nwalkers, ndim)

sampler_f360m = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_f360m, args=(sci1_data_f360m, sci1_err_f360m, sci2_data_f360m, sci2_err_f360m)
)

#For Testing
# burnin=10
# nsteps = 100

#For run-time
burnin= 400
nsteps = 3000

print("Starting MCMC Burn-in with {} steps".format(burnin))
state = sampler_f360m.run_mcmc(pos, burnin,progress=True)
sampler_f360m.reset()
print("Starting main MCMC run with {} steps".format(nsteps))
sampler_f360m.run_mcmc(state, nsteps, progress=True)

samples = sampler_f360m.get_chain()
labels = ["Ring 1 Scale", "Ring 2 Scale", "Ring 3 Scale", "Ring 4 Scale", "Fractional Variance\nIncrease"]

tau = sampler_f360m.get_autocorr_time()
print("Autocorrelation times: {}".format(tau))
max_tau = np.max(tau).astype(int)

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
    axes.set_title("f360m Roll 1\n(Data - Model) / Error")
    cbar = fig.colorbar(im3,ax=axes)
    cbar.set_label(label=r'$\sigma$',size=12)
    # fig.suptitle("f360m")
    axes.set_ylabel(r"$\Delta y ('')$",fontsize=12)
    axes.set_xlabel(r"$\Delta x ('')$",fontsize=12)
    plt.savefig("F360M_residuals_sigma.png")

    


plot_chains_corner = False
if plot_chains_corner: 
    import corner
    figure = corner.corner(flat_samples_f360m,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True,title_fmt='.3f')
    plt.savefig("f360m_corner.png")

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")


plt.show()