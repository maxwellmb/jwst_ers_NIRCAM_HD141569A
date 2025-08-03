from datetime import datetime
import glob 
from jax import config, grad
from jaxopt import ScipyMinimize,ScipyBoundedMinimize

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'


import jax.numpy as jnp
from deconvolution_utils import *
from astropy.io import fits

from nircam_disk_utils import * 

from mcrdi_plots import plot_full_model,plot_disk_model_resids

from vip_scattered_light_disk_jaxed import ScatteredLightDisk,compute_scattered_light_image_hg3

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

########################################################
########## Read in the two science data files ##########
########################################################

new=True
if new: 
    suffix = ""
    datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/231108/"
    datadir_f360m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/231108/"
else: 
    suffix = "_older"
    datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/older/"
    datadir_f360m = "/home/maxmb/Library/jwst_hd141569a_lib/data/F360M/jk/older/"

# datadir = "/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/older/"
# sci1_filename = datadir+"jw01386117001_03106_00001_nrcalong_calints.fits"
# sci2_filename = datadir+"jw01386118001_03106_00001_nrcalong_calints.fits"
sci1_filename = datadir+"jw01386117001_03106_00001_nrcalong_calints_mstar_subtracted.fits"
sci2_filename = datadir+"jw01386118001_03106_00001_nrcalong_calints_mstar_subtracted.fits"


sci1_data,sci1_err = read_fits(sci1_filename)
sci2_data,sci2_err = read_fits(sci2_filename)

data_shape = sci1_data.shape

#########################################
####### Read in the M-star Models #######
#########################################

# m_star_dir = "/home/maxmb/Library/jwst_hd141569a_lib/mstar_models/"
# m_star_roll1_filename = "bcmodel_jw01386117001_03106_00001_nrcalong_calints.fits"
# m_star_roll1_data = fits.open(m_star_dir+m_star_roll1_filename)[0].data.astype(jnp.float32)


# m_star_roll2_filename = "bcmodel_jw01386118001_03106_00001_nrcalong_calints.fits"
# m_star_roll2_data = fits.open(m_star_dir+m_star_roll2_filename)[0].data.astype(jnp.float32)

# #subtract it off the data
# sci1_data = sci1_data - m_star_roll1_data
# sci2_data = sci2_data - m_star_roll2_data

###################################
########## Crop the data ##########
###################################
y1 = 100
y2 = 240
x1 = 80
x2 = 220
sci1_data_crop = sci1_data[y1:y2,x1:x2].astype(jnp.float32)
sci2_data_crop = sci2_data[y1:y2,x1:x2].astype(jnp.float32)
sci1_err_crop = sci1_err[y1:y2,x1:x2].astype(jnp.float32)
sci2_err_crop = sci2_err[y1:y2,x1:x2].astype(jnp.float32)

data_crop_shape = sci1_data_crop.shape

osamp=2
######################################################
########## Read in the reference data files ##########
######################################################

ref_filenames = sorted(glob.glob(datadir+"jw01386116001_03106_0000?_nrcalong_calints.fits"))
n_refs = len(ref_filenames)

ref_data = jnp.zeros([n_refs,y2-y1,x2-x1])
scales = jnp.zeros([2*n_refs])

for i,ref_filename in enumerate(ref_filenames): 
    this_ref_data = read_fits(ref_filename,return_error=False)
    ref_data = ref_data.at[i,:,:].set(this_ref_data[y1:y2,x1:x2])

    ###First guess scaling factor for each PSF
    scales = scales.at[i].set(jnp.max(sci1_data_crop)/jnp.max(ref_data[i,:,:])) #science 1 indices
    scales = scales.at[i+n_refs].set(jnp.max(sci2_data_crop)/jnp.max(ref_data[i,:,:])) #science 2 indices

######################################
######### Simulation Setup ###########
######################################
pixel_scale=0.063 # nircam pixel scale in arcsec/px
nx = x2-x1 # number of pixels of your image in X
ny = y2-y1 # number of pixels of your image in Y
cent = [173.4-y1,149.1-x1] #[y_cent,x_cent] #Cropped cent

########################################################
######### Scaling values for some parameters ###########
########################################################

############# SMA ##############
#Parameters from October
a0_1 = 40.0  # semimajoraxis of the inner disk in au
a0_2 = 220.0  # semimajoraxis of the inner disk in au
a0_3 = 510.0  # semimajoraxis of the inner disk in au
a0_4 = 330.0

################# Relative flux #####################
#Parameters from October
f0_1 = 3e4
f0_2 = 7.5e3
f0_3 = 140
f0_4 = 410

# A scaling factor because the new data slightly brighter than the old data
new_flux_scaling = 1.23
f0_1 *= new_flux_scaling
f0_2 *= new_flux_scaling
f0_3 *= new_flux_scaling
f0_4 *= new_flux_scaling

# ################# Scattering phase thing #####################
n_nodes = 6

### Starting point based on a previous fit. 
# x0 = jnp.load("../231108/hg3fit_F300M_m_stars_bounded_quad.npz.npy")
x0 = jnp.load("hg3fit_F300M_m_stars_bounded_quad_new.npz.npy")

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

nircam_psf_list = [psfs,psf_inds,im_mask,unique_inds]

##########################################
##### Generate starting guess model ######
##########################################

print("Generating first guess model")
model1, model2 = gen_roll_images_hg3_w_psf(x0,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                        a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                        nx=nx,ny=ny,n_nodes=6)

################################
##### Show starting guess ######
################################
show_start = True
if show_start:
    ref_psf_data1, ref_psf_data2 = gen_psf_model(x0,ref_data)
    # plot_full_model(model1,model2,ref_psf_data1,ref_psf_data2,sci1_data_crop,sci2_data_crop,savefile=False)
    plot_disk_model_resids(model1,model2,ref_psf_data1,ref_psf_data2,sci1_data_crop,sci2_data_crop,
                           savefile=False,vmax1=50)

chi2_0 = chi2_hg3(x0,
              sci1_data_crop,sci1_err_crop,
              sci2_data_crop,sci2_err_crop,
              ref_data,cent,data_crop_shape,nircam_psf_list,
              f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
              a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)
print("Starting chi2: {}".format(chi2_0))

#Save the science data minus the models to fits files: 
# save_start = True
# if save_start: 
#     fits.writeto("sci1_data_minus_model.fits",np.array(sci1_data_crop-model1),overwrite=True)
#     fits.writeto("sci2_data_minus_model.fits",np.array(sci2_data_crop-model2),overwrite=True)
#     fits.writeto("sci1_minus_sci2.fits",np.array(sci1_data_crop-model1)-np.array(sci2_data_crop-model2),overwrite=True)

###################
##### Do Fit ######
###################

# grad_test = grad(chi2_hg3)(x0,sci1_data_crop,sci1_err_crop,
#             sci2_data_crop,sci2_err_crop,
#             ref_data,cent,data_crop_shape,nircam_psf_list,
#             f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
#             a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)


do_fit = False
# params=x0
if do_fit: 
    maxiter = 5000
    # maxiter = 2
    # print("Starting grad test")
    t = datetime.now()
    # grad_test = grad(chi2)(x0,sci1_data_crop,sci1_err_crop,
    #           sci2_data_crop,sci2_err_crop,
    #           ref_data,cent,data_crop_shape,nircam_psf_list,
    #           f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_2_5,
    #           a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_2_5,)
    # print("chi2 grad test : {}".format(datetime.now()-t))

    print("Starting minimization: {}".format(datetime.now()))
    start=datetime.now()

    # lbfgsb = ScipyMinimize(fun=chi2_hg3, method="l-bfgs-b",maxiter=maxiter,jit=False)
    # result = lbfgsb.run(x0,sci1_data_crop,sci1_err_crop,
    #           sci2_data_crop,sci2_err_crop,
    #           ref_data,cent,data_crop_shape,nircam_psf_list,
    #           f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_2_5,
    #           a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_2_5,)


    # lbfgsb = ScipyBoundedMinimize(fun=chi2_hg3, method="l-bfgs-b",maxiter=maxiter,jit=False)
    lbfgsb = ScipyBoundedMinimize(fun=chi2_hg3, method="l-bfgs-b",maxiter=maxiter,jit=False,options={'ftol':1e-12})
    lower_bounds = jnp.zeros_like(x0)* jnp.inf * -1
    upper_bounds = jnp.ones_like(x0) * jnp.inf

    ### HG Bounds ###
    lower_bounds = lower_bounds.at[6:6+n_nodes].set([-1,-1,-1,0,0,0])
    lower_bounds = lower_bounds.at[10+n_nodes:10+2*n_nodes].set([-1,-1,-1,0,0,0])
    lower_bounds = lower_bounds.at[14+2*n_nodes:14+3*n_nodes].set([-1,-1,-1,0,0,0])
    lower_bounds = lower_bounds.at[18+3*n_nodes:18+4*n_nodes].set([-1,-1,-1,0,0,0])

    upper_bounds = upper_bounds.at[6:6+n_nodes].set([1,1,1,jnp.inf,jnp.inf,jnp.inf])
    upper_bounds = upper_bounds.at[10+n_nodes:10+2*n_nodes].set([1,1,1,jnp.inf,jnp.inf,jnp.inf])
    upper_bounds = upper_bounds.at[14+2*n_nodes:14+3*n_nodes].set([1,1,1,jnp.inf,jnp.inf,jnp.inf])
    upper_bounds = upper_bounds.at[18+3*n_nodes:18+4*n_nodes].set([1,1,1,jnp.inf,jnp.inf,jnp.inf])
    # gs_ws_1 = jnp.array(x[6:6+n_nodes]) #HG g and w parameters of disk 1
    # gs_ws_2 = jnp.array(x[10+n_nodes:10+2*n_nodes]) #HG g and w parameters of disk 2
    # gs_ws_3 = jnp.array(x[14+2*n_nodes:14+3*n_nodes]) #HG g and w parameters of disk 3
    # gs_ws_4 = jnp.array(x[18+3*n_nodes:18+4*n_nodes]) #HG g and w parameters of disk 4


    bounds = (lower_bounds, upper_bounds)
    result = lbfgsb.run(x0,bounds,sci1_data_crop,sci1_err_crop,
            sci2_data_crop,sci2_err_crop,
            ref_data,cent,data_crop_shape,nircam_psf_list,
            f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4)
    # lbfgsb_sol = lbfgsb.run(w_init, bounds=bounds, data=(X, y)).params

    params = result.params
    params_reformat = jnp.array(params)

    chi2_final = chi2_hg3(params,sci1_data_crop,sci1_err_crop,
              sci2_data_crop,sci2_err_crop,
              ref_data,cent,data_crop_shape,nircam_psf_list,
              f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
              a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)
    end=datetime.now()
    print("It took {} for {} iterations".format(end-start,result.state[3]))
    print("Best Fit chi2: {}".format(chi2_final))
          
    print_params(params)
    
    params_filename = "hg3fit_F300M_m_stars_bounded_quad_new.npz"
    print("Saving best-fit paramters to {}".format(params_filename))
    jnp.save(params_filename,params)
    plot_fit = True
    if plot_fit:

        ###############################################################
        ######## Plot convolved models, data and residuals ############
        ###############################################################
        fig, axes = plt.subplots(2,4,figsize=(20,7))
        axes = axes.flatten()

        model1, model2 = gen_roll_images_hg3_w_psf(params,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                        a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                        nx=nx,ny=ny)
        ref_psf_data1, ref_psf_data2 = gen_psf_model(params,ref_data)
        plot_full_model(model1,model2,ref_psf_data1,ref_psf_data2,sci1_data_crop,sci2_data_crop,savefile=False)

        ###########################################
        ############ Plot the raw model ###########
        ###########################################

        # fig2,axes2 = plt.subplots(1,2,)
        
        # ima20 = axes2[0].imshow(model1)
        # plt.colorbar(ima20,ax=axes2[0])

        # ima21 = axes2[1].imshow(model1,norm=LogNorm())
        # plt.colorbar(ima21,ax=axes2[1])
        # plt.savefig("raw_model.png")


        ######################################
        ############ Plot the spfs ###########
        ######################################

        # fig3 = plt.figure()
        
        # plt.plot(cosphi_nodes,params[6:6+n_nodes],label="Disk1")
        # plt.plot(cosphi_nodes,params[10+n_nodes:10+2*n_nodes],label="Disk2")
        # plt.plot(cosphi_nodes,params[14+2*n_nodes:14+3*n_nodes],label="Disk3")
        # plt.plot(cosphi_nodes,params[18+3*n_nodes:18+4*n_nodes],label="Disk2.5")
        