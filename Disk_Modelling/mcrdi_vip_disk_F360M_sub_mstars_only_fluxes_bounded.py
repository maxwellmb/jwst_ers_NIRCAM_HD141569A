## This version only has the fluxes, offsets and PSF weights as free parameters, but fixes the geometry and SPFs to those of F300M. 

from datetime import datetime 
import glob 
from jax import config, grad
from jaxopt import ScipyMinimize

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'

import jax.numpy as jnp
from deconvolution_utils import *
from astropy.io import fits

from nircam_disk_utils import * 

from mcrdi_plots import plot_full_model

from vip_scattered_light_disk_jaxed import ScatteredLightDisk,compute_scattered_light_image_hg3

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

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

sci1_filename = datadir_f360m+"jw01386117001_03107_00001_nrcalong_calints_mstar_subtracted.fits"
sci2_filename = datadir_f360m+"jw01386118001_03107_00001_nrcalong_calints_mstar_subtracted.fits"

sci1_data,sci1_err = read_fits(sci1_filename)
sci2_data,sci2_err = read_fits(sci2_filename)

data_shape = sci1_data.shape

#########################################
####### Read in the M-star Models #######
#########################################

# m_star_dir = "/home/maxmb/Library/jwst_hd141569a_lib/mstar_models/"
# m_star_roll1_filename = "bcmodel_jw01386117001_03107_00001_nrcalong_calints.fits"
# m_star_roll1_data = fits.open(m_star_dir+m_star_roll1_filename)[0].data.astype(jnp.float32)


# m_star_roll2_filename = "bcmodel_jw01386118001_03107_00001_nrcalong_calints.fits"
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

# y1 = 70
# y2 = 270
# x1 = 50
# x2 = 250
sci1_data_crop = sci1_data[y1:y2,x1:x2].astype(jnp.float32)
sci2_data_crop = sci2_data[y1:y2,x1:x2].astype(jnp.float32)
sci1_err_crop = sci1_err[y1:y2,x1:x2].astype(jnp.float32)
sci2_err_crop = sci2_err[y1:y2,x1:x2].astype(jnp.float32)

data_crop_shape = sci1_data_crop.shape

osamp=2
######################################################
########## Read in the reference data files ##########
######################################################

ref_filenames = sorted(glob.glob(datadir_f360m+"jw01386116001_03108_0000?_nrcalong_calints.fits"))
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
a0_1 = 40. # semimajoraxis of the inner disk in au
a0_2 = 220. # semimajoraxis of the inner disk in au
a0_3 = 410. # semimajoraxis of the inner disk in au
# ### That weird extra disk thing
a0_4 = 300.

a0_1 = 40.0  # semimajoraxis of the inner disk in au
a0_2 = 220.0  # semimajoraxis of the inner disk in au
a0_3 = 510.0  # semimajoraxis of the inner disk in au
a0_4 = 330.0


######### Relative flux ###########
f0_1 = 500*120.*10/10./3*4.
f0_2 = 50*250.*10/5./3/2
f0_3 = 5000./20
f0_4 = 50*150.*8/50./10/4

f0_1 = 1.5e5
f0_2 = 2.5e3
f0_3 = 140
f0_4 = 410

# new_flux_scaling = 1.01
# f0_1 *= new_flux_scaling
# f0_2 *= new_flux_scaling
# f0_3 *= new_flux_scaling
# f0_4 *= new_flux_scaling


# ################# Scattering phase thing #####################
n_nodes = 6

#Get the starting guesse for other parameters from a previous F360M fit. 
# x0_old =jnp.load("../hg3fit_F360M.npz.npy")
# x0_old = jnp.delete(x0_old,18+4*n_nodes) #Knock out two parameters that aren't used anymore. 
# x0_old = jnp.delete(x0_old,19+4*n_nodes) #delete the last element which is the scaling factor for the PSF

# ### Get the morphology from an F300M fit and shoe-horn it into the x0_old
# # xF300M =jnp.load("hg3fit_F300M_m_stars.npz.npy")
# xF300M = jnp.load("../230613/hg3fit_F300M_m_stars_bounded.npz.npy")
# x0_new = x0_old[2]
# x0_new = jnp.append(x0_new,x0_old[n_nodes+6])
# x0_new = jnp.append(x0_new,x0_old[2*n_nodes+10])
# x0_new = jnp.append(x0_new,x0_old[3*n_nodes+14])
# x0_new = jnp.append(x0_new,x0_old[4*n_nodes+18:4*n_nodes+22])
# x0_new = jnp.append(x0_new,x0_old[-2*n_refs:])

# #After the first run we can load this up. 
x0_new = jnp.load("../230613/hg3fit_F360M_m_stars_only_fluxes.npz.npy")

# #Create a full parameter list to help with plotting. 
# x0_full = jnp.copy(xF300M)
# x0_full = x0_full.at[2].set(x0_new[0])
# x0_full = x0_full.at[n_nodes+6].set(x0_new[1])
# x0_full = x0_full.at[2*n_nodes+10].set(x0_new[2])
# x0_full = x0_full.at[3*n_nodes+14].set(x0_new[3])
# x0_full = x0_full.at[4*n_nodes+18:4*n_nodes+22].set(x0_new[4:8])
# x0_full = x0_full.at[-2*n_refs:].set(x0_new[-2*n_refs:])

x0_f360m_fluxes = jnp.load("../231005/hg3fit_F360M_m_stars_only_fluxes.npz.npy")
x0_f360m_fluxes = jnp.delete(x0_f360m_fluxes,8) #Get rid of obsolute parameter. 
x0_f360m = jnp.copy(jnp.load("../231005/hg3fit_F300M_m_stars_bounded.npy")) #We'll reuse many of the morphological parameters.
x0_f360m = x0_f360m.at[2].set(x0_f360m_fluxes[0])
x0_f360m = x0_f360m.at[n_nodes+6].set(x0_f360m_fluxes[1])
x0_f360m = x0_f360m.at[2*n_nodes+10].set(x0_f360m_fluxes[2])
x0_f360m = x0_f360m.at[3*n_nodes+14].set(x0_f360m_fluxes[3])
x0_f360m = x0_f360m.at[4*n_nodes+18:4*n_nodes+22].set(x0_new[4:8])
x0_f360m = x0_f360m.at[-2*n_refs:].set(x0_new[-2*n_refs:])

x0_f360m = jnp.delete(x0_f360m,slice(4*n_nodes+22, 4*n_nodes+34)) #Get rid of obsolute parameter. 

x0_f360m_fluxes = x0_f360m_fluxes.at[4:8].set(x0_new[4:8])
x0_f360m_fluxes = x0_f360m_fluxes.at[-2*n_refs:].set(x0_new[-2*n_refs:]) #Reference weights
# x0_f360m = jnp.delete(x0_f360m,np.s_[4*n_nodes+18:22+5*n_nodes])
# x0_f360m = jnp.delete(x0_f360m,np.s_[-2:])

x0_full = jnp.copy(x0_f360m)

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

nircam_psf_list = [psfs,psf_inds,im_mask,unique_inds]

##########################################
##### Generate starting guess model ######
##########################################

print("Generating first guess model")
model1, model2 = gen_roll_images_hg3_w_psf(x0_f360m,ref_data,cent,nircam_psf_list,
                                           f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                        a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                        nx=nx,ny=ny,n_nodes=6)

################################
##### Show starting guess ######
################################
show_start = True
if show_start:
    ref_psf_data1, ref_psf_data2 = gen_psf_model(x0_f360m,ref_data)
    plot_full_model(model1,model2,ref_psf_data1,ref_psf_data2,sci1_data_crop,sci2_data_crop,savefile=False)

# chi2_0 = chi2_hg3(x0,
#               sci1_data_crop,sci1_err_crop,
#               sci2_data_crop,sci2_err_crop,
#               ref_data,cent,data_crop_shape,nircam_psf_list,
#               f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
#               a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)
chi2_0 = chi2_hg3_just_flux(x0_f360m_fluxes,x0_f360m,
                            sci1_data_crop,sci1_err_crop,
                            sci2_data_crop,sci2_err_crop,
                            ref_data,cent,data_crop_shape,nircam_psf_list,
                            f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)
print("Starting chi2: {}".format(chi2_0))
###################
##### Do Fit ######
###################

do_fit = False
# params=x0
if do_fit: 
    maxiter = 2
    # print("Starting grad test")
    t = datetime.now()


    print("Starting minimization: {}".format(datetime.now()))
    start=datetime.now()

    lbfgsb = ScipyMinimize(fun=chi2_hg3_just_flux, method="l-bfgs-b",maxiter=maxiter,jit=False)
    result = lbfgsb.run(x0_f360m_fluxes,x0_f360m,sci1_data_crop,sci1_err_crop,
              sci2_data_crop,sci2_err_crop,
              ref_data,cent,data_crop_shape,nircam_psf_list,
              f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
              a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)

    params = result.params
    params_reformat = jnp.array(params)

    chi2_final = chi2_hg3_just_flux(params,x0_f360m,sci1_data_crop,sci1_err_crop,
              sci2_data_crop,sci2_err_crop,
              ref_data,cent,data_crop_shape,nircam_psf_list,
              f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
              a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,)
    end=datetime.now()
    print("It took {} for {} iterations".format(end-start,result.state[3]))
    print("Best Fit chi2: {}".format(chi2_final))
          
    
    
    params_filename = "hg3fit_F360M_m_stars_only_fluxes.npz"
    print("Saving best-fit paramters to {}".format(params_filename))
    jnp.save(params_filename,params)

    params_full = jnp.copy(x0_full)
    params_full = params_full.at[2].set(params[0])
    params_full = params_full.at[n_nodes+6].set(params[1])
    params_full = params_full.at[2*n_nodes+10].set(params[2])
    params_full = params_full.at[3*n_nodes+14].set(params[3])
    params_full = params_full.at[4*n_nodes+18:4*n_nodes+22].set(params[4:8])
    params_full = params_full.at[-2*n_refs:].set(params[-2*n_refs:])
    # print_params(params_full)


    params_filename = "hg3fit_F360M_full_params"
    print("Saving best-fit paramters to {}".format(params_filename))
    jnp.save(params_filename,params_full)


    plot_fit = True
    if plot_fit:

        ###############################################################
        ######## Plot convolved models, data and residuals ############
        ###############################################################
        fig, axes = plt.subplots(2,4,figsize=(20,7))
        axes = axes.flatten()

        model1, model2 = gen_roll_images_hg3_w_psf(params_full,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
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
        