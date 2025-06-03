import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from jax import jit
from deconvolution_utils import *

import matplotlib.pyplot as plt

from vip_scattered_light_disk_jaxed import ScatteredLightDisk,compute_scattered_light_image_hg3,compute_scattered_light_image_hg2

gen_image_hg3 = jit(compute_scattered_light_image_hg3)
gen_image_hg2 = jit(compute_scattered_light_image_hg2)


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

# def read_webbpsf_psfs(psfs_filename = "psfs.npy",psf_offsets_filename = "psf_offsets.npy",
#                       im_mask_filename = "im_mask_rolls.npy",psf_inds_filename="psf_inds_rolls.npy",
#                       osamp=2):
#     '''
#     Read in the previously generated webbpsfs
#     '''
#     im_mask_rolls = jnp.load("im_mask_rolls.npy")[0]
#     if x1 is not None and x2 is not None:
#         im_mask = im_mask_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
#         psf_inds_rolls = jnp.load("psf_inds_rolls.npy")
#         psf_inds = psf_inds_rolls[0][y1*osamp:y2*osamp,x1*osamp:x2*osamp]
#     psf_offsets = jnp.load(psf_offsets_filename)
#     psfs = jnp.load(psfs_filename)

def single_hg3_disk_model(inc,pa,alpha_in,alpha_out,a, cent,gs_ws,flux_scaling,
                    ksi0=3.,gamma=2.,beta=1.,dstar=111.61,
                    nx=140,ny=140,pixel_scale=0.063,n_nodes=6):
    '''
    Make a single disk model! 

    Note the scattering phase function input to ScatteredLightDisk is overwritten
    '''

    #The ScatteredLightDisk object
    disk_sld = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=inc, omega=0, pxInArcsec=pixel_scale, pa=pa,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'HG', 'g':0., 'polar':False},
                                flux_max=1,cent=cent)
    
    #The scattering phase function object
    density = disk_sld.dust_density.dust_distribution_calc

    #Generate the disk image
    disk_image = gen_image_hg3(disk_sld.y_map,
                            disk_sld.x_map,
                            disk_sld.itilt,
                            disk_sld.dust_density.dust_distribution_calc.zmax,
                            disk_sld.cosi,
                            disk_sld.sini,
                            disk_sld.xdo,
                            disk_sld.ydo,
                            disk_sld.omega,
                            disk_sld.pxInAU,
                            density.a,
                            density.ain,
                            density.aout,
                            density.amin,
                            density.dens_at_r0,
                            density.ksi0,
                            density.beta,
                            density.gamma,
                            gs_ws[:n_nodes//2], gs_ws[n_nodes//2:],
                            disk_sld.rmin,
                            density.rmax,
                            )
    
    return flux_scaling*disk_image

def single_hg2_disk_model(inc,pa,alpha_in,alpha_out,a, cent,gs_ws,flux_scaling,
                    ksi0=3.,gamma=2.,beta=1.,dstar=111.61,
                    nx=140,ny=140,pixel_scale=0.063):
    '''
    Make a single disk model! 

    Note the scattering phase function input to ScatteredLightDisk is overwritten
    '''

    #The ScatteredLightDisk object
    disk_sld = ScatteredLightDisk(nx=nx, ny=ny, distance=dstar,
                                itilt=inc, omega=0, pxInArcsec=pixel_scale, pa=pa,
                                density_dico={'name':'2PowerLaws','ain':alpha_in,'aout':alpha_out,
                                              'a':a,'e':0.0,'ksi0':ksi0,'gamma':gamma,'beta':beta},
                                spf_dico={'name':'HG', 'g':0., 'polar':False},
                                flux_max=1,cent=cent)
    
    #The scattering phase function object
    density = disk_sld.dust_density.dust_distribution_calc

    #Generate the disk image
    disk_image = gen_image_hg2(disk_sld.y_map,
                            disk_sld.x_map,
                            disk_sld.itilt,
                            disk_sld.dust_density.dust_distribution_calc.zmax,
                            disk_sld.cosi,
                            disk_sld.sini,
                            disk_sld.xdo,
                            disk_sld.ydo,
                            disk_sld.omega,
                            disk_sld.pxInAU,
                            density.a,
                            density.ain,
                            density.aout,
                            density.amin,
                            density.dens_at_r0,
                            density.ksi0,
                            density.beta,
                            density.gamma,
                            gs_ws[:2], gs_ws[2],
                            disk_sld.rmin,
                            density.rmax,
                            )
    
    return flux_scaling*disk_image

def quad_disk_model_hg3(x,cent,
                    f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                    a0_1=40, a0_2=220, a0_3=410, a0_4=300,
                    nx=140,ny=140,pixel_scale=0.063,
                    distance=111.61,ksi0=3.,gamma=2.,beta=1.,n_nodes=6, return_all = False):
    '''
    Generate the sum of four disks

    Assume the have the same inclination, position angle, gamma, beta, ksi0, and distance
    '''

    inc = jnp.degrees(x[0]) #Inclination in degrees
    pa = jnp.degrees(x[1]) #Position angle in degrees

    ### Disk 1 ###
    f_1 = x[2]*f0_1 #Flux scale of disk 1
    a_1 = x[3]*a0_1 #Radius scale of disk 1
    alpha_in_1 = x[4] #Inner radius power law index of disk 1
    alpha_out_1 = x[5] #Outer radius power law index of disk 1
    gs_ws_1 = jnp.array(x[6:6+n_nodes]) #HG g and w parameters of disk 1

    disk1_image = single_hg3_disk_model(inc,pa,alpha_in_1,alpha_out_1,a_1,cent,gs_ws_1,
                                        f_1,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale,n_nodes=n_nodes)


    ### Disk 2 ###
    f_2 = x[n_nodes+6]*f0_2 #Flux scale of disk 2
    a_2 = x[n_nodes+7]*a0_2 #Radius scale of disk 2
    alpha_in_2 = x[n_nodes+8] #Inner radius power law index of disk 2
    alpha_out_2 = x[n_nodes+9] #Outer radius power law index of disk 2
    gs_ws_2 = jnp.array(x[10+n_nodes:10+2*n_nodes]) #HG g and w parameters of disk 2

    disk2_image = single_hg3_disk_model(inc,pa,alpha_in_2,alpha_out_2,a_2,cent,gs_ws_2,
                                        f_2,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale,n_nodes=n_nodes)
    
    ### Disk 3 ###
    f_3 = x[2*n_nodes+10]*f0_3 #Flux scale of disk 3
    a_3 = x[2*n_nodes+11]*a0_3 #Flux scale of disk 4
    alpha_in_3 = x[2*n_nodes+12] #Inner radius power law index of disk 3
    alpha_out_3 = x[2*n_nodes+13] #Outer radius power law index of disk 3
    gs_ws_3 = jnp.array(x[14+2*n_nodes:14+3*n_nodes]) #HG g and w parameters of disk 3
    
    disk3_image = single_hg3_disk_model(inc,pa,alpha_in_3,alpha_out_3,a_3,cent,gs_ws_3,
                                        f_3,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale,n_nodes=n_nodes)
    
    ### Disk 4 ###
    f_4 = x[3*n_nodes+14]*f0_4 #Flux scale of disk 4
    a_4 = x[3*n_nodes+15]*a0_4 #Flux scale of disk 4
    alpha_in_4 = x[3*n_nodes+16] #Inner radius power law index of disk 4
    alpha_out_4 = x[3*n_nodes+17] #Outer radius power law index of disk 4
    gs_ws_4 = jnp.array(x[18+3*n_nodes:18+4*n_nodes]) #HG g and w parameters of disk 4
    
    disk4_image = single_hg3_disk_model(inc,pa,alpha_in_4,alpha_out_4,a_4,cent,gs_ws_4,
                                        f_4,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale,n_nodes=n_nodes)

    disk_image = disk1_image+disk2_image+disk3_image+disk4_image
    
    if return_all: 
        return disk_image, disk1_image, disk2_image, disk3_image, disk4_image
    else: 
        return disk_image

def quad_disk_model_hg2(x,cent,
                    f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                    a0_1=40, a0_2=220, a0_3=410, a0_4=300,
                    nx=140,ny=140,pixel_scale=0.063,
                    distance=111.61,ksi0=3.,gamma=2.,beta=1.,n_nodes=3):
    '''
    Generate the sum of four disks

    Assume the have the same inclination, position angle, gamma, beta, ksi0, and distance
    '''

    inc = x[0] #Inclination in degrees
    pa = x[1] #Position angle in degrees

    ### Disk 1 ###
    f_1 = x[2]*f0_1 #Flux scale of disk 1
    a_1 = x[3]*a0_1 #Radius scale of disk 1
    alpha_in_1 = x[4] #Inner radius power law index of disk 1
    alpha_out_1 = x[5] #Outer radius power law index of disk 1
    gs_ws_1 = jnp.array(x[6:6+n_nodes]) #HG g and w parameters of disk 1

    disk1_image = single_hg2_disk_model(inc,pa,alpha_in_1,alpha_out_1,a_1,cent,gs_ws_1,
                                        f_1,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale)

    ### Disk 2 ###
    f_2 = x[n_nodes+6]*f0_2 #Flux scale of disk 2
    a_2 = x[n_nodes+7]*a0_2 #Radius scale of disk 2
    alpha_in_2 = x[n_nodes+8] #Inner radius power law index of disk 2
    alpha_out_2 = x[n_nodes+9] #Outer radius power law index of disk 2
    gs_ws_2 = jnp.array(x[10+n_nodes:10+2*n_nodes]) #HG g and w parameters of disk 2

    disk2_image = single_hg2_disk_model(inc,pa,alpha_in_2,alpha_out_2,a_2,cent,gs_ws_2,
                                        f_2,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale)
    
    ### Disk 3 ###
    f_3 = x[2*n_nodes+10]*f0_3 #Flux scale of disk 3
    a_3 = x[2*n_nodes+11]*a0_3 #Flux scale of disk 4
    alpha_in_3 = x[2*n_nodes+12] #Inner radius power law index of disk 3
    alpha_out_3 = x[2*n_nodes+13] #Outer radius power law index of disk 3
    gs_ws_3 = jnp.array(x[14+2*n_nodes:14+3*n_nodes]) #HG g and w parameters of disk 3
    
    disk3_image = single_hg2_disk_model(inc,pa,alpha_in_3,alpha_out_3,a_3,cent,gs_ws_3,
                                        f_3,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale)
    
    ### Disk 4 ###
    f_4 = x[3*n_nodes+14]*f0_4 #Flux scale of disk 4
    a_4 = x[3*n_nodes+15]*a0_4 #Flux scale of disk 4
    alpha_in_4 = x[3*n_nodes+16] #Inner radius power law index of disk 4
    alpha_out_4 = x[3*n_nodes+17] #Outer radius power law index of disk 4
    gs_ws_4 = jnp.array(x[18+3*n_nodes:18+4*n_nodes]) #HG g and w parameters of disk 4
    
    disk4_image = single_hg2_disk_model(inc,pa,alpha_in_4,alpha_out_4,a_4,cent,gs_ws_4,
                                        f_4,ksi0=ksi0,gamma=gamma,beta=beta,dstar=distance,
                                        nx=nx,ny=ny,pixel_scale=pixel_scale)

    disk_image = disk1_image+disk2_image+disk3_image+disk4_image
    
    return disk_image

def fake_osamp(image,osamp=2):
    '''
    Create a fake oversampled image
    '''
    osamped_image = jnp.zeros([image.shape[0]*osamp,image.shape[1]*osamp])
    osamped_image = osamped_image.at[::2,::2].set(image)
    osamped_image = osamped_image.at[1::2,::2].set(image)
    osamped_image = osamped_image.at[::2,1::2].set(image)
    osamped_image = osamped_image.at[1::2,1::2].set(image)
    
    return osamped_image

def rotate_and_convolve_osamped(im,cent,output_shape,nircam_psf_list,osamp=2,
                                roll_angle_1=0., roll_angle_2=-5.,
                                dx1=0,dy1=0,dx2=0,dy2=0):
    '''
    Rotate a model image and then convolve it with the nircam PSF
    '''
    # im1 = pyklip_rotate(im[y1*osamp:y2*osamp,x1*osamp:x2*osamp], -280., jnp.array(crop_center))
    # im2 = pyklip_rotate(im, -285., jnp.array(cent[::-1]))

    new_cen1 = jnp.array([cent[1]+dx1,cent[0]+dy1])*osamp
    new_cen2 = jnp.array([cent[1]+dx2,cent[0]+dy2])*osamp
    # im1 = im.copy() #First image oriented to roll 1
    im1 = pyklip_rotate(im, roll_angle_1, jnp.array(cent[::-1])*osamp,new_center = new_cen1) #Second image 5 degree roll
    im2 = pyklip_rotate(im, roll_angle_2, jnp.array(cent[::-1])*osamp,new_center = new_cen2) #Second image 5 degree roll

    psfs,psf_inds,im_mask,unique_inds = nircam_psf_list
    osamp_model1 = convolve_with_spatial_psfs(im1,psfs,psf_inds,im_mask,unique_inds)
    osamp_model2 = convolve_with_spatial_psfs(im2,psfs,psf_inds,im_mask,unique_inds)
    #Rebin
    model1 = jnp.sum(osamp_model1.reshape(output_shape[0],osamp,output_shape[1],osamp),axis=(1,3))
    model2 = jnp.sum(osamp_model2.reshape(output_shape[0],osamp,output_shape[1],osamp),axis=(1,3))
    return model1,model2

def gen_roll_images_hg3(x,cent,nircam_psf_list,f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                          a0_1=40,a0_2=220,a0_3=401,a0_4=300,
                          nx=140,ny=140,n_nodes=6):
    '''
    Generate a disk image and rotate it and convolve it with the NIRCam PSF
    '''

    mod = quad_disk_model_hg3(x,cent,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                          a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                          nx=nx,ny=ny)
    mod_osamped = fake_osamp(mod)
    mod1,mod2 = rotate_and_convolve_osamped(mod_osamped,cent,[nx,ny],nircam_psf_list,
                                             dx1=x[18+4*n_nodes],dy1=x[19+4*n_nodes],
                                            dx2=x[20+4*n_nodes],dy2=x[21+4*n_nodes])
    return mod1,mod2

def gen_roll_images_hg2(x,cent,nircam_psf_list,f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                          a0_1=40,a0_2=220,a0_3=401,a0_4=300,
                          nx=140,ny=140,n_nodes=6):
    '''
    Generate a disk image and rotate it and convolve it with the NIRCam PSF
    '''

    mod = quad_disk_model_hg2(x,cent,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                          a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                          nx=nx,ny=ny)
    mod_osamped = fake_osamp(mod)
    mod1,mod2 = rotate_and_convolve_osamped(mod_osamped,cent,[nx,ny],nircam_psf_list,
                                             dx1=x[18+4*n_nodes],dy1=x[19+4*n_nodes],
                                            dx2=x[20+4*n_nodes],dy2=x[21+4*n_nodes])
    return mod1,mod2

def gen_psf_model(x,ref_data):
    '''
    Generate a PSF model
    '''

    n_refs = ref_data.shape[0]
    ref_scales_data1 = x[-2*n_refs:-n_refs] #Assume the PSF weights are the last 2*n_refs parameters
    ref_scales_data2 = x[-n_refs:]
    ref_psf_data1 = jnp.matmul(ref_data.T,ref_scales_data1).T/n_refs #PSF for Data 1
    ref_psf_data2 = jnp.matmul(ref_data.T,ref_scales_data2).T/n_refs #PSF for Data 2

    return ref_psf_data1, ref_psf_data2

def gen_roll_images_hg3_w_psf(x,ref_data,cent,nircam_psf_list,f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                          a0_1=40,a0_2=220,a0_3=401,a0_4=300,
                          nx=140,ny=140,n_nodes=6):
    '''
    Generate a disk image and rotate it and convolve it with the NIRCam PSF
    then add in the stellar PSF model
    '''
    models = gen_roll_images_hg3(x,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=nx,ny=ny,n_nodes=n_nodes)
    
    
    ref_psf_data1, ref_psf_data2 = gen_psf_model(x,ref_data)
    return models[1]+ref_psf_data1, models[0]+ref_psf_data2

def gen_roll_images_hg2_w_psf(x,ref_data,cent,nircam_psf_list,f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                          a0_1=40,a0_2=220,a0_3=401,a0_4=300,
                          nx=140,ny=140,n_nodes=6):
    '''
    Generate a disk image and rotate it and convolve it with the NIRCam PSF
    then add in the stellar PSF model
    '''
    models = gen_roll_images_hg2(x,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=nx,ny=ny,n_nodes=n_nodes)
    
    
    ref_psf_data1, ref_psf_data2 = gen_psf_model(x,ref_data)
    return models[1]+ref_psf_data1, models[0]+ref_psf_data2

def chi2_hg3(x,sci1_data,sci1_data_err,sci2_data,sci2_data_err,ref_data,cent,
         data_shape,nircam_psf_list,n_nodes=6,
         f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
            a0_1=40, a0_2=220, a0_3=410, a0_4=300):

    #Generate the model images, roll them, convolve them
    model_w_psf1, model_w_psf2 = gen_roll_images_hg3_w_psf(x,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=data_shape[1],ny=data_shape[0],n_nodes=6)
    ### Note this swamp: I seem to have tha PAs off for the two disks. 
    residuals1 = (sci2_data-model_w_psf2).flatten()
    residuals2 = (sci1_data-model_w_psf1).flatten()
    out = jnp.nansum(jnp.power(residuals1/sci1_data_err.flatten(),2)+jnp.power(residuals2/sci2_data_err.flatten(),2))
    return out

def chi2_hg3_just_flux(x,x2,sci1_data,sci1_data_err,sci2_data,sci2_data_err,ref_data,cent,
         data_shape,nircam_psf_list,n_nodes=6,
         f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
            a0_1=40, a0_2=220, a0_3=410, a0_4=300):
    '''
    Caclulcate the chi2 using the morphology from a previous fit (x2), letting the fluxes, offsets and PSF weights vary
    '''
    n_refs = ref_data.shape[0]
    #We're going to sub in the new fluxes into x2: 
    #First all the disk fluxes
    x2 = x2.at[2].set(x[0])
    x2 = x2.at[n_nodes+6].set(x[1])
    x2 = x2.at[2*n_nodes+10].set(x[2])
    x2 = x2.at[3*n_nodes+14].set(x[3])
    #Now the centering
    x2 = x2.at[4*n_nodes+18:4*n_nodes+22].set(x[4:8])
    #Now the PSF weighting terms
    x2 = x2.at[-2*n_refs:].set(x[-2*n_refs:])

    #Generate the model images, roll them, convolve them
    model_w_psf1, model_w_psf2 = gen_roll_images_hg3_w_psf(x2,ref_data,cent,nircam_psf_list,
                                                           f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=data_shape[1],ny=data_shape[0],n_nodes=n_nodes)
    ### Note this swamp: I seem to have tha PAs off for the two disks. 
    residuals2 = (sci2_data-model_w_psf2).flatten()
    residuals1 = (sci1_data-model_w_psf1).flatten()

    out = jnp.nansum(jnp.power(residuals1/sci1_data_err.flatten(),2)+jnp.power(residuals2/sci2_data_err.flatten(),2))
    return out

def chi2_hg3_flux_and_spf(x,x2,sci1_data,sci1_data_err,sci2_data,sci2_data_err,ref_data,cent,
         data_shape,nircam_psf_list,n_nodes=6,
         f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
            a0_1=40, a0_2=220, a0_3=410, a0_4=300):
    '''
    Caclulcate the chi2 using the morphology from a previous fit (x2), letting the fluxes, SPFs offsets and PSF weights vary
    '''
    n_refs = ref_data.shape[0]
    #We're going to sub in the new fluxes into x2: 
    #First all the disk fluxes
    x2 = x2.at[2].set(x[0])
    x2 = x2.at[n_nodes+6].set(x[1])
    x2 = x2.at[2*n_nodes+10].set(x[2])
    x2 = x2.at[3*n_nodes+14].set(x[3])
    #Now the centering
    x2 = x2.at[4*n_nodes+18:4*n_nodes+22].set(x[4:8])
    #Now the spfs
    x2 = x2.at[6:6+n_nodes].set(x[8:8+n_nodes])
    x2 = x2.at[10+n_nodes:10+2*n_nodes].set(x[8+n_nodes:8+2*n_nodes])
    x2 = x2.at[14+2*n_nodes:14+3*n_nodes].set(x[8+2*n_nodes:8+3*n_nodes])
    x2 = x2.at[18+3*n_nodes:18+4*n_nodes].set(x[8+3*n_nodes:8+4*n_nodes])
    #Now the PSF weighting terms
    x2 = x2.at[-2*n_refs:].set(x[-2*n_refs:])

    #Generate the model images, roll them, convolve them
    model_w_psf1, model_w_psf2 = gen_roll_images_hg3_w_psf(x2,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=data_shape[1],ny=data_shape[0],n_nodes=n_nodes)
    ### Note this swamp: I seem to have tha PAs off for the two disks. 
    residuals1 = (sci2_data-model_w_psf2).flatten()
    residuals2 = (sci1_data-model_w_psf1).flatten()
    out = jnp.nansum(jnp.power(residuals1/sci1_data_err.flatten(),2)+jnp.power(residuals2/sci2_data_err.flatten(),2))
    return out

def chi2_hg2(x,sci1_data,sci1_data_err,sci2_data,sci2_data_err,ref_data,cent,
         data_shape,nircam_psf_list,n_nodes=6,
         f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
            a0_1=40, a0_2=220, a0_3=410, a0_4=300):

    model_w_psf1, model_w_psf2 = gen_roll_images_hg2_w_psf(x,ref_data,cent,nircam_psf_list,f0_1=f0_1,f0_2=f0_2,f0_3=f0_3,f0_4=f0_4,
                            a0_1=a0_1,a0_2=a0_2,a0_3=a0_3,a0_4=a0_4,
                            nx=data_shape[1],ny=data_shape[0],n_nodes=6)
    ### Note this swamp: I seem to have tha PAs off for the two disks. 
    residuals1 = (sci2_data-model_w_psf2).flatten()
    residuals2 = (sci1_data-model_w_psf1).flatten()
    out = jnp.nansum(jnp.power(residuals1/sci1_data_err.flatten(),2)+jnp.power(residuals2/sci2_data_err.flatten(),2))
    return out

def print_params(params,n_nodes=6,
                 f0_1=120000,f0_2=25000,f0_3=5000,f0_4=12000,
                    a0_1=40, a0_2=220, a0_3=410, a0_4=300,):
    '''
    Print out the best fit parameters
    '''

    print("####################")
    print("Global Parameters")
    print("####################")
    print("Inclination: {:.1f} degrees".format(params[0]))
    print("Position Angle: {:.1f} degrees".format((params[1]+180+280)%360))
    print("Roll 1 center: [{:.2f},{:.2f}]".format(149.1+params[20+4*n_nodes],173.4+params[21+4*n_nodes]))
    print("Roll 2 center: [{:.2f},{:.2f}]".format(149.1+params[22+4*n_nodes],173.4+params[23+4*n_nodes]))

    print("########## Disk 1 ############")
    print("Flux Scaling Factor: {:.0f}".format(params[2]*f0_1))
    print("Semi-major axis: {:.0f}".format(params[3]*a0_1))
    print("a_in: {:.1f}".format(params[4]))
    print("a_out: {:.1f}".format(params[5]))

    print("\n########## Disk 2 ############")
    print("Flux Scaling Factor: {:.0f}".format(params[n_nodes+6]*f0_2))
    print("Semi-major axis: {:.0f}".format(params[n_nodes+7]*a0_2))
    print("a_in: {:.1f}".format(params[8+n_nodes]))
    print("a_out: {:.1f}".format(params[9+n_nodes]))

    print("\n########## Disk 3 ############")
    print("Flux Scaling Factor: {:.0f}".format(params[2*n_nodes+10]*f0_3))
    print("Semi-major axis: {:.0f}".format(params[2*n_nodes+11]*a0_3))
    print("a_in: {:.1f}".format(params[12+2*n_nodes]))
    print("a_out: {:.1f}".format(params[13+2*n_nodes]))

    print("\n########## Disk 4 ############")
    print("Flux Scaling Factor: {:.0f}".format(params[3*n_nodes+14]*f0_4))
    print("Semi-major axis: {:.0f}".format(params[3*n_nodes+15]*a0_4))
    print("a_in: {:.1f}".format(params[16+3*n_nodes]))
    print("a_out: {:.1f}".format(params[17+3*n_nodes]))
