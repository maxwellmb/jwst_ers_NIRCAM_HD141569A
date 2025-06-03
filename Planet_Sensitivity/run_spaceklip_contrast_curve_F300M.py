from __future__ import division

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from spaceKLIP import database, imagetools, analysistools, coron1pipeline,\
    coron2pipeline, coron3pipeline, pyklippipeline, classpsfsubpipeline

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Set input and output directories.
    # idir = 'uncal/'
    # idir = 'spaceklip/stage2/'
    # idir = '/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/'
    idir = '/home/maxmb/Library/jwst_hd141569a_lib/data/F300M/jk/spaceklip/klipsub/'
    # idir = 'spaceklip/klipsub/'
    # idir = 'spaceklip/injected/KL100/C1/KLIP_FM/'
    odir = idir+'contrast_curves/'

    # Get FITS files.
    # fitsfiles = sorted([idir + f for f in os.listdir(idir)
    #                     if f.endswith('_uncal.fits')])
    # fitsfiles = sorted([idir + f for f in os.listdir(idir)
    #                     if f.endswith('_calints.fits')])
    fitsfiles = sorted([idir + f for f in os.listdir(idir)
                        if f.endswith('-KLmodes-all.fits')
                        # and f.startswith('INJECTED')])
                        and 'ADI+RDI' in f])

    # Initialize spaceKLIP database.
    Database = database.Database(output_dir=odir)

    # Read FITS files.
    # Database.read_jwst_s012_data(datapaths=fitsfiles,
    #                              bgpaths=None)
    Database.read_jwst_s3_data(datapaths=fitsfiles)

    # Run Coron1Pipeline, Coron2Pipeline, and Coron3Pipeline. Additional step
    # parameters can be passed using the steps keyword as outlined here:
    # https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
    # steps = {'saturation': {'n_pix_grow_sat': 1,
    #                         'grow_diagonal': False},
    #          'refpix': {'odd_even_columns': True,
    #                     'odd_even_rows': True,
    #                     'nlower': 4,
    #                     'nupper': 4,
    #                     'nleft': 4,
    #                     'nright': 4,
    #                     'nrow_off': 0,
    #                     'ncol_off': 0},
    #          'dark_current': {'skip': True},
    #          'jump': {'rejection_threshold': 4.,
    #                   'three_group_rejection_threshold': 4.,
    #                   'four_group_rejection_threshold': 4.},
    #          'ramp_fit': {'save_calibrated_ramp': False}}
    # coron1pipeline.run_obs(database=Database,
    #                        steps=steps,
    #                        subdir='stage1')
    # steps = {'outlier_detection': {'skip': False}}
    # coron2pipeline.run_obs(database=Database,
    #                        steps=steps,
    #                        subdir='stage2')
    # steps = {'klip': {'truncate': 100}}
    # coron3pipeline.run_obs(database=Database,
    #                        steps=steps,
    #                        subdir='stage3')

    # Initialize spaceKLIP image manipulation tools.
    # ImageTools = imagetools.ImageTools(database=Database)

    # Remove first frame due to reset switch charge delay. Only required for
    # MIRI.
    # ImageTools.remove_frames(index=[0],
    #                          types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                          subdir='removed')

    # Median-subtract each frame to mitigate uncalibrated bias drifts. Only
    # required for NIRCam.
    # ImageTools.subtract_median(types=['SCI', 'SCI_TA', 'SCI_BG',
    #                                   'REF', 'REF_TA', 'REF_BG'],
    #                            subdir='medsub')

    # Crop all frames.
    # ImageTools.crop_frames(npix=1,
    #                        types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                        subdir='cropped')

    # Identify and fix bad pixels using custom spaceKLIP routines. Multiple
    # routines can be combined in a custom order by joining them with + signs.
    # - bpclean: use sigma clipping to identify additional bad pixels.
    # - custom:  use a custom bad pixel map.
    # - timemed: replace pixels which are only bad in some frames with their
    #            median value from the good frames.
    # - dqmed:   replace bad pixels with the median of the surrounding good
    #            pixels.
    # - medfilt: replace bad pixels with an image plane median filter.
    # ImageTools.fix_bad_pixels(method='bpclean+timemed+dqmed+medfilt',
    #                           bpclean_kwargs={'sigclip': 5.,
    #                                           'shift_x': [-1, 0, 1],
    #                                           'shift_y': [-1, 0, 1]},
    #                           custom_kwargs={},
    #                           timemed_kwargs={},
    #                           dqmed_kwargs={'shift_x': [-1, 0, 1],
    #                                         'shift_y': [-1, 0, 1]},
    #                           medfilt_kwargs={'size': 4},
    #                           subdir='bpcleaned')

    # Perform background subtraction to remove MIRI glowstick. Only required
    # for MIRI.
    # ImageTools.subtract_background(nsplit=1,
    #                                subdir='bgsub')

    # Replace all nans.
    # ImageTools.replace_nans(cval=0.,
    #                         types=['SCI', 'REF'],
    #                         subdir='nanreplaced')

    # Blur frames.
    # ImageTools.blur_frames(fact='auto',
    #                        types=['SCI', 'REF'],
    #                        subdir='blurred')

    # Recenter frames. Before, update the NIRCam coronagraphic mask centers to
    # the on-sky values measured by Jarron. Might not be required for simulated
    # data!
    # ImageTools.update_nircam_centers()
    # ImageTools.recenter_frames(method='fourier',
    #                            subpix_first_sci_only=False,
    #                            spectral_type='A2V',
    #                            kwargs={},
    #                            subdir='recentered')

    # Use image registration to align all frames in the concatenation to the
    # first science frame in that concatenation.
    # ImageTools.align_frames(method='fourier',
    #                         kwargs={},
    #                         subdir='aligned')

    # Coadd frames.
    # ImageTools.coadd_frames(nframes=None,
    #                         types=['SCI', 'REF'],
    #                         subdir='coadded')

    # Apply high-pass filter.
    # size = {}
    # size['JWST_NIRCAM_NRCALONG_F300M_MASKRND_MASKA335R_SUB320A335R'] = 3.
    # size['JWST_NIRCAM_NRCALONG_F360M_MASKRND_MASKA335R_SUB320A335R'] = 3.
    # ImageTools.hpf(size=size,
    #                types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                subdir='filtered')

    # Pad all frames.
    # ImageTools.pad_frames(npix=160,
    #                       cval=np.nan,
    #                       types=['SCI', 'REF'],
    #                       subdir='padded')

    # Run pyKLIP pipeline. Additional parameters for the klip_dataset function
    # can be passed using the kwargs keyword.
    # kwargs = {'mode': ['ADI+RDI'],
    #           'annuli': [1],
    #           'subsections': [1],
    #           'numbasis': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
    #           'highpass': False,
    #           'algo': 'klip',
    #           'save_rolls': True}
    # pyklippipeline.run_obs(database=Database,
    #                        kwargs=kwargs,
    #                        subdir='klipsub')

    # Run classical PSF subtraction pipeline.
    # kwargs = {'combine_dithers': True,
    #           'save_rolls': True,
    #           'mask_bright': None}
    # classpsfsubpipeline.run_obs(database=Database,
    #                             kwargs=kwargs,
    #                             subdir='psfsub')

    # Initialize spaceKLIP analysis tools.
    AnalysisTools = analysistools.AnalysisTools(Database)

    # Extract companions.
    # AnalysisTools.extract_companions(companions=[[0.286, 0.465, 1e-4]],
    #                                  starfile='hd_141569.vot',
    #                                  mstar_err=0.,
    #                                  spectral_type='A2V',
    #                                  klmode='max',
    #                                  date='auto',
    #                                  use_fm_psf=True,
    #                                  highpass=False,
    #                                  fitmethod='mcmc',
    #                                  fitkernel='diag',
    #                                  subtract=True,
    #                                  inject=False,
    #                                  overwrite=True,
    #                                  subdir='recovered')

    # Compute raw contrast.
    AnalysisTools.raw_contrast(starfile='hd_141569.vot',
                               spectral_type='A2V',
                               companions=[[-6.741, 5.733, 16.],
                                           [-5.607, 5.040, 16.]],
                               subdir='rawcon')

    # pdb.set_trace()
