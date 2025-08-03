# jwst_ers_NIRCAM_HD141569A

This is a repository that holds the analysis scripts for Millar-Blanchaer et al. (in prep), analyzing the JWST ERS NIRCam data on HD 141569A. The code in this respository has been organized so that you can run it in the below order to replicate the analysis in the the paper. Before starting, you must download the L1 "*uncal.fits" files from MAST. They can be on MAST at [this DOI](http://dx.doi.org/10.17909/ty1h-9x40). 

1. Stage1Stage2 (Section 3.0): Carry out the basic JWST pipeline calibrations, as run by spaceKLIP.
2. Fit_NIRCam_Parameters (Section 3.1): Fit for the NIRCam instrument parameters defocus, pupil shear and pupil rotation.
3. Sub_Mstars (Section 3.2): Fit a model to and subtract the two bright M-star companions.
4. Disk_Modelling (Section 5.1): Perform MCRDI to find the best-fit disk and PSF-subtraction weights. 
5. Subtract_PSF (Section 5.1): Subtract the best-fit PSF found in the MCRDI process. 
6. Deconvolution (Section 4): Deconvolve the disk image (after PSF-subtraction).
7. Flux_Ratio (Section 5.2): Get the flux ratio between the two filters. 
8. Planet_Sensitivity (Section 7): Calculate the contrast curves and get hte planet sensitivity.

Authors: Max Millar-Blanchaer, Kellen Lawson
