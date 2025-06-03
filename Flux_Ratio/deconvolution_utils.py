import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import convolve
from jax.scipy import ndimage

@jit
def psf_convolve_jax(im,psf_im):
    '''
    jax-based PSF convolution 
    '''
    imcon = convolve(im.astype(jnp.float64), psf_im.astype(jnp.float64), mode='same')
    return imcon

@jit
def convolve_with_spatial_psfs(im0, psfs, psf_inds, im_mask,unique_inds):
    """
    (Based off a functio by Kellen Lawson) 

    Creates a PSF-convolved image where each pixel of the input image has been convolved with the
    nearest spatially-sampled PSF.

    Note: This can be sped up a little by preparing a boolean array where each slice is the shape
          of im0 and is True where the corresponding slice in psfs is the nearest match. However, 
          if `psfs' is very finely sampled, this would result in a very large array (e.g., if psfs
          samples at every pixel in im0, this would produce an array of shape (ny*nx, ny, nx)). 
          In testing, the time saved was marginal enough (~5%) that I've avoided this approach in
          favor of the more memory conscious inputs here.
    ___________
    Parameters:

        im0: ndarray
            2D image array to be convolved.

        psfs: ndarray
            3D image array of spatially-sampled PSFs with which to convolve im0. Generally, each

        psf_inds: ndarray
            2D array (same shape as im0; dtype=int); each entry gives the index of the slice in psfs with 
            which that pixel in im0 should be convolved.

    Optional:

        im_mask: ndarray
            2D array of the coronagraph throughput (same shape as im0), by which im0 will be multiplied before
            convolution.

    Returns:
        imcon: ndarray
            Convolved image of the same shape as im0.
    """
    im = im0.copy()
    im *= im_mask

    # Define the function for a single slice of psfs
    def convolve_slice(slice_num):
        msk_i = psf_inds == slice_num
        im_to_convolve = jnp.where(msk_i[..., None], im[..., None], 0.)
        return psf_convolve_jax(im_to_convolve[:,:,0], psfs[slice_num])

    # Vectorize the slice function

    # Iterate over unique slice indices in psf_inds and apply the convolution to the corresponding image pixels
    # unique_inds = jnp.unique(psf_inds)
    imcon_list = []
    for i in unique_inds:
        imcon_slice = convolve_slice(i)
        imcon_list.append(imcon_slice)

    # Stack the convolved slices and construct the output image
    imcon_crop = jnp.stack(imcon_list)
    # print(np.size(imcon_crop))
    # imcon = jnp.zeros_like(im)
    # imcon = jax.ops.index_add(imcon, jax.ops.index[y1:y2+1, x1:x2+1], imcon_crop)
    return jnp.sum(imcon_crop,axis=0)

@jit
def l1_l2_norm_regularizer(image, t=1e-2):

    # Compute spatial gradient terms
    diff_x = jnp.abs(image[1:, :-1] - image[1:, 1:])
    diff_y = jnp.abs(image[:-1, 1:] - image[1:, 1:])
    
    # Compute L1 norm of spatial gradient
    # l1_norm = jnp.sum(jnp.abs(diff_x)) + jnp.sum(jnp.abs(diff_y))
    # Compute L2 norm of spatial gradient
    # l2_norm = jnp.sqrt(jnp.square(diff_x) + jnp.square(diff_y))
    
    # Compute L2 norm of spatial gradient with the t parameter
    l2_norm = jnp.sqrt(jnp.square(diff_x) + jnp.square(diff_y) + jnp.square(t))
    
    #Following the Edge-Preserving smoothness in Denneulin
    return jnp.sum(l2_norm)

@jit
def pyklip_rotate(img, angle, center, new_center=None, flipx=False, astr_hdr=None):
    """
    Note: Function originally from pyklip, but stripped down a bit 

    Rotate an image by the given angle about the given center.
    Optional: can shift the image to a new image center after rotation. Also can reverse x axis for those left
              handed astronomy coordinate systems

    Args:
        img: a 2D image
        angle: angle CCW to rotate by (degrees)
        center: 2 element list [x,y] that defines the center to rotate the image to respect to
        new_center: 2 element list [x,y] that defines the new image center after rotation
        flipx: reverses x axis after rotation
        astr_hdr: wcs astrometry header for the image
    Returns:
        resampled_img: new 2D image
    """
    #convert angle to radians
    angle_rad = jnp.radians(angle)

    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))

    #if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - center[0]
        dy = new_center[1] - center[1]
        x -= dx
        y -= dy

    #flip x if needed to get East left of North
    if flipx is True:
        x = center[0] - (x - center[0])

    #do rotation. CW rotation formula to get a CCW of the image
    xp = (x-center[0])*jnp.cos(angle_rad) + (y-center[1])*jnp.sin(angle_rad) + center[0]
    yp = -(x-center[0])*jnp.sin(angle_rad) + (y-center[1])*jnp.cos(angle_rad) + center[1]

    # resampled_img = jax_pyklip_nan_map_coordinates_2d(img, yp, xp)
    resampled_img = ndimage.map_coordinates(jnp.copy(img), jnp.array([yp, xp]),order=1,cval = 0.)

    return resampled_img

@jit
def jax_pyklip_nan_map_coordinates_2d(img, yp, xp, mc_kwargs=None):
    """
    This function was originally from pyklip

    JAX-compatible version of scipy.ndimage.map_coordinates() that handles nans for 2-D transformations. Only works in 2-D!

    Do NaN detection by defining any pixel in the new coordiante system (xp, yp) as a nan
    If any one of the neighboring pixels in the original image is a nan (e.g. (xp, yp) = 
    (120.1, 200.1) is nan if either (120, 200), (121, 200), (120, 201), (121, 201) is a nan)

    Args:
        img (jax.numpy.array): 2-D image that is looking to be transformed
        yp (jax.numpy.array): 2-D array of y-coordinates that the image is evaluated out
        xp (jax.numpy.array): 2-D array of x-coordinates that the image is evaluated out
        mc_kwargs (dict): other parameters to pass into the map_coordinates function.

    Returns:
        transformed_img (jax.numpy.array): 2-D transformed image. Each pixel is evaluated at the (yp, xp) specified by xp and yp. 
    """
    # check if optional parameters are passed in
    if mc_kwargs is None:
        mc_kwargs = {}
    # if nothing specified, we will pad transformations with jnp.nan
    if "cval" not in mc_kwargs:
        mc_kwargs["cval"] = jnp.nan

    # check all four pixels around each pixel and see whether they are nans
    xp_floor = jnp.clip(jnp.floor(xp).astype(int), 0, img.shape[1]-1)
    xp_ceil = jnp.clip(jnp.ceil(xp).astype(int), 0, img.shape[1]-1)
    yp_floor = jnp.clip(jnp.floor(yp).astype(int), 0, img.shape[0]-1)
    yp_ceil = jnp.clip(jnp.ceil(yp).astype(int), 0, img.shape[0]-1)
    rotnans = jnp.where(jnp.isnan(img[yp_floor.ravel(), xp_floor.ravel()]) | 
                       jnp.isnan(img[yp_floor.ravel(), xp_ceil.ravel()]) |
                       jnp.isnan(img[yp_ceil.ravel(), xp_floor.ravel()]) |
                       jnp.isnan(img[yp_ceil.ravel(), xp_ceil.ravel()]))

    # resample image based on new coordinates, set nan values as median
    nanpix = jnp.where(jnp.isnan(img))
    medval = jnp.nanmedian(img)
    img_copy = jnp.copy(img)
    img_copy = jnp.where(jnp.isnan(img_copy), medval, img_copy)
    transformed_img = ndimage.map_coordinates(img_copy, jnp.array([yp, xp]), **mc_kwargs)

    # mask nans
    img_shape = transformed_img.shape
    transformed_img = jnp.reshape(transformed_img, [img_shape[0] * img_shape[1]])
    transformed_img = jnp.where(rotnans, jnp.nan, transformed_img)
    transformed_img = jnp.reshape(transformed_img, img_shape)

    return transformed_img
