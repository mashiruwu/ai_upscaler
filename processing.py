import cv2
import numpy as np

def downscale_image(img_np, factor=0.5):
    if factor >= 1.0:
        return img_np
    h, w = img_np.shape[:2]
    new_h = int(h * factor)
    new_w = int(w * factor)
    return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

def match_texture(original_img, upscaled_img):
    """
    Matches the luminance distribution (texture/grain) of the original image 
    to the upscaled image using LAB color space histogram matching.
    """
    # Resize original to match upscaled
    h, w = upscaled_img.shape[:2]
    orig_resized = cv2.resize(original_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Convert both to LAB
    orig_lab = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2LAB)
    up_lab = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2LAB)
    
    # Extract L (Luminance) channels
    orig_l = orig_lab[:,:,0]
    up_l = up_lab[:,:,0]
    
    # Histogram matching for L channel
    matched_l = histogram_match(up_l, orig_l)
    
    # Blend the matched L channel slightly to avoid over-matching
    blended_l = cv2.addWeighted(matched_l, 0.4, up_l, 0.6, 0)
    
    # Put back into LAB
    up_lab[:,:,0] = blended_l
    
    # Convert back to RGB
    matched_img = cv2.cvtColor(up_lab, cv2.COLOR_LAB2RGB)
    return matched_img

def histogram_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    return interp_t_values[bin_idx].reshape(oldshape).astype(source.dtype)
