import SimpleITK as sitk
import numpy as np

def overlay_mask(img, mask, out):

    img_gray = sitk.GetImageFromArray(np.mean(img, axis=2).astype(np.uint8))
    overlay = sitk.LabelOverlay(img_gray, sitk.GetImageFromArray(mask))
    sitk.WriteImage(overlay, out)
    return sitk.GetArrayFromImage(overlay)
