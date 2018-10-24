from skimage.measure import label, regionprops


def od_statistics(img, filename, thresh=0.5):

    stats_dict = dict()
    binary_image = img > thresh
    labeled = label(binary_image)
    props = regionprops(labeled, intensity_image=img)
    stats_dict['no_objects'] = len(props)
    props = sorted(props, key=lambda p: p.area, reverse=True)

    if len(props) > 0:
        row, col = props[0].centroid
        stats_dict['area'] = props[0].area
        stats_dict['x'] = row
        stats_dict['y'] = col
        stats_dict['eccentricity'] = props[0].eccentricity
        stats_dict['mean_intensity'] = props[0].mean_intensity
        stats_dict['orientation'] = props[0].orientation

    stats_dict['filename'] = filename
    return stats_dict