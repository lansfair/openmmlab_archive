from osgeo import gdal
import numpy as np

s2_band_stats = dict(
    # mean=[
    #     1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2,
    #     2599.7, 732.1, 12.1, 1820.6, 1118.2
    # ],
    # std=[
    #     897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5,
    #     1750.1, 475.1, 98.3, 1216.5, 736.7
    # ],
    mean=[
        0,0,0,0
    ],
    std=[
        1,1,1,1
    ],
)

SAR_path = "/mnt/ht2-nas2/EO_test/dataset/DFC2025 BRIGHT/dfc25_track2_trainval/train/post-event/bata-explosion_00000000_post_disaster.tif"
OPt_path = "/mnt/ht2-nas2/EO_test/dataset/DFC2025 BRIGHT/dfc25_track2_trainval/train/pre-event/bata-explosion_00000000_pre_disaster.tif"

ds1 = gdal.Open(SAR_path).ReadAsArray()

bands = []
ds1 = ds1[None,...]
bands.append(ds1)


ds2 = gdal.Open(OPt_path).ReadAsArray()
mean = s2_band_stats["mean"]
std = s2_band_stats["std"]
mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

ds = np.concatenate((ds2, ds1),axis=0).transpose(1,2,0)
ds = (ds - mean) / std
pass
