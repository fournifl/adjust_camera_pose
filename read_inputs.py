import cv2
import pandas as pd
import numpy as np
from pyproj import Transformer
import georaster
import tifffile
import georef
from geo import world_2_pix, pix_2_world, compute_xyz_from_pix_and_uv_from_geo

def read_ortho(f_ortho):

    # extent of tif file
    band1 = georaster.SingleBandRaster(f_ortho, load_data=False)
    extent = band1.extent # xmin, xmax, ymin, ymax

    # get data
    data = tifffile.imread(f_ortho)

    return extent, data

def read_img(f_img, georef_params):
    img = cv2.imread(f_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # undistort image
    img = georef_params.undistort(img)
    return img

def read_tif_mnt(f_mnt, epsg_mnt, georef_params, ss_ech_factor=2):

    # extent of tif file
    band1 = georaster.SingleBandRaster(f_mnt, load_data=False)
    xmin, xmax, ymin, ymax = band1.extent

    # get mnt data
    data = tifffile.imread(f_mnt)

    # create masked array
    data = np.ma.array(data, mask=data < -32760)

    # get x, y grid coordinates
    x = np.linspace(xmin, xmax, data.shape[1])
    y = np.linspace(ymax, ymin, data.shape[0])
    x, y = np.meshgrid(x, y)
    x = x[~data.mask].flatten()
    y = y[~data.mask].flatten()
    transformer = Transformer.from_crs(epsg_mnt, georef_params.georef_local_srs.horizontal_srs.srid, always_xy=True)
    x, y = transformer.transform(x, y)

    # compress data
    data = data.compressed().flatten()

    # mnt uv from geo
    u, v = world_2_pix(np.vstack([x, y, data]), georef_params)

    return data[::ss_ech_factor], x[::ss_ech_factor], y[::ss_ech_factor], u[::ss_ech_factor], v[::ss_ech_factor]

def read_gcps(f_gcps, georef_params):
    df_gcps = pd.read_csv(f_gcps,
                                              usecols=['easting', 'northing', 'elevation', 'U', 'V'])
    # create U_undist and V_undist
    undist_pts = cv2.undistortPoints(np.array(df_gcps[['U', 'V']]).astype(float),
                                     georef_params.intrinsic_parameters.camera_matrix,
                                     georef_params.distortion_coefficients.array,
                                     P=georef_params.intrinsic_parameters.camera_matrix)
    undist_pts = undist_pts.reshape((undist_pts.shape[0], undist_pts.shape[2]))
    df_gcps['U_undist'] = undist_pts[:, 0]
    df_gcps['V_undist'] = undist_pts[:, 1]

    # remarkable pts xyz from pix, and uv from geo
    xyz_gcps_from_pix, u_gcps_from_geo, v_gcps_from_geo = (
        compute_xyz_from_pix_and_uv_from_geo(df_gcps[['U', 'V', 'elevation']],
                                             np.array(
                                                 df_gcps[['easting', 'northing', 'elevation']]),
                                             georef_params))
    return df_gcps, xyz_gcps_from_pix, u_gcps_from_geo, v_gcps_from_geo

def read_litto3d_pts(f_litto3d, georef_params, ss_ech_factor=1):
    # read csv (x, y, z)
    df_litto3d_pts = pd.read_csv(f_litto3d)
    # convert litto3d semi points in u, v coordinates
    u_litto, v_litto = world_2_pix(df_litto3d_pts[['x', 'y', 'z']], georef_params)
    return df_litto3d_pts[::ss_ech_factor], u_litto[::ss_ech_factor], v_litto[::ss_ech_factor]

