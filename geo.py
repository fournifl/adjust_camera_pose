import numpy as np

def pix_2_world(uvz, georef_params):
    # convert pix points to geo local
    geo_local = georef_params.pix2geo(uvz)
    geo_world = georef_params.local_srs.m_w_l @ geo_local
    return geo_world

def world_2_pix(xyz, georef_params):
    geo_local = georef_params.local_srs.m_l_w @ xyz
    u, v = georef_params.geo2pix(geo_local)
    return u ,v

def compute_xyz_from_pix_and_uv_from_geo(uvz, xyz, georef_params):
    xyz_from_pix = pix_2_world(uvz, georef_params)
    u, v = world_2_pix(xyz, georef_params)
    return xyz_from_pix, u, v

def reprojection_error(xy1, xy2):
    distances = np.linalg.norm(xy1 - xy2, axis=1)
    return distances