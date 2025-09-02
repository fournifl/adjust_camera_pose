import json
from nicegui import ui
from georef.in_out import CameraParameters, read_json_file
import numpy as np
import cv2
from copy import copy
import pandas as pd
from pyproj import Transformer
from georef.operators import Georef, IntrinsicMatrix
import georaster
import tifffile

def read_ortho(f_ortho):

    # extent of tif file
    band1 = georaster.SingleBandRaster(f_ortho, load_data=False)
    extent = band1.extent # xmin, xmax, ymin, ymax

    # get data
    data = tifffile.imread(f_ortho)

    return extent, data

def read_img(f_img):
    img = cv2.imread(f_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # undistort image
    img = georef_params.undistort(img)
    return img

def read_tif_mnt(f_mnt, epsg_mnt, georef_params, ss_ech_factor=100):

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

def read_remarkable_pts(f_corresp_pts_remarquables):
    df_corresp_pts_remarquables = pd.read_csv(f_corresp_pts_remarquables,
                                              usecols=['easting', 'northing', 'elevation', 'U', 'V'])
    # create U_undist and V_undist
    undist_pts = cv2.undistortPoints(np.array(df_corresp_pts_remarquables[['U', 'V']]).astype(float),
                                     georef_params.intrinsic_parameters.camera_matrix,
                                     georef_params.distortion_coefficients.array,
                                     P=georef_params.intrinsic_parameters.camera_matrix)
    undist_pts = undist_pts.reshape((undist_pts.shape[0], undist_pts.shape[2]))
    df_corresp_pts_remarquables['U_undist'] = undist_pts[:, 0]
    df_corresp_pts_remarquables['V_undist'] = undist_pts[:, 1]

    # remarkable pts xyz from pix, and uv from geo
    xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo = (
        compute_xyz_from_pix_and_uv_from_geo(df_corresp_pts_remarquables[['U', 'V', 'elevation']],
                                             np.array(
                                                 df_corresp_pts_remarquables[['easting', 'northing', 'elevation']]),
                                             georef_params))
    return df_corresp_pts_remarquables, xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo

def get_adjustable_elements(georef_params):
    # camera angles
    cam_angles = georef_params.extrinsic.beachcam_angles
    cam_angles_init_tmp = copy(cam_angles)
    cam_angles_init = {}
    cam_angles_init[0] = cam_angles_init_tmp[0]
    cam_angles_init[1] = cam_angles_init_tmp[1]
    cam_angles_init[2] = cam_angles_init_tmp[2]

    # camera origin
    origin = georef_params.extrinsic.origin
    origin_init = {}
    origin_init[0] = origin[0]
    origin_init[1] = origin[1]
    origin_init[2] = origin[2]

    # fx, fy
    focal  = [georef_params.intrinsic.fx, georef_params.intrinsic.fy]
    focal_init = {}
    focal_init[0] = focal[0]
    focal_init[1] = focal[1]

    # save all in one dictionnary
    adjustable_elements = {}
    adjustable_elements['angles'] = cam_angles
    adjustable_elements['angles_init'] = cam_angles_init
    adjustable_elements['orig'] = origin
    adjustable_elements['orig_init'] = origin_init
    adjustable_elements['focal'] = focal
    adjustable_elements['focal_init'] = focal_init

    return adjustable_elements

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

def toggle_scatter_remarkables():
    global sc_uv_rkables
    global sc_uv_rkables_from_geo
    global sc_geo_rkables
    global sc_geo_rkables_from_pix
    if sc_uv_rkables is None:
        # Ajouter les scatters
        sc_uv_rkables = ax1.scatter(df_corresp_pts_remarquables['U_undist'], df_corresp_pts_remarquables['V_undist'],
                                    c='b', s=6, label='remarkables')
        sc_uv_rkables_from_geo = ax1.scatter(u_remarkables_from_geo, v_remarkables_from_geo, c='cyan', s=6,
                                             label='remarkables from geo')
        sc_geo_rkables = ax2.scatter(df_corresp_pts_remarquables['easting'], df_corresp_pts_remarquables['northing'],
                                    c='b', s=6, label='remarkables')
        sc_geo_rkables_from_pix = ax2.scatter(xyz_remarkables_from_pix[0, :], xyz_remarkables_from_pix[1, :], c='cyan',
                                              s=6, label='remarkables from pix')
    else:
        # Supprimer le scatter existant
        sc_uv_rkables.remove()
        sc_uv_rkables = None
        sc_uv_rkables_from_geo.remove()
        sc_uv_rkables_from_geo = None
        sc_geo_rkables.remove()
        sc_geo_rkables = None
        sc_geo_rkables_from_pix.remove()
        sc_geo_rkables_from_pix = None
        reset_sliders()
        # cam_angles = cam_angles_init

    ax1.legend(fontsize=16)
    ax2.legend(fontsize=16)

    plot.update()

def toggle_mnt():
    global sc_geo_mnt
    global sc_uv_mnt_from_geo
    if sc_geo_mnt is None:
        sc_uv_mnt_from_geo = ax1.scatter(mnt_u_from_geo, mnt_v_from_geo, c= mnt_z, cmap='RdYlBu_r', s=2,
                                         label='mnt drone from geo', alpha=0.7)
        sc_geo_mnt = ax2.scatter(mnt_x, mnt_y, c=mnt_z, cmap='RdYlBu_r', s=6, label='mnt drone')
        ax1.set_xlim([2100, 2390])
        ax1.set_ylim([1100, 1330])
        ax1.yaxis.set_inverted(True)
    else:
        # Supprimer le scatter existant
        sc_geo_mnt.remove()
        sc_geo_mnt = None
        sc_uv_mnt_from_geo.remove()
        sc_uv_mnt_from_geo = None

    ax2.legend(fontsize=16)
    plot.update()

def update_plot(value, key, i_key):

    # get value from slider, and apply it to the corresponding adjustable element
    adjustable_elements[key][i_key] = value

    # compute new extrinsinc parameters if necessary
    if key in ['orig', 'angles']:
        # Calculer le nouveau georef_params
        updated_georef = georef_params.extrinsic.from_origin_beachcam_angles(adjustable_elements['orig'],
                                                                             adjustable_elements['angles'])
        georef_params.extrinsic = updated_georef
    # compute new intrinsinc parameters (focal lengths) if necessary
    elif key == 'focal':
        intrinsinc = IntrinsicMatrix(value, value, georef_params.intrinsic.cx, georef_params.intrinsic.cy)
        georef_params.intrinsic = intrinsinc

    # Calcul de uv_remarkables_from_geo et xyz_remarkables_from_pix en fonction du slider
    if sc_uv_rkables is not None:
        xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo = (
            compute_xyz_from_pix_and_uv_from_geo(df_corresp_pts_remarquables[['U', 'V', 'elevation']],
                                                 np.array(df_corresp_pts_remarquables[
                                                                          ['easting', 'northing', 'elevation']]),
                                                 georef_params))

        # Mise à jour des scatter plots
        sc_uv_rkables_from_geo.set_offsets(np.c_[u_remarkables_from_geo, v_remarkables_from_geo])
        sc_geo_rkables_from_pix.set_offsets(np.c_[xyz_remarkables_from_pix[0, :], xyz_remarkables_from_pix[1, :]])
    if sc_uv_mnt_from_geo is not None:
        mnt_u_from_geo, mnt_v_from_geo = world_2_pix(np.vstack([mnt_x, mnt_y, mnt_z]), georef_params)
        # Mise à jour des scatter plots
        sc_uv_mnt_from_geo.set_offsets(np.c_[mnt_u_from_geo, mnt_v_from_geo])

    plot.update()

def add_slider(sliders, label, key, i_key, dminmax, step):
    ui.label(label)
    sliders[f'{key}_{i_key}'] = ui.slider(min=adjustable_elements[f'{key}_init'][i_key] - dminmax,
                                          max=adjustable_elements[f'{key}_init'][i_key] + dminmax,
                                          value=adjustable_elements[key][i_key], step=step,
                                          on_change=lambda e: update_plot(e.value, key, i_key)).props('label')
    return sliders

def reset_sliders():
    for name, slider in sliders.items():
        key = name.split('_')[0]
        i_key = int(name.split('_')[1])
        slider.value = adjustable_elements[key + '_init'][i_key]

def write_adjusted_camera_parameters():

    # load original camera parameters
    param_dict = read_json_file(f_camera_parameters)
    validated_parameters = CameraParameters.model_validate(param_dict)
    validated_params_dict = CameraParameters.model_dump(validated_parameters)

    # update camera parameters with updated rvec, tvec
    validated_params_dict['extrinsic_parameters']['rvec']  = list(np.squeeze(georef_params.extrinsic.rvec.astype(float)))
    validated_params_dict['extrinsic_parameters']['tvec'] = list(np.squeeze(georef_params.extrinsic.tvec.astype(float)))
    camera_parameters = CameraParameters(**validated_params_dict)
    camera_parameters_dict = CameraParameters.model_dump(camera_parameters)

    # save updated camera parameters to json
    json_str = json.dumps(camera_parameters_dict, indent=2)
    with open("adjusted_camera_parameters.json", "w") as f:
        f.write(json_str)


# parameters
f_img = '/home/florent/Projects/Etretat/Etretat_central2/images/raw/A_Etretat_central2_2fps_600s_20240223_14_00.jpg'
f_corresp_pts_remarquables = '/home/florent/Projects/Etretat/Geodesie/GCPS/GCPS_20200113/fichier_correspondances_Etretat_gcp_mars_2020_avec_images_before_march_2024.csv'
f_camera_parameters = 'camera_parameters_cam44.json'
f_ortho = '/home/florent/Projects/Etretat/Geodesie/orthophoto_2025.tif'
f_mnt = '/home/florent/Projects/Etretat/Geodesie/MNT_drone/03_etretat_20210402_DEM_selection_groyne_medium.tif'


# read ortho
extent_ortho, data_ortho = read_ortho(f_ortho)

# get georef parameters
georef_params = Georef.from_param_file(f_camera_parameters)

# get camera angles
adjustable_elements = get_adjustable_elements(georef_params)

# read raw image
img = read_img(f_img)

# read remarkable points in correspondance file
df_corresp_pts_remarquables, xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo = read_remarkable_pts(f_corresp_pts_remarquables)

# read mnt drone
mnt_z, mnt_x, mnt_y, mnt_u_from_geo, mnt_v_from_geo = read_tif_mnt(f_mnt, 2154, georef_params)

# plot raw and ortho images
with ui.matplotlib(figsize=(28, 12), tight_layout=True) as plot:
    ax1 = plot.figure.add_subplot(121)
    ax1.imshow(img)
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax2 = plot.figure.add_subplot(122)
    ax2.imshow(data_ortho, extent=extent_ortho, aspect='equal')
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax2.set_xlim([298245, 298295])
    ax2.set_ylim([5509940, 5509990])
    plot.update()


# Variables pour stocker les scatters
sc_uv_rkables = None
sc_uv_rkables_from_geo = None
sc_geo_rkables = None
sc_geo_rkables_from_pix = None
sc_geo_mnt = None
sc_uv_mnt_from_geo = None


# Buttons NiceGUI
with ui.button_group().classes('mx-auto'):

    # bouton points remarquables
    button_rkables = ui.button('remarkable pts', on_click=toggle_scatter_remarkables)
    button_rkables.style('width: 200px; height: 20px; font-size: 15px;')

    # bouton mnt drone
    button_topo_pts = ui.button('mnt drone', on_click=toggle_mnt)
    button_topo_pts.style('width: 200px; height: 20px; font-size: 15px;')

    # bouton save georef
    button_save_georef = ui.button('save georef', on_click=write_adjusted_camera_parameters)
    button_save_georef.style('width: 200px; height: 20px; font-size: 15px;')


# Sliders NiceGUI
# Ajouter du CSS personnalisé qui s'applique à tous les labels de sliders
ui.add_head_html('''
<style>
        :root {
            --nicegui-default-padding: 0.5rem;
            --nicegui-default-gap: 0.2rem;
        }
    </style>
# <style>
# .q-slider__label {
#     font-size: 20px;  /* taille du texte */
#     font-weight: bold;           /* facultatif, mettre en gras */
#     color: darkblue;             /* changer la couleur */
# }
# </style>
''')
sliders = {}
sliders = add_slider(sliders, label='yaw (°)', key='angles', i_key=0, dminmax=0.5, step=0.01)
sliders = add_slider(sliders, label='pitch (°)', key='angles', i_key=1, dminmax=0.5, step=0.01)
sliders = add_slider(sliders, label='roll (°)', key='angles', i_key=2, dminmax=0.5, step=0.01)
sliders = add_slider(sliders, label="camera's origin x (local coordinates)", key='orig', i_key=0, dminmax=2, step=0.1)
sliders = add_slider(sliders, label="camera's origin y (local coordinates)", key='orig', i_key=1, dminmax=2, step=0.1)
sliders = add_slider(sliders, label="camera's origin z (local coordinates)", key='orig', i_key=2, dminmax=2, step=0.1)
sliders = add_slider(sliders, label="focal (pixels)", key='focal', i_key=0, dminmax=200, step=5)
# sliders = add_slider(sliders, label="fy (pixels)", key='focal', i_key=0, dminmax=200, step=10)


ui.run()

# Bouton pour tout réinitialiser au niveau des sliders
ui.button('Reset sliders', on_click=reset_sliders)