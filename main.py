from nicegui import ui
import numpy as np
import cv2
from copy import copy
import pandas as pd
from georef.operators import Georef, ExtrinsicMatrix
import matplotlib.pyplot as plt
import georaster
import tifffile

def read_ortho(f_ortho):

    # extent of tif file
    band1 = georaster.SingleBandRaster(f_ortho, load_data=False)
    extent = band1.extent # xmin, xmax, ymin, ymax

    # get data
    data = tifffile.imread(f_ortho)

    return extent, data

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
                                             label='remarquables from geo')
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
        cam_angles = cam_angles_init

    ax1.legend(fontsize=16)
    ax2.legend(fontsize=16)

    plot.update()

def update_plot(new_angle, i_angle, origin):

    cam_angles[i_angle]  = new_angle

    # Calculer le nouveau georef_params
    updated_georef = georef_params.extrinsic.from_origin_beachcam_angles(origin, cam_angles)
    georef_params.extrinsic = updated_georef

    # Mettre à jour les points u_remarkables_from_geo et xyz_remarkables_from_pix en fonction du slider
    xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo = (
        compute_xyz_from_pix_and_uv_from_geo(df_corresp_pts_remarquables[['U', 'V', 'elevation']],
                                             np.array(df_corresp_pts_remarquables[
                                                                      ['easting', 'northing', 'elevation']]),
                                             georef_params))

    # Mettre à jour les scatter
    sc_uv_rkables_from_geo.set_offsets(np.c_[u_remarkables_from_geo, v_remarkables_from_geo])
    sc_geo_rkables_from_pix.set_offsets(np.c_[xyz_remarkables_from_pix[0, :], xyz_remarkables_from_pix[1, :]])
    plot.update()

def reset_sliders():
    for name, slider in sliders.items():
        slider.value = cam_angles_init[name]

# parameters
f_img = '/home/florent/Projects/Etretat/Etretat_central2/images/raw/A_Etretat_central2_2fps_600s_20240223_14_00.jpg'
f_corresp_pts_remarquables = '/home/florent/Projects/Etretat/Geodesie/GCPS/GCPS_20200113/fichier_correspondances_Etretat_gcp_mars_2020_avec_images_before_march_2024.csv'
f_camera_parameters = 'camera_parameters_cam44.json'
f_ortho = '/home/florent/Projects/Etretat/Geodesie/orthophoto_2025.tif'


# read ortho
extent_ortho, data_ortho = read_ortho(f_ortho)

# get georef parameters
georef_params = Georef.from_param_file(f_camera_parameters)

# read camera angles
cam_angles = georef_params.extrinsic.beachcam_angles
cam_angles_init_tmp = copy(cam_angles)
cam_angles_init = {}
cam_angles_init[0] = cam_angles_init_tmp[0]
cam_angles_init[1] = cam_angles_init_tmp[1]
cam_angles_init[2] = cam_angles_init_tmp[2]
origin = georef_params.extrinsic.origin

# read raw image
img = cv2.imread(f_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# undistort image
img = georef_params.undistort(img)

# read remarkable points in correspondance file
df_corresp_pts_remarquables = pd.read_csv(f_corresp_pts_remarquables, usecols=['easting', 'northing', 'elevation', 'U', 'V'])
# create U_undist and V_undist
undist_pts = cv2.undistortPoints(np.array(df_corresp_pts_remarquables[['U', 'V']]).astype(float),
                                 georef_params.intrinsic_parameters.camera_matrix, georef_params.distortion_coefficients.array,
                                 P=georef_params.intrinsic_parameters.camera_matrix)
undist_pts = undist_pts.reshape((undist_pts.shape[0], undist_pts.shape[2]))
df_corresp_pts_remarquables['U_undist'] = undist_pts[:, 0]
df_corresp_pts_remarquables['V_undist'] = undist_pts[:, 1]

xyz_remarkables_from_pix, u_remarkables_from_geo, v_remarkables_from_geo = (
    compute_xyz_from_pix_and_uv_from_geo(df_corresp_pts_remarquables[['U', 'V', 'elevation']],
                                         np.array(df_corresp_pts_remarquables[['easting', 'northing', 'elevation']]),
                                         georef_params))

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
    plot.update()  # <- indispensable pour afficher dès le début


# Variables pour stocker les scatters
sc_uv_rkables = None
sc_uv_rkables_from_geo = None
sc_geo_rkables = None
sc_geo_rkables_from_pix = None


# boutons
with ui.button_group():

    # bouton points remarquables
    button_rkables = ui.button('remarkable pts', on_click=toggle_scatter_remarkables)
    button_rkables.style('width: 250px; height: 80px; font-size: 20px;')

    # bouton 3d to points
    button_topo_pts = ui.button('topo pts', )
    button_topo_pts.style('width: 200px; height: 80px; font-size: 20px;')


# Sliders NiceGUI

# Ajouter du CSS personnalisé qui s'applique à tous les labels de sliders
ui.add_head_html('''
<style>
.q-slider__label {
    font-size: 20px !important;  /* taille du texte */
    font-weight: bold;           /* facultatif, mettre en gras */
    color: darkblue;             /* changer la couleur */
}
</style>
''')

sliders = {}
ui.label('angle 1')
sliders[0] = ui.slider(min=cam_angles_init[0] - 0.5, max=cam_angles_init[0] + 0.5, value=cam_angles[0],
                   step=0.1, on_change=lambda e1: update_plot(e1.value, 0, origin)).props('label')
ui.label().bind_text_from(sliders[0], 'value')

ui.label('angle 2')
sliders[1] = ui.slider(min=cam_angles_init[1] - 1, max=cam_angles_init[1] + 1, value=cam_angles[1],
                    step=0.1, on_change=lambda e2: update_plot(e2.value, 1, origin))
ui.label().bind_text_from(sliders[1], 'value')

ui.label('angle 3')
sliders[2] = ui.slider(min=cam_angles_init[2] - 1, max=cam_angles_init[2] + 1, value=cam_angles[2],
                    step=0.1, on_change=lambda e3: update_plot(e3.value, 2, origin))
ui.label().bind_text_from(sliders[2], 'value')
ui.run()

# Bouton pour tout réinitialiser au niveau des sliders
ui.button('Reset sliders', on_click=reset_sliders)