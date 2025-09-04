import json
from nicegui import ui
from georef.in_out import CameraParameters, read_json_file
import numpy as np
from copy import copy
from nicegui.element import Element
from georef.operators import Georef, IntrinsicMatrix
import io
import base64
from read_inputs import read_ortho, read_img, read_gcps, read_tif_mnt
from geo import world_2_pix, compute_xyz_from_pix_and_uv_from_geo, reprojection_error
from adjustText import adjust_text


def get_initial_camera_params(georef_params):
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
    initial_camera_params = {}
    initial_camera_params['angles'] = cam_angles_init
    initial_camera_params['orig'] = origin_init
    initial_camera_params['focal'] = focal_init

    return initial_camera_params

def toggle_scatter_gcps():
    global sc_uv_gcps
    global sc_uv_gcps_from_geo
    global sc_geo_gcps
    global sc_geo_gcps_from_pix
    global txt_err_reproj
    global txt_err_reproj_stats

    if sc_uv_gcps is None:
        # Ajouter les scatters

        # u, v
        sc_uv_gcps = ax1.scatter(df_gcps['U_undist'], df_gcps['V_undist'],
                                 c='b', s=6, label='gcps')
        sc_uv_gcps_from_geo = ax1.scatter(u_gcps_from_geo, v_gcps_from_geo, c='cyan', s=6,
                                          label='gcps from geo')
        # geo
        sc_geo_gcps = ax2.scatter(df_gcps['easting'], df_gcps['northing'],
                                  c='b', s=6, label='gcps')
        sc_geo_gcps_from_pix = ax2.scatter(xyz_gcps_from_pix[0, :], xyz_gcps_from_pix[1, :], c='cyan',
                                           s=6, label='gcps from pix')

        set_sc_axis_limits(ax1, 0, width, 0, height, reverse_yaxis=True, margin=300)
        set_sc_axis_limits(ax2, extent_ortho[0], extent_ortho[1], extent_ortho[2], extent_ortho[3], margin=20)
        # reprojection error
        reproj_error = reprojection_error(xyz_gcps_from_pix[0:2, :].T, np.array(df_gcps[['easting', 'northing']]))
        labels = ["{:.2f}".format(reproj_error[i]) for i in range(len(reproj_error))]
        txt_err_reproj = [ax2.text(xi, yi, label)
                          for xi, yi, label in zip(x_gcps_from_pix, y_gcps_from_pix, labels)]
        adjust_text(txt_err_reproj, ax=ax2)
        txt_err_reproj_stats =  ax2.text(0.2, 0.9,
                                         'Reprojection errors (mean, std):\n{:.2f}, {:.2f}'.format(
                                             reproj_error.mean(), reproj_error.std()),
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=ax2.transAxes, fontsize=16, clip_on=True)

    else:
        # Supprimer le scatter existant
        sc_uv_gcps.remove()
        sc_uv_gcps = None
        sc_uv_gcps_from_geo.remove()
        sc_uv_gcps_from_geo = None
        sc_geo_gcps.remove()
        sc_geo_gcps = None
        sc_geo_gcps_from_pix.remove()
        sc_geo_gcps_from_pix = None
        [txt_err_reproj[i].remove() for i in range(len(txt_err_reproj))]
        txt_err_reproj = None
        txt_err_reproj_stats.remove()
        txt_err_reproj_stats = None
        reset_sliders()
        set_sc_axis_limits(ax1, 0, width, 0, height, reverse_yaxis=True, margin=300)
        set_sc_axis_limits(ax2, extent_ortho[0], extent_ortho[1], extent_ortho[2], extent_ortho[3], margin=20)
    ax1.legend(fontsize=16)
    ax2.legend(fontsize=16)

    optimized_update_plot()

def toggle_mnt():
    global sc_geo_mnt
    global sc_uv_mnt_from_geo
    if sc_geo_mnt is None:
        sc_uv_mnt_from_geo = ax1.scatter(mnt_u_from_geo, mnt_v_from_geo, c= mnt_z, cmap='RdYlBu_r', s=2,
                                         label='mnt drone from geo', alpha=0.7)
        sc_geo_mnt = ax2.scatter(mnt_x, mnt_y, c=mnt_z, cmap='RdYlBu_r', s=6, label='mnt drone')
        ax1.set_xlim([2100, 2390])
        ax1.set_ylim([1100, 1330])
    else:
        # Supprimer le scatter existant
        sc_geo_mnt.remove()
        sc_geo_mnt = None
        sc_uv_mnt_from_geo.remove()
        sc_uv_mnt_from_geo = None
        reset_sliders()
    set_sc_axis_limits(ax1, 0, width, 0, height, reverse_yaxis=True, margin=25)
    set_sc_axis_limits(ax2, extent_ortho[0], extent_ortho[1], extent_ortho[2], extent_ortho[3], margin=5)
    ax2.legend(fontsize=16)
    optimized_update_plot()

def update_plot():
    global flag_refresh
    global txt_err_reproj
    global txt_err_reproj_stats
    if not flag_refresh:
        return

    # get values from sliders and reconstruct georef params (extrinsinc, intrinsic)

    # extrinsinc params
    angles = np.array([sliders[f'angles_{i_angle}'].value for i_angle in range(3)])
    camera_position = np.array([sliders[f'orig_{i_orig}'].value for i_orig in range(3)])
    extrinsinc = georef_params.extrinsic.from_origin_beachcam_angles(camera_position, angles)
    georef_params.extrinsic = extrinsinc

    # intrinsinc params
    intrinsinc = IntrinsicMatrix(sliders['focal_0'].value, sliders['focal_0'].value, georef_params.intrinsic.cx,
                                 georef_params.intrinsic.cy)
    georef_params.intrinsic = intrinsinc

    # Calcul de uv_gcps_from_geo et xyz_gcps_from_pix
    if sc_uv_gcps is not None:
        xyz_gcps_from_pix, u_gcps_from_geo, v_gcps_from_geo = (
            compute_xyz_from_pix_and_uv_from_geo(df_gcps[['U', 'V', 'elevation']],
                                                 np.array(df_gcps[
                                                              ['easting', 'northing', 'elevation']]),
                                                 georef_params))
        # calcul erreur de reprojection
        reproj_error = reprojection_error(xyz_gcps_from_pix[0:2, :].T, np.array(df_gcps[['easting', 'northing']]))

        # Mise à jour des scatter plots
        sc_uv_gcps_from_geo.set_offsets(np.c_[u_gcps_from_geo, v_gcps_from_geo])
        sc_geo_gcps_from_pix.set_offsets(np.c_[xyz_gcps_from_pix[0, :], xyz_gcps_from_pix[1, :]])
        [txt_err_reproj[i].remove() for i in range(len(txt_err_reproj))]
        txt_err_reproj_stats.remove()
        labels = ["{:.2f}".format(reproj_error[i]) for i in range(len(reproj_error))]
        txt_err_reproj = [ax2.text(xi, yi, label)
                          for xi, yi, label in zip(x_gcps_from_pix, y_gcps_from_pix, labels)]
        adjust_text(txt_err_reproj, ax=ax2)
        txt_err_reproj_stats = ax2.text(0.2, 0.9,
                                        'Reprojection errors (mean, std):\n{:.2f}, {:.2f}'.format(
                                            reproj_error.mean(), reproj_error.std()),
                                        horizontalalignment='center', verticalalignment='center',
                                        transform=ax2.transAxes, fontsize=16)

    # Calcul des mnts points pix
    if sc_uv_mnt_from_geo is not None:
        mnt_u_from_geo, mnt_v_from_geo = world_2_pix(np.vstack([mnt_x, mnt_y, mnt_z]), georef_params)
        # Mise à jour des scatter plots
        sc_uv_mnt_from_geo.set_offsets(np.c_[mnt_u_from_geo, mnt_v_from_geo])

    optimized_update_plot()

def add_slider(sliders, label, key, i_key, dminmax, step):
    ui.label(label)
    sliders[f'{key}_{i_key}'] = ui.slider(min=initial_camera_params[f'{key}'][i_key] - dminmax,
                                          max=initial_camera_params[f'{key}'][i_key] + dminmax,
                                          value=initial_camera_params[f'{key}'][i_key], step=step,
                                          on_change=lambda e: update_plot()).props('label')
    return sliders

def reset_sliders():
    global flag_refresh
    global georef_params
    flag_refresh = False
    for name, slider in sliders.items():
        key = name.split('_')[0]
        i_key = int(name.split('_')[1])
        slider.value = initial_camera_params[key][i_key]
    flag_refresh = True
    georef_params = georef_params_init

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

def optimized_update_plot():
    global plot
    with io.BytesIO() as output:
        plot.figure.savefig(output, format='jpg')
        output.seek(0)
        img_base64 = base64.b64encode(output.read()).decode()
        plot._props['innerHTML'] = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    Element.update(plot)

def set_sc_axis_limits(ax, xmin_orig, xmax_orig, ymin_orig, ymax_orig, reverse_yaxis=False, margin=50):
    # Récupérer tous les PathCollection de l'axe (scatter = PathCollection)
    from matplotlib.collections import PathCollection
    all_offsets = []
    for coll in ax.collections:
        if isinstance(coll, PathCollection):  # ✅ correction ici
            offsets = coll.get_offsets()
            if len(offsets) > 0:
                all_offsets.append(offsets)

    # Si on a trouvé des points scatter, recalculer les bornes
    if all_offsets:
        all_offsets = np.vstack(all_offsets)  # concatène tous les points
        # Limites avec marge mais bornées à l'image
        x_min = max(xmin_orig, all_offsets[:, 0].min() - margin)
        x_max = min(xmax_orig, all_offsets[:, 0].max() + margin)
        y_min = max(ymin_orig, all_offsets[:, 1].min() - margin)
        y_max = min(ymax_orig, all_offsets[:, 1].max() + margin)
        print()
        ax.set_xlim(x_min, x_max)
        if reverse_yaxis:
            ax.set_ylim(y_max, y_min)  # y axis inversion
        else:
            ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(xmin_orig, xmax_orig)
        if reverse_yaxis:
            ax.set_ylim(ymax_orig, ymin_orig)  # y axis inversion
        else:
            ax.set_ylim(ymin_orig, ymax_orig)

# parameters
f_img = '/home/florent/Projects/Etretat/Etretat_central2/images/raw/A_Etretat_central2_2fps_600s_20240223_14_00.jpg'
# f_gcps = '/home/florent/Projects/Etretat/Geodesie/GCPS/GCPS_20200113/fichier_correspondances_Etretat_gcp_mars_2020_avec_images_before_march_2024.csv'
f_gcps = '/home/florent/Projects/Etretat/Geodesie/GCPS_2024/gcps_CAM44.csv'
f_camera_parameters = 'camera_parameters_cam44.json'
# f_camera_parameters = 'camera_parameters_cam44_orig.json'
f_ortho = '/home/florent/Projects/Etretat/Geodesie/orthophoto_2025.tif'
f_mnt = '/home/florent/Projects/Etretat/Geodesie/MNT_drone/03_etretat_20210402_DEM_selection_groyne_medium.tif'


# read ortho
extent_ortho, data_ortho = read_ortho(f_ortho)

# get georef parameters
georef_params = Georef.from_param_file(f_camera_parameters)
georef_params_init = copy(georef_params)

# get camera angles
initial_camera_params = get_initial_camera_params(georef_params)

# read raw image
img = read_img(f_img, georef_params_init)
height, width, _ = img.shape

# read gcps points
df_gcps, xyz_gcps_from_pix, u_gcps_from_geo, v_gcps_from_geo = read_gcps(f_gcps, georef_params_init)
x_gcps_from_pix = xyz_gcps_from_pix[0, :]
y_gcps_from_pix = xyz_gcps_from_pix[1, :]

# read mnt drone
mnt_z, mnt_x, mnt_y, mnt_u_from_geo, mnt_v_from_geo = read_tif_mnt(f_mnt, 2154, georef_params)

# plot raw and ortho images
with ui.matplotlib(figsize=(20, 12), tight_layout=True) as plot:
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
    optimized_update_plot()

# Variables pour stocker les scatters
sc_uv_gcps = None
sc_uv_gcps_from_geo = None
sc_geo_gcps = None
sc_geo_gcps_from_pix = None
sc_err_reproj = None
txt_err_reproj = None
txt_err_reproj_stats = None
sc_geo_mnt = None
sc_uv_mnt_from_geo = None
flag_refresh = True

# Buttons NiceGUI
with ui.button_group().classes('mx-auto'):

    # bouton points gcps
    button_gcps = ui.button('gcps pts', on_click=toggle_scatter_gcps)
    button_gcps.style('width: 200px; height: 20px; font-size: 15px;')

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
        .q-slider__label {
    font-size: 20px !important;  /* taille du texte */
    font-weight: bold;           /* facultatif, mettre en gras */
    color: darkblue;             /* changer la couleur */
}
    </style>
''')
sliders = {}
sliders = add_slider(sliders, label='yaw (°)', key='angles', i_key=0, dminmax=0.5, step=0.01)
sliders = add_slider(sliders, label='pitch (°)', key='angles', i_key=1, dminmax=1.5, step=0.01)
sliders = add_slider(sliders, label='roll (°)', key='angles', i_key=2, dminmax=0.5, step=0.01)
sliders = add_slider(sliders, label="camera's origin x (local coordinates)", key='orig', i_key=0, dminmax=2, step=0.1)
sliders = add_slider(sliders, label="camera's origin y (local coordinates)", key='orig', i_key=1, dminmax=2.5, step=0.1)
sliders = add_slider(sliders, label="camera's origin z (local coordinates)", key='orig', i_key=2, dminmax=2.5, step=0.1)
sliders = add_slider(sliders, label="focal (pixels)", key='focal', i_key=0, dminmax=200, step=5)

ui.run()