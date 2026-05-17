# Copyright (C) 2024  Max Planck Institute for Intelligent Systems Tuebingen, Marilyn Keller 

import argparse
import os
import numpy as np
import yaml
import imgui

from aitviewer.renderables.lines import Lines
from aitviewer.renderables.osim import OSIMSequence
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.markers import Markers
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer

import config as cg
from smpl2ab.markers.smpl_markers import SmplMarker
from smpl2ab.utils.smpl_utils import load_smpl_seq


# ---------------------------------------------------------------------------
# Joint-angle panel configuration
# ---------------------------------------------------------------------------

# OpenSim .mot DOF groups  (column names from the IK output)
OSIM_GROUPS = [
    ("Hip R",      ["hip_flexion_r",  "hip_adduction_r",  "hip_rotation_r"]),
    ("Hip L",      ["hip_flexion_l",  "hip_adduction_l",  "hip_rotation_l"]),
    ("Knee",       ["knee_angle_r",   "knee_angle_l"]),
    ("Ankle",      ["ankle_angle_r",  "ankle_angle_l"]),
    ("Shoulder R", ["shoulder_r_x",   "shoulder_r_y",     "shoulder_r_z"]),
    ("Shoulder L", ["shoulder_l_x",   "shoulder_l_y",     "shoulder_l_z"]),
    ("Elbow",      ["elbow_flexion_r","elbow_flexion_l"]),
    ("Wrist R",    ["wrist_flexion_r","wrist_deviation_r"]),
    ("Wrist L",    ["wrist_flexion_l","wrist_deviation_l"]),
]

# SMPL body-joint groups: (display_name, [(dof_id, body_joint_idx, axis_idx), ...])
# body_joint_idx is 0-based into poses_body (joint 1 = left hip = idx 0, etc.)
SMPL_GROUPS = [
    ("Hip R",      [("hip_r_x",   1, 0), ("hip_r_y",   1, 1), ("hip_r_z",   1, 2)]),
    ("Hip L",      [("hip_l_x",   0, 0), ("hip_l_y",   0, 1), ("hip_l_z",   0, 2)]),
    ("Knee R",     [("knee_r_x",  4, 0), ("knee_r_y",  4, 1), ("knee_r_z",  4, 2)]),
    ("Knee L",     [("knee_l_x",  3, 0), ("knee_l_y",  3, 1), ("knee_l_z",  3, 2)]),
    ("Ankle R",    [("ankle_r_x", 7, 0), ("ankle_r_y", 7, 1), ("ankle_r_z", 7, 2)]),
    ("Ankle L",    [("ankle_l_x", 6, 0), ("ankle_l_y", 6, 1), ("ankle_l_z", 6, 2)]),
    ("Shoulder R", [("shr_x",    16, 0), ("shr_y",    16, 1), ("shr_z",    16, 2)]),
    ("Shoulder L", [("shl_x",    15, 0), ("shl_y",    15, 1), ("shl_z",    15, 2)]),
    ("Elbow R",    [("elbow_r_x",18, 0), ("elbow_r_y",18, 1), ("elbow_r_z",18, 2)]),
    ("Elbow L",    [("elbow_l_x",17, 0), ("elbow_l_y",17, 1), ("elbow_l_z",17, 2)]),
    ("Wrist R",    [("wrist_r_x",20, 0), ("wrist_r_y",20, 1), ("wrist_r_z",20, 2)]),
    ("Wrist L",    [("wrist_l_x",19, 0), ("wrist_l_y",19, 1), ("wrist_l_z",19, 2)]),
]


def parse_mot(mot_path):
    """Parse an OpenSim .mot file → dict {col_name: np.array[T] in degrees}."""
    with open(mot_path) as f:
        lines = f.readlines()
    in_degrees = False
    header_end = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "indegrees" in low and "yes" in low:
            in_degrees = True
        if line.strip() == "endheader":
            header_end = i + 1
            break
    cols = lines[header_end].strip().split("\t")
    data = {c: [] for c in cols[1:]}  # skip 'time'
    for line in lines[header_end + 1:]:
        parts = line.strip().split()
        if len(parts) < len(cols):
            continue
        for j, col in enumerate(cols[1:], 1):
            data[col].append(float(parts[j]))
    scale = 1.0 if in_degrees else np.degrees(1.0)
    return {k: np.array(v) * scale for k, v in data.items()}


def extract_smpl_angles(poses_body_np):
    """poses_body_np: (T, 69) numpy float array.
    Returns dict {dof_id: np.array[T] in degrees} for joints in SMPL_GROUPS."""
    result = {}
    for _group, dofs in SMPL_GROUPS:
        for dof_id, bi, di in dofs:
            result[dof_id] = np.degrees(poses_body_np[:, bi * 3 + di])
    return result


def load_id_npz(path):
    """Load resistance_band_id.py output .npz.
    Returns (data_dict, arm_cols, anchor (3,), wrist_pos (T,3))."""
    raw       = np.load(path, allow_pickle=True)
    col_names = [str(c) for c in raw['id_col_names']]
    id_matrix = raw['id_data']   # (T, N)
    data_dict = {col_names[i]: id_matrix[:, i] for i in range(len(col_names))}
    arm_cols  = set(str(c) for c in raw['arm_col_names'])
    anchor    = raw['anchor'].astype(np.float32)     # (3,)
    wrist_pos = raw['wrist_pos'].astype(np.float32)  # (T, 3)
    return data_dict, arm_cols, anchor, wrist_pos


class JointAngleViewer(Viewer):
    """Viewer subclass that adds two imgui panels showing joint-angle plots."""

    def __init__(self, osim_data, smpl_data, id_data=None, id_arm_cols=None, **kwargs):
        super().__init__(**kwargs)
        self._osim_data   = osim_data      # {col_name: np.array[T] degrees}
        self._smpl_data   = smpl_data      # {dof_id:  np.array[T] degrees}
        self._id_data     = id_data or {}  # {col_moment: np.array[T] N·m}
        self._id_arm_cols = id_arm_cols or set()
        all_osim = [dof for _, dofs in OSIM_GROUPS for dof in dofs]
        all_smpl = [dof for _, dofs in SMPL_GROUPS for dof, *_ in dofs]
        all_id   = [f"{dof}_moment" for _, dofs in OSIM_GROUPS for dof in dofs]
        self._checked = {k: False for k in all_osim + all_smpl + all_id}
        for col in self._id_arm_cols:
            if col in self._checked:
                self._checked[col] = True
        self.gui_controls["joint_angles"] = self._gui_joint_angles

    # ------------------------------------------------------------------
    # Distinct colours for the superimposed plot
    _SERIES_COLORS = [
        (1.0, 0.35, 0.35, 1.0),  # red
        (0.35, 0.85, 0.35, 1.0),  # green
        (0.35, 0.55, 1.0,  1.0),  # blue
        (1.0, 0.80, 0.20, 1.0),  # yellow
        (0.90, 0.40, 0.90, 1.0),  # purple
        (0.25, 0.90, 0.90, 1.0),  # cyan
        (1.0, 0.60, 0.20, 1.0),  # orange
        (0.60, 0.90, 0.30, 1.0),  # lime
    ]

    def _combined_plot(self, dof_list, data_dict, uid, frame, unit='°'):
        """Draw all currently-checked DOFs from dof_list superimposed on one plot."""
        series = [(d, data_dict[d]) for d in dof_list
                  if self._checked.get(d) and data_dict.get(d) is not None]
        if not series:
            imgui.text_colored("(tick DOFs above to show them here)", 0.5, 0.5, 0.5, 1.0)
            return

        T    = max(len(v) for _, v in series)
        vmin = min(float(v.min()) for _, v in series)
        vmax = max(float(v.max()) for _, v in series)
        vrng = vmax - vmin or 1.0

        w  = max(imgui.get_content_region_available_width(), 50.0)
        h  = 100.0
        imgui.invisible_button(f"##comb_{uid}", w, h)
        rmin = imgui.get_item_rect_min()
        rmax = imgui.get_item_rect_max()
        pw   = rmax[0] - rmin[0]
        ph   = rmax[1] - rmin[1]

        dl = imgui.get_window_draw_list()
        dl.add_rect_filled(rmin[0], rmin[1], rmax[0], rmax[1],
                           imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0))
        dl.add_rect(rmin[0], rmin[1], rmax[0], rmax[1],
                    imgui.get_color_u32_rgba(0.45, 0.45, 0.45, 1.0))

        for ci, (dof, vals) in enumerate(series):
            r, g, b, a = self._SERIES_COLORS[ci % len(self._SERIES_COLORS)]
            col = imgui.get_color_u32_rgba(r, g, b, a)
            n   = len(vals)
            pts = []
            for xi in range(n):
                x = rmin[0] + (xi / max(n - 1, 1)) * pw
                y = rmax[1] - ((float(vals[xi]) - vmin) / vrng) * ph
                pts.append((x, y))
            # draw as connected line segments
            for xi in range(len(pts) - 1):
                dl.add_line(pts[xi][0], pts[xi][1],
                            pts[xi+1][0], pts[xi+1][1], col, 1.2)

        # vertical cursor
        t = frame / max(T - 1, 1)
        cx = rmin[0] + t * pw
        dl.add_line(cx, rmin[1], cx, rmax[1],
                    imgui.get_color_u32_rgba(1.0, 0.85, 0.0, 1.0), 1.5)

        # inline legend
        for ci, (dof, vals) in enumerate(series):
            r, g, b, a = self._SERIES_COLORS[ci % len(self._SERIES_COLORS)]
            cur = float(vals[min(frame, len(vals) - 1)])
            imgui.text_colored(f"{dof}: {cur:.1f}{unit}", r, g, b, a)

    # ------------------------------------------------------------------
    def _plot_row(self, dof_id, values, frame, unit='°'):
        """Draw one checkbox row; if checked, draw plot + vertical cursor."""
        _, self._checked[dof_id] = imgui.checkbox(f"{dof_id}##{dof_id}", self._checked[dof_id])
        if self._checked[dof_id] and values is not None and len(values) > 1:
            T = len(values)
            cur = float(values[min(frame, T - 1)])
            w = max(imgui.get_content_region_available_width(), 50)
            imgui.plot_lines(
                f"##{dof_id}_pl",
                values.astype(np.float32),
                overlay_text=f"{cur:.1f}{unit}",
                scale_min=float(values.min()),
                scale_max=float(values.max()),
                graph_size=(w, 40),
            )
            # Vertical cursor line at current frame
            rmin = imgui.get_item_rect_min()
            rmax = imgui.get_item_rect_max()
            t = frame / max(T - 1, 1)
            x = rmin[0] + t * (rmax[0] - rmin[0])
            dl = imgui.get_window_draw_list()
            dl.add_line(x, rmin[1], x, rmax[1],
                        imgui.get_color_u32_rgba(1.0, 0.85, 0.0, 1.0), 1.5)

    def _gui_joint_angles(self):
        frame = self.scene.current_frame_id
        W = self.window_size[0]

        # ---- OpenSim IK panel ----
        imgui.set_next_window_position(W - 330, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(310, 620, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("OpenSim IK Angles")
        if expanded:
            for group_name, dofs in OSIM_GROUPS:
                if imgui.tree_node(group_name):
                    for dof in dofs:
                        self._plot_row(dof, self._osim_data.get(dof), frame)
                    imgui.tree_pop()
            imgui.separator()
            imgui.set_next_item_open(False, imgui.ONCE)
            if imgui.tree_node("Superimposed"):
                all_osim = [dof for _, dofs in OSIM_GROUPS for dof in dofs]
                self._combined_plot(all_osim, self._osim_data, "osim", frame)
                imgui.tree_pop()
        imgui.end()

        # ---- SMPL angles panel ----
        imgui.set_next_window_position(W - 660, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(310, 620, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("SMPL Angles")
        if expanded:
            for group_name, dofs in SMPL_GROUPS:
                if imgui.tree_node(group_name):
                    for dof_id, *_ in dofs:
                        self._plot_row(dof_id, self._smpl_data.get(dof_id), frame)
                    imgui.tree_pop()
            imgui.separator()
            imgui.set_next_item_open(False, imgui.ONCE)
            if imgui.tree_node("Superimposed"):
                all_smpl = [dof for _, dofs in SMPL_GROUPS for dof, *_ in dofs]
                self._combined_plot(all_smpl, self._smpl_data, "smpl", frame)
                imgui.tree_pop()
        imgui.end()

        # ---- Joint Torques panel (only when ID data is loaded) ----
        if self._id_data:
            imgui.set_next_window_position(W - 990, 50, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(310, 620, imgui.FIRST_USE_EVER)
            expanded, _ = imgui.begin("Joint Torques [N\u00b7m]")
            if expanded:
                for group_name, dofs in OSIM_GROUPS:
                    group_keys = [f"{dof}_moment" for dof in dofs
                                  if f"{dof}_moment" in self._id_data]
                    if not group_keys:
                        continue
                    if imgui.tree_node(group_name):
                        for key in group_keys:
                            self._plot_row(key, self._id_data.get(key), frame, unit=' N\u00b7m')
                        imgui.tree_pop()
                imgui.separator()
                imgui.set_next_item_open(False, imgui.ONCE)
                if imgui.tree_node("Superimposed"):
                    all_id = [f"{dof}_moment" for _, dofs in OSIM_GROUPS for dof in dofs]
                    self._combined_plot(all_id, self._id_data, "id", frame, unit=' N\u00b7m')
                    imgui.tree_pop()
            imgui.end()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--osim_path', type=str, help='Path to OpenSim model (.osim)')
    parser.add_argument('--mot_path', type=str,  help='Path to OpenSim motion (.mot)')
    parser.add_argument('--smpl_motion_path', type=str,  help='Path to SMPL motion')
    parser.add_argument('--smpl_markers_path', type=str, default=cg.bsm_markers_on_smpl_path, help='Path to SMPL markers')
    parser.add_argument('--body_model', help='Body model to use (smpl or smplx)', default='smpl', choices=['smpl', 'smplx'])
    parser.add_argument('--gender', type=str, default=None, choices=['female', 'male', 'neutral'])
    parser.add_argument('--z_up', action='store_true', help='Set the z axis up')
    parser.add_argument('--gui', action='store_true', help='Open interactive viewer instead of exporting video')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video (ignored if --gui)')
    parser.add_argument('--offset', type=float, default=0.0, help='X-axis offset (m) between SMPL and OpenSim for side-by-side view')
    parser.add_argument('--load_camera_settings', action='store_true', help='Load saved camera settings')
    parser.add_argument('--id_path', type=str, default=None,
                        help='Path to band_id.npz from resistance_band_id.py (adds Joint Torques panel + band visualization)')

    args = parser.parse_args()
    
    to_display = []
    
    if args.body_model == 'smpl':
        # Full SMPLH models (with hand PCA) are not available on this system.
        # Use plain SMPL — body pose is identical; hands stay in rest pose for AMASS 156-dim data.
        body_model = 'smpl'
    else:
        body_model = args.body_model
        
    if args.gender is None:
        _npz = load_smpl_seq(args.smpl_motion_path)
        gender = _npz['gender']
    else:
        gender = args.gender
        
    smpl_layer = SMPLLayer(model_type=body_model, gender=gender)
    args = parser.parse_args()
    
    to_display = []
    
    fps = 30 

    # Load SMPL motion
    seq_smpl = SMPLSequence.from_amass(
        smpl_layer = smpl_layer,
        npz_data_path=os.path.join(args.smpl_motion_path), # AMASS/CMU/01/01_01_poses.npz
        fps_out=fps,
        name=f"{args.body_model.upper()} motion",
        show_joint_angles=False,
        z_up=args.z_up
    )
    to_display.append(seq_smpl)
   
    # Load result OpenSim motion
    osim_seq = OSIMSequence.from_files(osim_path=args.osim_path, 
                                       mot_file=args.mot_path, 
                                       name=f'OpenSim skeleton motion', 
                                       fps_out = fps,
                                       color_skeleton_per_part=False, 
                                       show_joint_angles=False, 
                                       is_rigged=False,
                                       ignore_geometry=True,
                                       z_up=args.z_up)
    
    if args.offset != 0.0:
        osim_seq.position[0] += args.offset
    to_display.append(osim_seq)


    # Load SMPL markers
    markers_dict = yaml.load(open(args.smpl_markers_path, 'r'), Loader=yaml.FullLoader)
    synthetic_markers = SmplMarker(seq_smpl.vertices, markers_dict, fps=fps, name='Markers')
    markers_seq = Markers(synthetic_markers.marker_trajectory, markers_labels=synthetic_markers.marker_names, 
                          name='SMPL markers',
                          color=(0, 1, 0, 1),
                          z_up=args.z_up)
    to_display.append(markers_seq)
    

    # Display in the viewer
    osim_data = parse_mot(args.mot_path) if args.mot_path else {}
    smpl_data = extract_smpl_angles(seq_smpl.poses_body.detach().cpu().numpy()) if seq_smpl is not None else {}
    if args.id_path:
        id_data, id_arm_cols, band_anchor, band_wrist = load_id_npz(args.id_path)
        # ── Resistance band visualization ──────────────────────────────────
        T = len(band_wrist)
        # Anchor: small red sphere at the fixed floor attachment point
        anchor_pts = np.tile(band_anchor[np.newaxis, np.newaxis], (T, 1, 1))  # (T,1,3)
        to_display.append(Spheres(anchor_pts, radius=0.03,
                                  color=(0.9, 0.15, 0.15, 1.0), name='Band anchor'))
        # Wrist attachment: orange sphere tracking the hand
        wrist_pts = band_wrist[:, np.newaxis, :]  # (T,1,3)
        to_display.append(Spheres(wrist_pts, radius=0.025,
                                  color=(1.0, 0.55, 0.0, 1.0), name='Band attachment'))
        # Band line: yellow cylinder from anchor to wrist
        band_lines = np.stack([
            np.tile(band_anchor, (T, 1)),  # (T,3) anchor repeated
            band_wrist,                    # (T,3) wrist
        ], axis=1)  # (T,2,3)
        to_display.append(Lines(band_lines, r_base=0.005,
                                color=(1.0, 0.9, 0.1, 0.85), mode='lines',
                                name='Resistance band'))
    else:
        id_data, id_arm_cols = {}, set()

    v = JointAngleViewer(osim_data, smpl_data, id_data=id_data, id_arm_cols=id_arm_cols)
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(*to_display)
    
    if seq_smpl is not None:
        v.lock_to_node(seq_smpl, (2, 0.7, 2), smooth_sigma=5.0)
    v.playback_fps = fps

    if args.load_camera_settings:
        v.scene.camera.load_cam()

    if args.gui or args.output is None:
        v.run()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        v.export_video(output_path=args.output, output_fps=fps)