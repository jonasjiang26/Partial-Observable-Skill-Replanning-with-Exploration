"""
Standalone helpers for generating pointclouds from MuJoCo cameras.

Content:
- depth_im_to_meters / camera intrinsics / extrinsics helpers
- extract_pointcloud_from_camera_adapt3r: per-camera RGBD -> world pointcloud
- extract_object_pointcloud: multi-camera merge + mask filtering for one object
"""

import os
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import robosuite.utils.transform_utils as T
from PIL import Image
import h5py
import json

import init_path  # 保证能 import libero
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING

# 设置环境变量以避免OpenGL/EGL警告
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# 抑制OpenGL错误检查以避免警告
import OpenGL
OpenGL.ERROR_CHECKING = False


def create_env_safely(problem_name, env_kwargs):
    """
    安全地创建LIBERO环境，避免OpenGL警告

    Args:
        problem_name (str): 问题名称
        env_kwargs (dict): 环境参数

    Returns:
        env: 创建的环境实例
    """
    import warnings
    import logging

    # 临时抑制警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # 抑制robosuite的日志
        logging.getLogger('robosuite').setLevel(logging.ERROR)

        try:
            env = TASK_MAPPING[problem_name](**env_kwargs)
            return env
        except Exception as e:
            print(f"环境创建失败: {e}")
            raise


# ---------------------------- Camera utilities ---------------------------- #
def depth_im_to_meters(depth_im, sim):
    """Convert normalized MuJoCo depth image (0-1) to meters."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return near / (1 - np.array(depth_im) * (1 - near / far))


def get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width):
    """Return 3x3 pinhole intrinsics for a MuJoCo camera."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, img_width / 2], [0, f, img_height / 2], [0, 0, 1]])


def get_camera_extrinsic_matrix(camera_name, sim):
    """Return 4x4 camera-to-world pose with axis correction matching Adapt3R."""
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    pose = T.make_pose(camera_pos, camera_rot)

    # Align axes so +z looks outward along the view direction.
    axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return pose @ axis_correction


def cammat2o3d(cam_mat, width, height):
    """Convert numpy intrinsics to Open3D PinholeCameraIntrinsic."""
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


# ---------------------------- Pointcloud extraction ---------------------------- #
def extract_pointcloud_from_camera_adapt3r(sim, camera_name, rgb_image, depth_image, img_height, img_width):
    """
    Backproject one camera's RGBD into a world-frame pointcloud using Open3D.

    Returns:
        points_world: (H, W, 3) array aligned with the input image layout.
    """
    depths_m = depth_im_to_meters(depth_image, sim)
    intrinsics = get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width)
    extrinsics = get_camera_extrinsic_matrix(camera_name, sim)

    rgb_im = o3d.geometry.Image(np.ascontiguousarray(rgb_image[::-1]))
    depth_im = o3d.geometry.Image(np.clip(np.ascontiguousarray(depths_m[::-1]), 0, 4))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, depth_im, convert_rgb_to_intensity=False, depth_trunc=5, depth_scale=1
    )

    o3d_cam_mat = cammat2o3d(intrinsics, img_width, img_height)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam_mat)
    transformed_cloud = cloud.transform(extrinsics)

    points_world = np.asarray(transformed_cloud.points).reshape(img_height, img_width, 3)
    return points_world[::-1]


def extract_object_pointcloud(
    env,
    object_name,
    camera_names=("agentview", "robot0_eye_in_hand"),
    resolution=512,
    voxel_size=0.005,
    erode_kernel_size=3,
    outlier_std_ratio=2.0,
    save_debug_files=False,
    verbose=False,
):
    """
    Extract a pointcloud for a specific object by merging multiple camera views.

    Pipeline:
    - Render depth + segmentation per camera
    - Convert RGBD to world points (Adapt3R/Open3D)
    - Mask to target object's geoms; optional erosion to drop noisy edges
    - Concatenate across cameras, remove outliers, voxel downsample
    """
    sim = env.sim

    body_id = object_name if isinstance(object_name, int) else sim.model.body_name2id(object_name)

    object_geom_ids = {gid for gid in range(sim.model.ngeom) if sim.model.geom_bodyid[gid] == body_id}
    if len(object_geom_ids) == 0:
        if verbose:
            print(f"   ⚠️  WARNING: No geometries found for object '{object_name}'")
        return np.zeros((0, 3))

    all_points = []
    for camera_name in camera_names:
        try:
            result = sim.render(width=resolution, height=resolution, camera_name=camera_name, depth=True)
            rgb, depth = result if isinstance(result, tuple) else (None, result)
            depth = np.array(depth)
            seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

            if rgb is not None:
                rgb_image = np.array(rgb)
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                if len(rgb_image.shape) == 2:
                    rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
            else:
                rgb_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            points_world = extract_pointcloud_from_camera_adapt3r(
                sim, camera_name, rgb_image, depth, resolution, resolution
            )
            points_world_flat = points_world.reshape(-1, 3)

            object_mask = np.zeros((resolution, resolution), dtype=bool)
            for geom_id in object_geom_ids:
                object_mask |= seg_img[:, :, 1] == geom_id

            if erode_kernel_size > 0:
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            if save_debug_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "scripts/phase3/pipeline/outputs/pointclouds/mplib_debug"
                os.makedirs(output_dir, exist_ok=True)
                seg_output_path = os.path.join(
                    output_dir, f"object_segmentation_{object_name}_{camera_name}_{timestamp}.png"
                )
                seg_image = (object_mask.astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(seg_image).save(seg_output_path)

            object_points = points_world_flat[object_mask.flatten()]
            if len(object_points) > 0:
                all_points.append(object_points)

        except Exception as e:
            if verbose:
                print(f"   ⚠️  WARNING: Failed to extract from camera '{camera_name}': {e}")
            continue

    if len(all_points) == 0:
        return np.zeros((0, 3))

    merged = np.concatenate(all_points, axis=0)

    if outlier_std_ratio > 0 and len(merged) > 10:
        centroid = merged.mean(axis=0)
        distances = np.linalg.norm(merged - centroid, axis=1)
        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + std_dist * outlier_std_ratio
        merged = merged[distances <= threshold]

    voxel_indices = np.floor(merged / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return merged[unique_indices]

if __name__ == "__main__":


# 1. 读 demo.hdf5
    demo_file = "/home/jing/LIBERO/LIBERO/libero/datasets/KITCHEN_SCENE1_put_all_the_items_in_the_cabinet_demo.hdf5"
    f = h5py.File(demo_file, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    env_kwargs = env_args["env_kwargs"]
    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = env_args["problem_name"]
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    states = f["data/demo_0/states"][()]
    actions = f["data/demo_0/actions"][()]
    print(states.shape, actions.shape)
    model_xml = f[f"data/demo_0"].attrs["model_file"]

    f.close()
    model_xml = libero_utils.postprocess_model_xml(model_xml, {})


    # 2. 补充相机等参数（和 create_dataset.py 基本一致，可以按需改）
    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,  # 不使用窗口渲染器
        has_offscreen_renderer=True,  # 使用离屏渲染器
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=True,   # 要深度图就 True
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=True,
    )

    # 设置渲染器为EGL后端（无头模式）
    env_kwargs['renderer'] = 'mujoco'
    env_kwargs['renderer_config'] = {
        'type': 'offscreen',
        'device_id': 0,
    }

    # 3. 创建 env (使用安全创建函数避免警告)
    env = create_env_safely(problem_name, env_kwargs)
    obs = env.reset()

    env.reset_from_xml_string(model_xml)
    env.sim.set_state_from_flattened(states[5760])
    env.sim.forward()
    model_xml = env.sim.model.get_xml()

    # 4. reset 一下，就可以 step / render / 取 obs 了

print("env 创建成功，obs keys:", obs.keys())
composed_pointcloud = extract_object_pointcloud(env, "glazed_rim_porcelain_ramekin_1_main")#glazed_rim_porcelain_ramekin_1_main
print("点云提取成功，点云 shape:", composed_pointcloud.shape)

# ---------------- 使用 Plotly 进行交互式 3D 可视化 ----------------
if composed_pointcloud.shape[0] == 0:
    print("点云为空，无法可视化")
else:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # 创建交互式 3D 散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=composed_pointcloud[:, 0],
            y=composed_pointcloud[:, 1],
            z=composed_pointcloud[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=composed_pointcloud[:, 2],  # 用 z 坐标作为颜色
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title="Z坐标")
            ),
            name='Point Cloud'
        )])
        
        fig.update_layout(
            title='Point Cloud Visualization - black_book_1_main',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # 保持坐标轴比例
            ),
            width=1000,
            height=800
        )
        
        # 保存为 HTML 文件（可以在浏览器中打开）
        output_html = "black_book_1_main_pointcloud.html"
        fig.write_html(output_html)
        print(f"交互式可视化已保存为 {output_html}")
        print(f"可以在浏览器中打开此文件进行交互式查看（旋转、缩放、平移）")
        
        # 如果环境支持，也可以尝试显示
        try:
            fig.show()
        except:
            print("无法显示窗口，但 HTML 文件已保存")
            
    except ImportError:
        print("Plotly 未安装，使用 Matplotlib 作为备选方案...")
        # 备选方案：使用 Matplotlib
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用 z 坐标作为颜色
        scatter = ax.scatter(
            composed_pointcloud[:, 0],
            composed_pointcloud[:, 1],
            composed_pointcloud[:, 2],
            c=composed_pointcloud[:, 2],
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization - black_book_1_main')
        plt.colorbar(scatter, ax=ax, label='Z坐标')
        
        # 保存图片
        output_png = "black_book_1_main_pointcloud.png"
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        print(f"可视化图片已保存为 {output_png}")
        
        try:
            plt.show()
        except:
            print("无法显示窗口，但图片已保存")
    
    # 同时保存为 PLY 文件（供其他工具使用）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(composed_pointcloud.astype(np.float64))
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    output_ply = "black_book_1_main.ply"
    o3d.io.write_point_cloud(output_ply, pcd)
    print(f"点云文件已保存为 {output_ply}（可用 CloudCompare、MeshLab 等工具打开）")

# 正确关闭环境，避免 EGL 清理警告
try:
    env.close()
except:
    pass