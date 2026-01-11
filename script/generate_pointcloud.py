"""
Standalone helpers for generating pointclouds from MuJoCo cameras.

Content:
- depth_im_to_meters / camera intrinsics / extrinsics helpers
- extract_pointcloud_from_camera_adapt3r: per-camera RGBD -> world pointcloud
- extract_object_pointcloud: multi-camera merge + mask filtering for one object
"""

import os
import sys
from datetime import datetime
import argparse
import cv2
import numpy as np
import open3d as o3d
import robosuite.utils.transform_utils as T
from PIL import Image
import h5py
import json
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING

# 确保可以从项目根目录导入 utils 包（无论当前工作目录在项目根还是 script/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.scene_graph_utils import SceneGraphAnalyzer, MergedThreeDInstance

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


# ---------------------------- Visualization utilities ---------------------------- #
class PointCloudVisualizer:
    """
    点云可视化工具：
    - 优先使用 Plotly 做交互式 3D 可视化
    - 如果 Plotly 不可用，则退回到 Matplotlib
    - 同时支持保存为 HTML / PNG / PLY 文件
    """

    @staticmethod
    def visualize(merged_pointcloud: np.ndarray,
                  title: str = "Point Cloud Visualization",
                  html_path: str = "pointcloud.html",
                  png_path: str = "pointcloud.png",
                  ply_path: str = "pointcloud.ply",
                  show: bool = True) -> None:
        """
        综合可视化入口：先尝试 Plotly，失败则使用 Matplotlib，并且保存为 PLY。
        """
        if merged_pointcloud.size == 0:
            print("点云为空，无法可视化")
            return

        # 1) 先尝试 Plotly
        used_plotly = PointCloudVisualizer._try_visualize_with_plotly(
            merged_pointcloud, title=title, html_path=html_path, show=show
        )

        # 2) 如果 Plotly 不可用，则使用 Matplotlib
        if not used_plotly:
            PointCloudVisualizer._visualize_with_matplotlib(
                merged_pointcloud, title=title, png_path=png_path, show=show
            )

        # 3) 无论使用哪种可视化方式，都保存为 PLY
        PointCloudVisualizer.save_as_ply(merged_pointcloud, ply_path=ply_path)

    @staticmethod
    def _try_visualize_with_plotly(merged_pointcloud: np.ndarray,
                                   title: str,
                                   html_path: str,
                                   show: bool) -> bool:
        """
        使用 Plotly 进行交互式 3D 点云可视化。
        返回值:
            True  如果 Plotly 可用并成功绘制
            False 如果 Plotly 未安装或导入失败
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=merged_pointcloud[:, 0],
                        y=merged_pointcloud[:, 1],
                        z=merged_pointcloud[:, 2],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=merged_pointcloud[:, 2],  # 用 z 坐标作为颜色
                            colorscale="Viridis",
                            opacity=0.8,
                            showscale=True,
                            colorbar=dict(title="Z坐标"),
                        ),
                        name="Point Cloud",
                    )
                ]
            )

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",  # 保持坐标轴比例
                ),
                width=1000,
                height=800,
            )

            # 保存为 HTML 文件（可以在浏览器中打开）
            fig.write_html(html_path)
            print(f"交互式可视化已保存为 {html_path}")
            print("可以在浏览器中打开此文件进行交互式查看（旋转、缩放、平移）")

            if show:
                try:
                    fig.show()
                except Exception:
                    print("无法显示窗口，但 HTML 文件已保存")

            return True

        except ImportError:
            print("Plotly 未安装，使用 Matplotlib 作为备选方案...")
            return False

    @staticmethod
    def _visualize_with_matplotlib(merged_pointcloud: np.ndarray,
                                   title: str,
                                   png_path: str,
                                   show: bool) -> None:
        """使用 Matplotlib 进行静态 3D 点云可视化并保存 PNG。"""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 使用 z 坐标作为颜色
        scatter = ax.scatter(
            merged_pointcloud[:, 0],
            merged_pointcloud[:, 1],
            merged_pointcloud[:, 2],
            c=merged_pointcloud[:, 2],
            cmap="viridis",
            s=10,
            alpha=0.6,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label="Z坐标")

        # 保存图片
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"可视化图片已保存为 {png_path}")

        if show:
            try:
                plt.show()
            except Exception:
                print("无法显示窗口，但图片已保存")

    @staticmethod
    def save_as_ply(merged_pointcloud: np.ndarray, ply_path: str) -> None:
        """将点云保存为 PLY 文件，方便在 CloudCompare、MeshLab 等工具中查看。"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pointcloud.astype(np.float64))
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"点云文件已保存为 {ply_path}（可用 CloudCompare、MeshLab 等工具打开）")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='从LIBERO中合成点云')
    parser.add_argument(
        '--load_path',
        type=str,
        default="/home/jing/LIBERO/LIBERO/libero/datasets/KITCHEN_SCENE1_i_want_to_cook_something_to_eat_demo.hdf5",
        required=False,
        help='h5文件路径'
    )
    parser.add_argument('--frame_id', type=int, required=True, help='帧ID')
    parser.add_argument(
        '--object_name',
        type=str,
        required=True,
        help='物体名称，可以用逗号分隔多个物体，例如 "butter_1_main,black_book_1_main"'
    )

    args = parser.parse_args()

    # 解析物体名称（支持逗号分隔多个）
    object_names = [name.strip() for name in args.object_name.split(",") if name.strip()]
    if len(object_names) == 0:
        print("错误: 至少需要提供 1 个物体名称（可以用逗号分隔多个物体）")
        exit(1)

    load_path = args.load_path
    frame_id = args.frame_id
    demo_file = load_path
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
        has_renderer=False,          # 不使用窗口渲染器
        has_offscreen_renderer=True, # 使用离屏渲染器
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=True,          # 要深度图就 True
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
    env.sim.set_state_from_flattened(states[frame_id])
    env.sim.forward()
    model_xml = env.sim.model.get_xml()

    print("env 创建成功，obs keys:", obs.keys())

    # 4. 为指定物体提取点云并可视化
    merged_pointclouds = []
    instances = []
    for obj_name in object_names:
        pc = extract_object_pointcloud(env, obj_name)
        print(f"点云提取成功 [{obj_name}]，点云 shape:", pc.shape)
        instance = MergedThreeDInstance(obj_name, pc)
        instances.append(instance)
        if pc.size > 0:
            merged_pointclouds.append(pc)

    scene_graph_analyzer = SceneGraphAnalyzer(instances)
    scene_graph_analyzer.print_scene_graph()

    if len(merged_pointclouds) == 0:
        print("所有物体的点云均为空，无法可视化")
    else:
        merged_pointcloud = np.concatenate(merged_pointclouds, axis=0)

        title = f"Point Cloud Visualization - {','.join(object_names)}"
        html_path = "black_book_1_main_pointcloud.html"
        png_path = "black_book_1_main_pointcloud.png"
        ply_path = "black_book_1_main.ply"

        # 统一使用上面定义好的可视化工具（内部自动优先 Plotly，失败回退 Matplotlib）
        PointCloudVisualizer.visualize(
            merged_pointcloud,
            title=title,
            html_path=html_path,
            png_path=png_path,
            ply_path=ply_path,
            show=True,
        )

    # 正确关闭环境，避免 EGL 清理警告
    try:
        env.close()
    except Exception:
        pass