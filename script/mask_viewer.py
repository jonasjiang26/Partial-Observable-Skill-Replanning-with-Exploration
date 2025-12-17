#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
# 添加utils目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mask_utils import MaskExtractor
from utils.depth_projection_utils import ThreeDInstance, merge_multi_frame_pointclouds

def visualize_3d_pointcloud_matplotlib(points_world: np.ndarray, title: str = "3D Point Cloud"):
    """
    使用matplotlib可视化3D点云。
    
    参数:
        points_world: (N, 3) 世界坐标系下的3D点云
        title: 图标题
    """
    # 打印点云统计信息
    print(f"\n点云统计信息:")
    print(f"  点数: {len(points_world)}")
    print(f"  X范围: [{points_world[:, 0].min():.4f}, {points_world[:, 0].max():.4f}] m, "
          f"标准差: {points_world[:, 0].std():.4f} m")
    print(f"  Y范围: [{points_world[:, 1].min():.4f}, {points_world[:, 1].max():.4f}] m, "
          f"标准差: {points_world[:, 1].std():.4f} m")
    print(f"  Z范围: [{points_world[:, 2].min():.4f}, {points_world[:, 2].max():.4f}] m, "
          f"标准差: {points_world[:, 2].std():.4f} m")
    
    # 检查Z轴变化
    z_range = points_world[:, 2].max() - points_world[:, 2].min()
    x_range = points_world[:, 0].max() - points_world[:, 0].min()
    y_range = points_world[:, 1].max() - points_world[:, 1].min()
    
    
    fig = plt.figure(figsize=(20, 12))
    
    # 创建四个子图：等比例、自适应、Z轴放大、侧面视图
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    # 绘制点云（使用Z值作为颜色）
    scatter1 = ax1.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], 
                           c=points_world[:, 2], cmap='viridis', s=2, alpha=0.8)
    scatter2 = ax2.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], 
                           c=points_world[:, 2], cmap='viridis', s=2, alpha=0.8)
    scatter3 = ax3.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], 
                           c=points_world[:, 2], cmap='viridis', s=2, alpha=0.8)
    scatter4 = ax4.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], 
                           c=points_world[:, 2], cmap='viridis', s=2, alpha=0.8)
    
    plt.colorbar(scatter1, ax=ax1, label='Z (m)')
    plt.colorbar(scatter2, ax=ax2, label='Z (m)')
    plt.colorbar(scatter3, ax=ax3, label='Z (m)')
    plt.colorbar(scatter4, ax=ax4, label='Z (m)')
    
    # 设置坐标轴标签（单位：米）
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    ax1.set_title(f'{title}\n(等比例视图)')
    ax2.set_title(f'{title}\n(自适应视图)')
    ax3.set_title(f'{title}\n(Z轴放大视图)')
    ax4.set_title(f'{title}\n(侧面视图 - 观察Z轴)')
    
    # 左图：等比例视图（所有轴使用相同的范围）
    max_range = np.array([x_range, y_range, z_range]).max() / 2.0
    mid_x = (points_world[:, 0].max() + points_world[:, 0].min()) * 0.5
    mid_y = (points_world[:, 1].max() + points_world[:, 1].min()) * 0.5
    mid_z = (points_world[:, 2].max() + points_world[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    ax1.set_box_aspect([1, 1, 1])  # 设置等比例
    
    # 中图：自适应视图（每个轴使用自己的范围）
    ax2.set_xlim(points_world[:, 0].min(), points_world[:, 0].max())
    ax2.set_ylim(points_world[:, 1].min(), points_world[:, 1].max())
    ax2.set_zlim(points_world[:, 2].min(), points_world[:, 2].max())
    
    # 右图：Z轴放大视图（XY轴使用自己的范围，Z轴放大）
    # 计算Z轴中心
    z_center = (points_world[:, 2].max() + points_world[:, 2].min()) * 0.5
    # Z轴放大倍数：使Z轴范围与XY轴中较大的范围相同
    z_scale = max(x_range, y_range) / z_range if z_range > 0 else 1.0
    z_expanded_range = max(x_range, y_range) / 2.0
    
    ax3.set_xlim(points_world[:, 0].min(), points_world[:, 0].max())
    ax3.set_ylim(points_world[:, 1].min(), points_world[:, 1].max())
    ax3.set_zlim(z_center - z_expanded_range, z_center + z_expanded_range)
    
    # 右下图：侧面视图（从Y轴方向看，突出Z轴变化）
    ax4.set_xlim(points_world[:, 0].min(), points_world[:, 0].max())
    ax4.set_ylim(points_world[:, 1].min(), points_world[:, 1].max())
    ax4.set_zlim(points_world[:, 2].min(), points_world[:, 2].max())
    
    # 设置不同的视角以便更好地观察
    ax1.view_init(elev=20, azim=45)  # 标准视角
    ax2.view_init(elev=20, azim=45)  # 标准视角
    ax3.view_init(elev=30, azim=60)  # 稍微倾斜以便看到Z轴变化
    ax4.view_init(elev=0, azim=90)   # 从侧面看（Y轴方向），突出Z轴变化
    
    plt.tight_layout()
    plt.show()
    
    # 额外提示和分析
    z_xy_ratio = z_range / max(x_range, y_range)
    print(f"\n点云形状分析（单位：米）:")
    print(f"  Z轴范围: {z_range:.4f} m")
    print(f"  XY轴最大范围: {max(x_range, y_range):.4f} m")
    print(f"  Z/XY比例: {z_xy_ratio:.4f} ({z_xy_ratio*100:.1f}%)")
    
    if z_range < 0.05:
        print(f"  警告: Z轴范围很小 ({z_range:.4f}m < 5cm)，物体可能确实很薄")
    elif z_xy_ratio < 0.3:
        print(f"  提示: Z轴变化相对于XY轴较小 ({z_xy_ratio*100:.1f}%)，物体可能比较扁平")
        print(f"        请查看Z轴放大视图（左下）和侧面视图（右下）来观察Z轴的变化")
    else:
        print(f"  点云在Z轴方向有较好的变化 ({z_xy_ratio*100:.1f}%)")

def visualize_3d_pointcloud_open3d(points_world: np.ndarray, title: str = "3D Point Cloud"):
    """
    使用open3d可视化3D点云（交互式，支持旋转、缩放等）。
    
    参数:
        points_world: (N, 3) 世界坐标系下的3D点云
        title: 窗口标题
    """
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    
    # 使用Z值作为颜色
    z_values = points_world[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-8)
    
    # 使用colormap（viridis）生成颜色
    colors = plt.cm.viridis(z_normalized)[:, :3]  # 只取RGB，不要alpha
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 计算点云范围
    x_range = points_world[:, 0].max() - points_world[:, 0].min()
    y_range = points_world[:, 1].max() - points_world[:, 1].min()
    z_range = points_world[:, 2].max() - points_world[:, 2].min()
    
    # 可视化
    print(f"显示3D点云（共{len(points_world)}个点）")
    print("提示: 鼠标左键旋转，中键平移，滚轮缩放，按Q或关闭窗口退出")
    
    # 如果Z轴范围很小，提示用户
    if z_range / max(x_range, y_range) < 0.1:
        print(f"\n提示: Z轴范围 ({z_range:.4f}m) 相对于XY轴很小，物体可能很薄。")
        print(f"      在open3d窗口中，可以旋转视图来观察Z轴的变化。")
    
    # 创建可视化窗口，设置初始视角
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd)
    
    # 设置视角以便更好地观察
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.rotate(300.0, 150.0)  # 设置初始旋转角度
    
    vis.run()
    vis.destroy_window()

def visualize_3d_pointcloud(points_world: np.ndarray, title: str = "3D Point Cloud", 
                             use_open3d: bool = False):
    """
    可视化3D点云（可选择使用matplotlib或open3d）。
    
    参数:
        points_world: (N, 3) 世界坐标系下的3D点云
        title: 图/窗口标题
        use_open3d: 如果为True，使用open3d（交互式更好）；否则使用matplotlib
    """
    if use_open3d:
        visualize_3d_pointcloud_open3d(points_world, title)
    else:
        visualize_3d_pointcloud_matplotlib(points_world, title)

def main():
    parser = argparse.ArgumentParser(description='可视化mask和3D点云')
    parser.add_argument('--load_path', type=str, required=True, help='h5文件路径')
    parser.add_argument('--frame_ids', type=int, nargs='+', required=True, help='帧ID列表，例如: --frame_ids 0 1 2')
    parser.add_argument('--object_id', type=int, default=118, help='要可视化的物体ID（默认118）')
    parser.add_argument('--view_type', type=str, default='agentview', 
                       choices=['agentview', 'eye_in_hand'], 
                       help='使用的视图类型（默认agentview）')
    parser.add_argument('--show_3d', action='store_true', default=True, help='是否显示3D点云（默认True）')
    parser.add_argument('--use_open3d', action='store_true', 
                       help='使用open3d进行交互式3D可视化（需要安装open3d: pip install open3d）')
    args = parser.parse_args()
    
    background_value = [5, 84, 20, 24, 25, 53, 29, 31, 39, 45, 43, 63, 48, 49, 51, 53, 57, 58, 47, 65, 66, 71, 72, 70, 73, 75, 78]
    viewer = MaskExtractor(background_value=background_value)
    load_path = args.load_path
    
    # 收集所有帧的3D实例
    instances_3d = []
    
    for frame_id in args.frame_ids:
        print(f"\n处理帧 {frame_id}...")
        viewer.load(load_path, frame_id)
        viewer.extract(binary=True, view_type=args.view_type)
        
        # 获取2D instance
        obj_2d = viewer.get_object_by_id(args.object_id)
        if obj_2d is None:
            print(f"警告: 帧 {frame_id} 中未找到object_id为{args.object_id}的物体")
            print(f"可用的物体ID: {[obj.object_id for obj in viewer.objects]}")
            continue
        
        # 创建3D instance
        obj_3d = ThreeDInstance(
            object_id=obj_2d.object_id,
            original_mask=obj_2d.original_mask,
            binary_mask=obj_2d.binary_mask,
            depth_agentview=obj_2d.depth_agentview,
            depth_eye_in_hand=obj_2d.depth_eye_in_hand,
            depth_agentview_full=viewer.depth_agentview,  # 完整的深度图
            depth_eye_in_hand_full=viewer.depth_eye_in_hand,
            ee_pos=viewer.ee_pos,
            ee_ori=viewer.ee_ori
        )
        instances_3d.append(obj_3d)
    
    # 合并多帧点云
    if len(instances_3d) == 0:
        print(f"错误: 没有找到任何帧中包含object_id为{args.object_id}的物体")
        return
    
    if args.show_3d:
        print(f"\n正在合并 {len(instances_3d)} 帧的点云...")
        print(f"使用视图类型: {args.view_type}")
        
        try:
            # 合并多帧点云
            points_world = merge_multi_frame_pointclouds(
                instances=instances_3d,
                view_type=args.view_type,
                depth_unit='m',
                depth_scale=1.0,
                convert_opengl_depth=True,
                opengl_near=0.001,
                opengl_far=50.0,
                denoise=True,
                denoise_method='combined',  # 使用半径滤波，更适合删除游离点
                denoise_params={"z_std_ratio": 1.5, "nb_neighbors": 20, "std_ratio": 2.0}  # 更严格的参数：半径0.02米，至少10个邻居
            )
            
            if len(points_world) == 0:
                print("错误: 合并后的点云为空")
                return
            
            print(f"\n合并后的3D点云形状: {points_world.shape}")
            
            # 计算合并后的3D包围盒
            xyz_min = points_world.min(axis=0)
            xyz_max = points_world.max(axis=0)
            print(f"\n合并后的3D包围盒（单位：米）:")
            print(f"  min: {xyz_min}")
            print(f"  max: {xyz_max}")
            print(f"  尺寸: {xyz_max - xyz_min} m")
            
            # 可视化合并后的点云
            visualize_3d_pointcloud(
                points_world, 
                title=f"合并点云 - Object ID {args.object_id} ({args.view_type}, {len(instances_3d)}帧)",
                use_open3d=args.use_open3d
            )
        except Exception as e:
            print(f"错误: 无法合并点云: {e}")
            import traceback
            traceback.print_exc()



if __name__ == '__main__':
    main()