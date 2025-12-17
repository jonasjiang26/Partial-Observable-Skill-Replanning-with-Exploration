#!/usr/bin/env python3

import h5py
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union, Dict
import os
import trimesh
from utils.depth_projection_utils import ThreeDInstance
import matplotlib.pyplot as plt
MATPLOTLIB_AVAILABLE = True
OPEN3D_AVAILABLE = True
import open3d as o3d

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
class SceneGraphAnalyzer:
    """
    场景图分析器，用于分析多个3D实例之间的空间关系。
    """
    
    def __init__(self, instances: List[ThreeDInstance]):
        """
        初始化场景图分析器。
        
        参数:
            instances: ThreeDInstance对象列表
        """
        self.instances = instances
        self._world_points_cache = {}  # 缓存每个实例的世界坐标点云
        self._world_bbox_cache = {}    # 缓存每个实例的世界坐标包围盒
        self._watertight_mesh_cache = {}  # 缓存每个实例的watertight mesh
    
    def _get_world_points(self, instance: ThreeDInstance, view_type: str = 'agentview',
                         **kwargs) -> np.ndarray:
        """
        获取实例的世界坐标点云（带缓存）
        
        参数:
            instance: ThreeDInstance对象
            view_type: 视图类型
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            世界坐标系下的点云 (N, 3) numpy数组
        """
        cache_key = (id(instance), view_type, str(kwargs))
        if cache_key not in self._world_points_cache:
            points = instance.project_to_world_coordinates(view_type=view_type, **kwargs)
            self._world_points_cache[cache_key] = points
            # 同时缓存mesh（如果trimesh可用）
            if trimesh is not None:
                mesh = self.get_watertight_mesh_from_points(points)
                self._watertight_mesh_cache[cache_key] = mesh
        return self._world_points_cache[cache_key]
    
    def _get_world_bbox(self, instance: ThreeDInstance, view_type: str = 'agentview',
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """获取实例的世界坐标包围盒（带缓存）"""
        cache_key = (id(instance), view_type, str(kwargs))
        if cache_key not in self._world_bbox_cache:
            bbox = instance.get_world_bbox3d(view_type=view_type, **kwargs)
            self._world_bbox_cache[cache_key] = bbox
        return self._world_bbox_cache[cache_key]

    def  get_watertight_mesh(self, instance: ThreeDInstance, view_type: str = 'agentview',
                             **kwargs) -> Optional['trimesh.Trimesh']:
        """
        获取实例的watertight mesh（带缓存）
        
        参数:
            instance: ThreeDInstance对象
            view_type: 视图类型
            **kwargs: 传递给instance.get_watertight_mesh的其他参数
        
        返回:
            trimesh.Trimesh对象，如果trimesh未安装或instance没有get_watertight_mesh方法则返回None
        """
        if trimesh is None:
            return None
        
        cache_key = (id(instance), view_type, str(kwargs))
        if cache_key not in self._watertight_mesh_cache:
            # 检查instance是否有get_watertight_mesh方法
            if hasattr(instance, 'get_watertight_mesh'):
                mesh = instance.get_watertight_mesh(view_type=view_type, **kwargs)
            else:
                # 如果没有，从点云创建
                points = self._get_world_points(instance, view_type, **kwargs)
                mesh = self.get_watertight_mesh_from_points(points)
            self._watertight_mesh_cache[cache_key] = mesh
        return self._watertight_mesh_cache[cache_key]

    def get_watertight_mesh_from_points(self, points: np.ndarray) -> Optional['trimesh.Trimesh']:
        """
        从点云获取watertight mesh
        
        参数:
            points: 点云 (N, 3) numpy数组
        
        返回:
            trimesh.Trimesh对象，如果trimesh未安装则返回None
        """
        if trimesh is None:
            print("错误: trimesh未安装，无法创建mesh")
            return None
        
        if len(points) == 0:
            print("警告: 点云为空，无法创建mesh")
            return None
        
        # 创建只包含顶点的mesh（faces=None表示没有面片信息）
        mesh = trimesh.Trimesh(vertices=points, faces=None)
        return mesh
    
    def is_on(self, obj1: ThreeDInstance, obj2: ThreeDInstance, 
              view_type: str = 'agentview',
              z_threshold: float = 0.01,
              overlap_ratio: float = 0.3,
              **kwargs) -> bool:
        """
        判断obj1是否在obj2上方（on关系）。
        
        条件：
        1. obj1的Z坐标（底部）高于obj2的Z坐标（顶部）
        2. obj1和obj2在XY平面的投影有重叠
        3. Z轴距离在合理范围内（考虑z_threshold）
        
        参数:
            obj1: 上方的物体
            obj2: 下方的物体
            view_type: 使用的视图类型
            z_threshold: Z轴距离阈值（米），obj1底部应该高于obj2顶部至少这个距离
            overlap_ratio: XY平面重叠比例阈值，默认0.3（30%）
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2上方
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
        
        # 检查Z轴关系：obj1的底部应该高于obj2的顶部
        obj1_bottom = bbox1_min[2]
        obj2_top = bbox2_max[2]
        
        if obj1_bottom < obj2_top + z_threshold:
            return False
        
        # 检查XY平面投影是否重叠
        # obj1在XY平面的投影
        obj1_x_min, obj1_x_max = bbox1_min[0], bbox1_max[0]
        obj1_y_min, obj1_y_max = bbox1_min[1], bbox1_max[1]
        
        # obj2在XY平面的投影
        obj2_x_min, obj2_x_max = bbox2_min[0], bbox2_max[0]
        obj2_y_min, obj2_y_max = bbox2_min[1], bbox2_max[1]
        
        # 计算重叠区域
        overlap_x = max(0, min(obj1_x_max, obj2_x_max) - max(obj1_x_min, obj2_x_min))
        overlap_y = max(0, min(obj1_y_max, obj2_y_max) - max(obj1_y_min, obj2_y_min))
        overlap_area = overlap_x * overlap_y
        
        # 计算obj1的投影面积
        obj1_area = (obj1_x_max - obj1_x_min) * (obj1_y_max - obj1_y_min)
        
        # 如果重叠面积占obj1投影面积的比例超过阈值，认为有重叠
        if obj1_area > 0:
            overlap_ratio_actual = overlap_area / obj1_area
            return overlap_ratio_actual >= overlap_ratio
        
        return False
    
    def is_in(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
              view_type: str = 'agentview',
              containment_ratio: float = 0.8,
              **kwargs) -> bool:
        """
        判断obj1是否在obj2内部（in关系）。
        
        条件：
        1. obj1的包围盒大部分在obj2的包围盒内部
        2. 或者obj1的点云大部分在obj2的包围盒内部
        
        参数:
            obj1: 内部的物体
            obj2: 外部的物体（容器）
            view_type: 使用的视图类型
            containment_ratio: 包含比例阈值，默认0.8（80%的点或体积需要在内部）
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2内部
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
        
        # 方法1：基于包围盒的包含关系
        # 检查obj1的包围盒是否大部分在obj2的包围盒内
        # 计算obj1包围盒在obj2内的体积比例
        
        # 计算重叠区域
        overlap_min = np.maximum(bbox1_min, bbox2_min)
        overlap_max = np.minimum(bbox1_max, bbox2_max)
        
        # 检查是否有重叠
        if np.any(overlap_max < overlap_min):
            return False
        
        # 计算重叠体积
        overlap_volume = np.prod(overlap_max - overlap_min)
        obj1_volume = np.prod(bbox1_max - bbox1_min)
        
        if obj1_volume > 0:
            volume_ratio = overlap_volume / obj1_volume
            if volume_ratio >= containment_ratio:
                return True
        
        # 方法2：基于点云的包含关系（更准确）
        points1 = self._get_world_points(obj1, view_type, **kwargs)
        if len(points1) == 0:
            return False
        
        # 计算obj1的点云中有多少点在obj2的包围盒内
        inside_mask = np.all((points1 >= bbox2_min) & (points1 <= bbox2_max), axis=1)
        inside_ratio = np.sum(inside_mask) / len(points1)
        
        return inside_ratio >= containment_ratio
    
    def is_under(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
                 view_type: str = 'agentview',
                 z_threshold: float = 0.01,
                 overlap_ratio: float = 0.3,
                 **kwargs) -> bool:
        """
        判断obj1是否在obj2下方（under关系）。
        
        参数:
            obj1: 下方的物体
            obj2: 上方的物体
            view_type: 使用的视图类型
            z_threshold: Z轴距离阈值（米）
            overlap_ratio: XY平面重叠比例阈值
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2下方
        """
        # under关系就是on关系的反向
        return self.is_on(obj2, obj1, view_type, z_threshold, overlap_ratio, **kwargs)
    
    def is_above(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
                 view_type: str = 'agentview',
                 z_threshold: float = 0.05,
                 **kwargs) -> bool:
        """
        判断obj1是否在obj2上方（above关系，不要求接触）。
        
        参数:
            obj1: 上方的物体
            obj2: 下方的物体
            view_type: 使用的视图类型
            z_threshold: Z轴最小距离阈值（米）
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2上方
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
        
        # obj1的底部应该高于obj2的顶部
        obj1_bottom = bbox1_min[2]
        obj2_top = bbox2_max[2]
        
        return obj1_bottom > obj2_top + z_threshold
    
    def is_below(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
                 view_type: str = 'agentview',
                 z_threshold: float = 0.05,
                 **kwargs) -> bool:
        """
        判断obj1是否在obj2下方（below关系，不要求接触）。
        
        参数:
            obj1: 下方的物体
            obj2: 上方的物体
            view_type: 使用的视图类型
            z_threshold: Z轴最小距离阈值（米）
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2下方
        """
        return self.is_above(obj2, obj1, view_type, z_threshold, **kwargs)
    
    def is_beside(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
                  view_type: str = 'agentview',
                  distance_threshold: float = 0.1,
                  z_overlap_ratio: float = 0.5,
                  **kwargs) -> bool:
        """
        判断obj1是否在obj2旁边（beside关系）。
        
        条件：
        1. obj1和obj2在Z轴方向有重叠
        2. obj1和obj2在XY平面的距离在阈值内
        
        参数:
            obj1: 物体1
            obj2: 物体2
            view_type: 使用的视图类型
            distance_threshold: XY平面最大距离阈值（米）
            z_overlap_ratio: Z轴重叠比例阈值
            **kwargs: 传递给project_to_world_coordinates的其他参数
        
        返回:
            True如果obj1在obj2旁边
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
        
        # 检查Z轴重叠
        z_overlap = max(0, min(bbox1_max[2], bbox2_max[2]) - max(bbox1_min[2], bbox2_min[2]))
        z_range1 = bbox1_max[2] - bbox1_min[2]
        z_range2 = bbox2_max[2] - bbox2_min[2]
        z_overlap_ratio_actual = z_overlap / max(z_range1, z_range2) if max(z_range1, z_range2) > 0 else 0
        
        if z_overlap_ratio_actual < z_overlap_ratio:
            return False
        
        # 计算XY平面的中心点距离
        center1 = (bbox1_min[:2] + bbox1_max[:2]) / 2
        center2 = (bbox2_min[:2] + bbox2_max[:2]) / 2
        distance = np.linalg.norm(center1 - center2)
        
        return distance <= distance_threshold
    
    def get_spatial_relations(self, obj1: ThreeDInstance, obj2: ThreeDInstance,
                              view_type: str = 'agentview',
                              **kwargs) -> Dict[str, bool]:
        """
        获取obj1和obj2之间的所有空间关系。
        
        参数:
            obj1: 物体1
            obj2: 物体2
            view_type: 使用的视图类型
            **kwargs: 传递给关系判断方法的其他参数
        
        返回:
            包含各种关系判断结果的字典
        """
        relations = {
            'on': self.is_on(obj1, obj2, view_type, **kwargs),
            'in': self.is_in(obj1, obj2, view_type, **kwargs),
            'under': self.is_under(obj1, obj2, view_type, **kwargs),
            'above': self.is_above(obj1, obj2, view_type, **kwargs),
            'below': self.is_below(obj1, obj2, view_type, **kwargs),
            'beside': self.is_beside(obj1, obj2, view_type, **kwargs),
        }
        return relations
    
    def build_scene_graph(self, view_type: str = 'agentview',
                         **kwargs) -> Dict[Tuple[int, int], Dict[str, bool]]:
        """
        构建场景图，包含所有实例对之间的空间关系。
        
        参数:
            view_type: 使用的视图类型
            **kwargs: 传递给关系判断方法的其他参数
        
        返回:
            字典，键为(obj1_id, obj2_id)元组，值为关系字典
        """
        scene_graph = {}
        
        for i, obj1 in enumerate(self.instances):
            for j, obj2 in enumerate(self.instances):
                if i != j:  # 不比较自己
                    relations = self.get_spatial_relations(obj1, obj2, view_type, **kwargs)
                    scene_graph[(obj1.object_id, obj2.object_id)] = relations
        
        return scene_graph
    
    def get_relations_for_object(self, obj: ThreeDInstance,
                                view_type: str = 'agentview',
                                **kwargs) -> Dict[int, Dict[str, bool]]:
        """
        获取指定物体与其他所有物体的空间关系。
        
        参数:
            obj: 目标物体
            view_type: 使用的视图类型
            **kwargs: 传递给关系判断方法的其他参数
        
        返回:
            字典，键为其他物体的object_id，值为关系字典
        """
        relations_dict = {}
        
        for other_obj in self.instances:
            if other_obj.object_id != obj.object_id:
                relations = self.get_spatial_relations(obj, other_obj, view_type, **kwargs)
                relations_dict[other_obj.object_id] = relations
        
        return relations_dict
    
    def print_scene_graph(self, view_type: str = 'agentview', **kwargs):
        """
        打印场景图（人类可读格式）。
        
        参数:
            view_type: 使用的视图类型
            **kwargs: 传递给关系判断方法的其他参数
        """
        scene_graph = self.build_scene_graph(view_type, **kwargs)
        
        print(f"\n场景图（视图类型: {view_type}）:")
        print("=" * 60)
        
        for (obj1_id, obj2_id), relations in scene_graph.items():
            # 只打印为True的关系
            true_relations = [rel for rel, value in relations.items() if value]
            if true_relations:
                print(f"物体 {obj1_id} -> 物体 {obj2_id}: {', '.join(true_relations)}")
        
        print("=" * 60)
    
    def visualize_scene_matplotlib(self, view_type: str = 'agentview',
                                   show_bbox: bool = True,
                                   show_relations: bool = True,
                                   show_points: bool = True,
                                   point_size: float = 1.0,
                                   **kwargs):
        """
        使用matplotlib可视化场景（所有实例的点云、包围盒和空间关系）。
        
        参数:
            view_type: 使用的视图类型
            show_bbox: 是否显示包围盒，默认True
            show_relations: 是否显示空间关系（箭头），默认True
            show_points: 是否显示点云，默认True
            point_size: 点云大小，默认1.0
            **kwargs: 传递给project_to_world_coordinates的其他参数
        """
        if not MATPLOTLIB_AVAILABLE:
            print("错误: matplotlib未安装，无法进行可视化")
            return
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取场景图
        scene_graph = self.build_scene_graph(view_type, **kwargs)
        
        # 为每个实例分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.instances)))
        obj_id_to_color = {obj.object_id: colors[i] for i, obj in enumerate(self.instances)}
        
        all_points = []
        all_bboxes = []
        
        # 收集所有点云和包围盒
        for instance in self.instances:
            points = self._get_world_points(instance, view_type, **kwargs)
            if len(points) > 0:
                all_points.append((instance.object_id, points))
            
            if show_bbox:
                bbox_min, bbox_max = self._get_world_bbox(instance, view_type, **kwargs)
                all_bboxes.append((instance.object_id, bbox_min, bbox_max))
        
        # 绘制点云
        if show_points:
            for obj_id, points in all_points:
                color = obj_id_to_color[obj_id]
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=[color], s=point_size, alpha=0.6, label=f'Object {obj_id}')
        
        # 绘制包围盒
        if show_bbox:
            for obj_id, bbox_min, bbox_max in all_bboxes:
                color = obj_id_to_color[obj_id]
                self._draw_bbox_matplotlib(ax, bbox_min, bbox_max, color, alpha=0.3)
        
        # 绘制空间关系
        if show_relations:
            for (obj1_id, obj2_id), relations in scene_graph.items():
                true_relations = [rel for rel, value in relations.items() if value]
                if true_relations:
                    # 找到对应的实例
                    obj1 = next((obj for obj in self.instances if obj.object_id == obj1_id), None)
                    obj2 = next((obj for obj in self.instances if obj.object_id == obj2_id), None)
                    
                    if obj1 and obj2:
                        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
                        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
                        
                        # 计算中心点
                        center1 = (bbox1_min + bbox1_max) / 2
                        center2 = (bbox2_min + bbox2_max) / 2
                        
                        # 绘制箭头
                        self._draw_arrow_matplotlib(ax, center1, center2, 
                                                   ', '.join(true_relations))
        
        # 设置标签和图例
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'场景图可视化 (视图类型: {view_type})')
        
        if show_points and len(all_points) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_scene_open3d(self, view_type: str = 'agentview',
                              show_bbox: bool = True,
                              show_relations: bool = True,
                              show_points: bool = True,
                              **kwargs):
        """
        使用open3d可视化场景（所有实例的点云、包围盒和空间关系）。
        
        参数:
            view_type: 使用的视图类型
            show_bbox: 是否显示包围盒，默认True
            show_relations: 是否显示空间关系（线条），默认True
            show_points: 是否显示点云，默认True
            **kwargs: 传递给project_to_world_coordinates的其他参数
        """
        if not OPEN3D_AVAILABLE:
            print("错误: open3d未安装，无法进行可视化")
            return
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"场景图可视化 (视图类型: {view_type})", 
                         width=1200, height=800)
        
        # 获取场景图
        scene_graph = self.build_scene_graph(view_type, **kwargs)
        
        # 为每个实例分配颜色
        colors = np.linspace(0, 1, len(self.instances))
        obj_id_to_color_idx = {obj.object_id: i for i, obj in enumerate(self.instances)}
        
        geometries = []
        
        # 收集所有点云和包围盒
        for instance in self.instances:
            points = self._get_world_points(instance, view_type, **kwargs)
            if len(points) == 0:
                continue
            
            color_idx = obj_id_to_color_idx[instance.object_id]
            # 使用colormap生成颜色
            color = plt.cm.tab20(color_idx)[:3] if MATPLOTLIB_AVAILABLE else [0.5, 0.5, 0.5]
            
            # 绘制点云
            if show_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.paint_uniform_color(color)
                vis.add_geometry(pcd)
                geometries.append(pcd)
            
            # 绘制包围盒
            if show_bbox:
                bbox_min, bbox_max = self._get_world_bbox(instance, view_type, **kwargs)
                bbox = self._create_bbox_open3d(bbox_min, bbox_max, color)
                vis.add_geometry(bbox)
                geometries.append(bbox)
        
        # 绘制空间关系
        if show_relations:
            for (obj1_id, obj2_id), relations in scene_graph.items():
                true_relations = [rel for rel, value in relations.items() if value]
                if true_relations:
                    obj1 = next((obj for obj in self.instances if obj.object_id == obj1_id), None)
                    obj2 = next((obj for obj in self.instances if obj.object_id == obj2_id), None)
                    
                    if obj1 and obj2:
                        bbox1_min, bbox1_max = self._get_world_bbox(obj1, view_type, **kwargs)
                        bbox2_min, bbox2_max = self._get_world_bbox(obj2, view_type, **kwargs)
                        
                        center1 = (bbox1_min + bbox1_max) / 2
                        center2 = (bbox2_min + bbox2_max) / 2
                        
                        # 创建线条
                        line = self._create_line_open3d(center1, center2)
                        vis.add_geometry(line)
                        geometries.append(line)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.rotate(300.0, 150.0)
        
        print(f"显示场景图可视化（共{len(self.instances)}个物体）")
        print("提示: 鼠标左键旋转，中键平移，滚轮缩放，按Q或关闭窗口退出")
        
        vis.run()
        vis.destroy_window()
    
    def visualize_scene(self, view_type: str = 'agentview',
                       use_open3d: bool = False,
                       show_bbox: bool = True,
                       show_relations: bool = True,
                       show_points: bool = True,
                       **kwargs):
        """
        可视化场景（统一接口）。
        
        参数:
            view_type: 使用的视图类型
            use_open3d: 如果为True，使用open3d（交互式更好）；否则使用matplotlib
            show_bbox: 是否显示包围盒，默认True
            show_relations: 是否显示空间关系，默认True
            show_points: 是否显示点云，默认True
            **kwargs: 传递给project_to_world_coordinates的其他参数
        """
        if use_open3d:
            self.visualize_scene_open3d(view_type, show_bbox, show_relations, 
                                       show_points, **kwargs)
        else:
            self.visualize_scene_matplotlib(view_type, show_bbox, show_relations, 
                                          show_points, **kwargs)
    
    def _draw_bbox_matplotlib(self, ax, bbox_min: np.ndarray, bbox_max: np.ndarray,
                             color, alpha: float = 0.3):
        """在matplotlib中绘制3D包围盒"""
        x_min, y_min, z_min = bbox_min
        x_max, y_max, z_max = bbox_max
        
        # 定义8个顶点
        vertices = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])
        
        # 定义12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 垂直边
        ]
        
        # 绘制边
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, color=color, alpha=alpha, linewidth=1.5)
    
    def _draw_arrow_matplotlib(self, ax, start: np.ndarray, end: np.ndarray, label: str = ""):
        """在matplotlib中绘制3D箭头"""
        # 绘制箭头线
        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                 'r-', linewidth=2, alpha=0.6)
        
        # 在中间位置添加标签
        mid = (start + end) / 2
        ax.text(mid[0], mid[1], mid[2], label, fontsize=8, color='red')
    
    def _create_bbox_open3d(self, bbox_min: np.ndarray, bbox_max: np.ndarray,
                            color: Tuple[float, float, float]) -> o3d.geometry.LineSet:
        """创建open3d包围盒线框"""
        x_min, y_min, z_min = bbox_min
        x_max, y_max, z_max = bbox_max
        
        # 定义8个顶点
        points = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])
        
        # 定义12条边
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7],  # 垂直边
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        
        return line_set
    
    def _create_line_open3d(self, start: np.ndarray, end: np.ndarray) -> o3d.geometry.LineSet:
        """创建open3d线条"""
        points = np.array([start, end])
        lines = [[0, 1]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        
        return line_set
