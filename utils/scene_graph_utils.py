#!/usr/bin/env python3

from re import X
import h5py
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union, Dict
import os
import json
import trimesh
import matplotlib.pyplot as plt
MATPLOTLIB_AVAILABLE = True
OPEN3D_AVAILABLE = True
import open3d as o3d

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class MergedThreeDInstance:
    
    def __init__(self, instance: str, merged_points: np.ndarray):
        self.instance = instance
        self.merged_points = merged_points
        self.merged_bbox = np.array([merged_points.min(axis=0), merged_points.max(axis=0)])
        self.merged_mesh = trimesh.Trimesh(vertices=merged_points, faces=None)



class SceneGraphAnalyzer:
    """
    场景图分析器，用于分析多个3D实例之间的空间关系。
    """
    
    def __init__(self, instances: List[MergedThreeDInstance]):
        """
        初始化场景图分析器。
        
        参数:
            instances: ThreeDInstance对象列表
        """
        self.instances = instances
        self._world_points_cache = {}  # 缓存每个实例的世界坐标点云
        self._world_bbox_cache = {}    # 缓存每个实例的世界坐标包围盒
        self._watertight_mesh_cache = {}  # 缓存每个实例的watertight mesh

    def _get_world_bbox(self, instance: MergedThreeDInstance):
        xyz_min, xyz_max = instance.merged_points.min(axis=0), instance.merged_points.max(axis=0)
        return xyz_min, xyz_max
    
    def  get_watertight_mesh(self, instance: MergedThreeDInstance,
                             view_type: str = 'agentview',
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
    
    def is_on(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
              z_threshold: float = 0.03) -> bool:
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
        bbox1_min, bbox1_max = self._get_world_bbox(obj1)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2)
        
        # 检查Z轴关系：obj1的底部应该高于obj2的顶部
        obj1_bottom = bbox1_min[2]
        obj2_top = bbox2_max[2]
        
        
        if obj1_bottom > obj2_top - z_threshold and obj1_bottom < obj2_top + z_threshold:
            center1_x = (bbox1_min[0] + bbox1_max[0]) / 2
            center1_y = (bbox1_min[1] + bbox1_max[1]) / 2

            if center1_x > bbox2_min[0] and center1_x < bbox2_max[0] and center1_y > bbox2_min[1] and center1_y < bbox2_max[1]:
                return True
            else:
                return False
        else:
            return False
    
    def is_in(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
              containment_ratio: float = 0.9,
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
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, **kwargs)
        
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
        points1 = obj1.merged_points
        if len(points1) == 0:
            return False
        
        # 计算obj1的点云中有多少点在obj2的包围盒内
        inside_mask = np.all((points1 >= bbox2_min) & (points1 <= bbox2_max), axis=1)
        inside_ratio = np.sum(inside_mask) / len(points1)
        
        return inside_ratio >= containment_ratio
    
    def is_under(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
                 z_threshold: float = 0.03
                 ) -> bool:
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
        bbox1_min, bbox1_max = self._get_world_bbox(obj1)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2)
        
        # 检查Z轴关系：obj1的底部应该高于obj2的顶部
        obj1_top = bbox1_max[2]
        obj2_bottom = bbox2_min[2]
        
        
        if obj1_top < obj2_bottom + z_threshold and obj1_top > obj2_bottom - z_threshold:
            center1_x = (bbox1_min[0] + bbox1_max[0]) / 2
            center1_y = (bbox1_min[1] + bbox1_max[1]) / 2
            if bbox1_max[0] < bbox2_min[0] or bbox1_min[0] > bbox2_max[0] or bbox1_max[1] < bbox2_min[1] or bbox1_min[1] > bbox2_max[1]:
                return False
            else:
                return True
        else:
            return False
    
    def is_left(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
                overlap_ratio: float = 0.5,
                **kwargs) -> bool:
        """
        判断obj1是否在obj2的左侧（left关系）。
        
        条件：
        1. obj1的Y坐标（中心或最大Y值）小于obj2的Y坐标（中心或最小Y值）
        2. obj1和obj2在Z轴方向有重叠
        3. obj1和obj2在X轴方向的距离在合理范围内（可选）
        
        参数:
            obj1: 左侧的物体
            obj2: 右侧的物体
            z_overlap_ratio: Z轴重叠比例阈值，默认0.5
            x_distance_threshold: X轴方向最大距离阈值（米），默认0.2
            **kwargs: 传递给_get_world_bbox的其他参数
        
        返回:
            True如果obj1在obj2左侧（Y轴负方向）
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, **kwargs)
        
        # 检查Z轴重叠
        z_overlap = max(0, min(bbox1_max[2], bbox2_max[2]) - max(bbox1_min[2], bbox2_min[2]))
        z_range1 = bbox1_max[2] - bbox1_min[2]
        z_range2 = bbox2_max[2] - bbox2_min[2]
        z_overlap_ratio_actual = z_overlap / max(z_range1, z_range2) if max(z_range1, z_range2) > 0 else 0
        
        if z_overlap_ratio_actual < overlap_ratio:
            return False
        
        # 检查Y轴关系：obj1应该在obj2的左侧（Y轴负方向）
        # 使用中心点或边界来判断
        center1_y = (bbox1_min[1] + bbox1_max[1]) / 2
        center2_y = (bbox2_min[1] + bbox2_max[1]) / 2
        
        if center1_y >= center2_y:
            return False
       
        x_overlap = max(0, min(bbox1_max[0], bbox2_max[0]) - max(bbox1_min[0], bbox2_min[0]))
        x_range1 = bbox1_max[0] - bbox1_min[0]
        x_range2 = bbox2_max[0] - bbox2_min[0]
        x_overlap_ratio_actual = x_overlap / max(x_range1, x_range2) if max(x_range1, x_range2) > 0 else 0
        
        if x_overlap_ratio_actual < overlap_ratio:
            return False        


        else:
            return True
    
    def is_right(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
                 overlap_ratio: float = 0.5,
                 **kwargs) -> bool:
        """
        判断obj1是否在obj2的右侧（right关系）。
        
        条件：
        1. obj1的Y坐标（中心或最小Y值）大于obj2的Y坐标（中心或最大Y值）
        2. obj1和obj2在Z轴方向有重叠
        3. obj1和obj2在X轴方向的距离在合理范围内（可选）
        
        参数:
            obj1: 右侧的物体
            obj2: 左侧的物体
            overlap_ratio: X轴重叠比例阈值，默认0.5
            **kwargs: 传递给_get_world_bbox的其他参数
        
        返回:
            True如果obj1在obj2右侧（Y轴正方向）
        """
        # right关系就是left关系的反向
        return self.is_left(obj2, obj1, overlap_ratio, **kwargs)
    
    def is_behind(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
                  overlap_ratio: float = 0.5,
                  **kwargs) -> bool:
        """
        判断obj1是否在obj2的后面（behind关系）。
        
        条件：
        1. obj1的X坐标（中心）小于obj2的X坐标（中心）
        2. obj1和obj2在Z轴方向有重叠
        3. obj1和obj2在Y轴方向的距离在合理范围内（可选）
        
        参数:
            obj1: 后面的物体
            obj2: 前面的物体
            overlap_ratio: X轴重叠比例阈值，默认0.5
            **kwargs: 传递给_get_world_bbox的其他参数
        
        返回:
            True如果obj1在obj2后面（X轴负方向）
        """
        bbox1_min, bbox1_max = self._get_world_bbox(obj1, **kwargs)
        bbox2_min, bbox2_max = self._get_world_bbox(obj2, **kwargs)
        
        # 检查Z轴重叠
        z_overlap = max(0, min(bbox1_max[2], bbox2_max[2]) - max(bbox1_min[2], bbox2_min[2]))
        z_range1 = bbox1_max[2] - bbox1_min[2]
        z_range2 = bbox2_max[2] - bbox2_min[2]
        z_overlap_ratio_actual = z_overlap / max(z_range1, z_range2) if max(z_range1, z_range2) > 0 else 0
        
        if z_overlap_ratio_actual < overlap_ratio:
            return False
        
        # 检查X轴关系：obj1应该在obj2的后面（X轴负方向）
        center1_x = (bbox1_min[0] + bbox1_max[0]) / 2
        center2_x = (bbox2_min[0] + bbox2_max[0]) / 2
        
        if center1_x >= center2_x:
            return False
        
        # 可选：检查Y轴距离是否在合理范围内
        y_overlap = max(0, min(bbox1_max[1], bbox2_max[1]) - max(bbox1_min[1], bbox2_min[1]))
        y_range1 = bbox1_max[1] - bbox1_min[1]
        y_range2 = bbox2_max[1] - bbox2_min[1]
        y_overlap_ratio_actual = y_overlap / max(y_range1, y_range2) if max(y_range1, y_range2) > 0 else 0
        
        if y_overlap_ratio_actual < overlap_ratio:
            return False
        else:
            return True

    def is_in_front_of(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
                       overlap_ratio: float = 0.5,
                       **kwargs) -> bool:
        """
        判断obj1是否在obj2的前面（in_front_of关系）。
        
        条件：
        1. obj1的X坐标（中心）大于obj2的X坐标（中心）
        2. obj1和obj2在Z轴方向有重叠
        3. obj1和obj2在Y轴方向的距离在合理范围内（可选）
        
        参数:
            obj1: 前面的物体
            obj2: 后面的物体
            overlap_ratio: X轴重叠比例阈值，默认0.5
            **kwargs: 传递给_get_world_bbox的其他参数
        
        返回:
            True如果obj1在obj2前面（X轴正方向）
        """
        return self.is_behind(obj2, obj1, overlap_ratio, **kwargs)


    
    def get_spatial_relations(self, obj1: MergedThreeDInstance, obj2: MergedThreeDInstance,
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
            'on': self.is_on(obj1, obj2, **kwargs),
            'in': self.is_in(obj1, obj2, **kwargs),
            # 'under': self.is_under(obj1, obj2, **kwargs),
            'left': self.is_left(obj1, obj2, **kwargs),
            'right': self.is_right(obj1, obj2, **kwargs),
            'behind': self.is_behind(obj1, obj2, **kwargs),
            'in_front_of': self.is_in_front_of(obj1, obj2, **kwargs),
        }
        return relations
    
    def build_scene_graph(self) -> Dict[Tuple[int, int], Dict[str, bool]]:
        """
        构建场景图，包含所有实例对之间的空间关系。
        
        参数:
        
        返回:
            字典，键为(obj1_id, obj2_id)元组，值为关系字典
        """
        scene_graph = {}
        
        for i, obj1 in enumerate(self.instances):
            for j, obj2 in enumerate(self.instances):
                if i != j:  # 不比较自己
                    relations = self.get_spatial_relations(obj1, obj2)
                    scene_graph[(obj1.instance, obj2.instance)] = relations
        
        return scene_graph
    
    def get_relations_for_object(self, obj: MergedThreeDInstance,
                                ) -> Dict[int, Dict[str, bool]]:
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
                relations = self.get_spatial_relations(obj, other_obj)
                relations_dict[other_obj.object_id] = relations
        
        return relations_dict
    
    def print_scene_graph(self):
        """
        打印场景图（JSON格式）。
        
        参数:
        """
        scene_graph = self.build_scene_graph()
        
        # 收集所有出现过的对象ID
        object_ids = set()
        relations_list = []
        
        for (obj1_id, obj2_id), relations in scene_graph.items():
            # 只处理为True的关系
            for rel_type, is_true in relations.items():
                if is_true:
                    object_ids.add(obj1_id)
                    object_ids.add(obj2_id)
                    relations_list.append({
                        "type": rel_type,
                        "subject": obj1_id,
                        "object": obj2_id
                    })
        
        # 构建JSON结构
        result = {
            "objects": [{"id": obj_id} for obj_id in sorted(object_ids)],
            "relations": relations_list
        }
        
        # 打印JSON格式
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    def visualize_scene_matplotlib(self,
                                   show_bbox: bool = True,
                                   show_relations: bool = True,
                                   show_points: bool = True,
                                   point_size: float = 1.0):
        """
        使用matplotlib可视化场景（所有实例的点云、包围盒和空间关系）。
        
        参数:
            show_bbox: 是否显示包围盒，默认True
            show_relations: 是否显示空间关系（箭头），默认True
            show_points: 是否显示点云，默认True
            point_size: 点云大小，默认1.0
        """
        if not MATPLOTLIB_AVAILABLE:
            print("错误: matplotlib未安装，无法进行可视化")
            return
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取场景图
        scene_graph = self.build_scene_graph()
        
        # 为每个实例分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.instances)))
        obj_id_to_color = {obj.object_id: colors[i] for i, obj in enumerate(self.instances)}
        
        all_points = []
        all_bboxes = []
        
        # 收集所有点云和包围盒
        for instance in self.instances:
            points = self._get_world_points(instance)
            if len(points) > 0:
                all_points.append((instance.object_id, points))
            
            if show_bbox:
                bbox_min, bbox_max = self._get_world_bbox(instance)
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
                        bbox1_min, bbox1_max = self._get_world_bbox(obj1)
                        bbox2_min, bbox2_max = self._get_world_bbox(obj2)
                        
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
        ax.set_title('场景图可视化')
        
        if show_points and len(all_points) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_scene_open3d(self,
                              show_bbox: bool = True,
                              show_relations: bool = True,
                              show_points: bool = True):
        """
        使用open3d可视化场景（所有实例的点云、包围盒和空间关系）。
        
        参数:
            show_bbox: 是否显示包围盒，默认True
            show_relations: 是否显示空间关系（线条），默认True
            show_points: 是否显示点云，默认True
        """
        if not OPEN3D_AVAILABLE:
            print("错误: open3d未安装，无法进行可视化")
            return
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="场景图可视化",
                         width=1200, height=800)
        
        # 获取场景图
        scene_graph = self.build_scene_graph()
        
        # 为每个实例分配颜色
        colors = np.linspace(0, 1, len(self.instances))
        obj_id_to_color_idx = {obj.object_id: i for i, obj in enumerate(self.instances)}
        
        geometries = []
        
        # 收集所有点云和包围盒
        for instance in self.instances:
            points = self._get_world_points(instance)
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
                bbox_min, bbox_max = self._get_world_bbox(instance)
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
                        bbox1_min, bbox1_max = self._get_world_bbox(obj1)
                        bbox2_min, bbox2_max = self._get_world_bbox(obj2)
                        
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
    
    def visualize_scene(self):
        """
        可视化场景（统一接口）。
        
        参数:
        """
        self.visualize_scene_open3d()
        self.visualize_scene_matplotlib()
    
    def _draw_bbox_matplotlib(self, ax: Axes3D, bbox_min: np.ndarray, bbox_max: np.ndarray,
                             color: Tuple[float, float, float], alpha: float = 0.3):
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
    
    def _draw_arrow_matplotlib(self, ax: Axes3D, start: np.ndarray, end: np.ndarray, label: str = ""):
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
        vertices = np.array([
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
        line_set.points = o3d.utility.Vector3dVector(vertices)
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
