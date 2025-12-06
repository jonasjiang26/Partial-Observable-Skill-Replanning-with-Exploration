#!/usr/bin/env python3

import h5py
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union, Dict
import os

from utils.depth_projection_utils import ThreeDInstance


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
    
    def _get_world_points(self, instance: ThreeDInstance, view_type: str = 'agentview',
                         **kwargs) -> np.ndarray:
        """获取实例的世界坐标点云（带缓存）"""
        cache_key = (id(instance), view_type, str(kwargs))
        if cache_key not in self._world_points_cache:
            points = instance.project_to_world_coordinates(view_type=view_type, **kwargs)
            self._world_points_cache[cache_key] = points
        return self._world_points_cache[cache_key]
    
    def _get_world_bbox(self, instance: ThreeDInstance, view_type: str = 'agentview',
                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """获取实例的世界坐标包围盒（带缓存）"""
        cache_key = (id(instance), view_type, str(kwargs))
        if cache_key not in self._world_bbox_cache:
            bbox = instance.get_world_bbox3d(view_type=view_type, **kwargs)
            self._world_bbox_cache[cache_key] = bbox
        return self._world_bbox_cache[cache_key]
    
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
