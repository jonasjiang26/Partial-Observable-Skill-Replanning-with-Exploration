#!/usr/bin/env python3

import h5py
import numpy as np
import cv2
from typing import List, Optional, Tuple, Union
import os


class TwoDInstance:
    """
    物体mask实例类，表示一个物体的分割掩码。
    
    属性:
        object_id: 物体ID
        mask: 原始mask（物体区域保持原ID值，其他为0）
        binary_mask: 二值mask（物体区域为255，其他为0）
        shape: mask的形状 (height, width)
    """
    
    def __init__(self, object_id: int, original_mask: np.ndarray, binary_mask: np.ndarray, depth_agentview: np.ndarray, depth_eye_in_hand: np.ndarray):
        """
        初始化物体mask实例。
        
        参数:
            object_id: 物体ID
            original_mask: 原始mask数组
            binary_mask: 二值mask数组
        """
        self.object_id = object_id
        self.original_mask = original_mask
        self.binary_mask = binary_mask
        self.shape = original_mask.shape
        self.depth_agentview = depth_agentview
        self.depth_eye_in_hand = depth_eye_in_hand



    @property
    def area(self) -> int:
        """
        获取物体mask的面积（像素数量）。
        
        返回:
            物体占用的像素数量
        """
        return np.sum(self.binary_mask > 0)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """
        获取物体的边界框 (x_min, y_min, x_max, y_max)。
        
        返回:
            边界框坐标元组
        """
        coords = np.where(self.binary_mask > 0)
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return (x_min, y_min, x_max, y_max)
    
    @property
    def center(self) -> Tuple[int, int]:
        """
        获取物体的中心点坐标 (x, y)。
        
        返回:
            中心点坐标元组
        """
        coords = np.where(self.binary_mask > 0)
        if len(coords[0]) == 0:
            return (0, 0)
        
        y_center = int(np.mean(coords[0]))
        x_center = int(np.mean(coords[1]))
        return (x_center, y_center)
    
    def save(self, output_path: str, binary: bool = True) -> bool:
        """
        保存mask到文件。
        
        参数:
            output_path: 输出文件路径
            binary: 如果为True，保存二值mask；如果为False，保存原始mask
        
        返回:
            是否保存成功
        """
        mask_to_save = self.binary_mask if binary else self.mask
        success = cv2.imwrite(output_path, mask_to_save)
        return success
    
    def get_cropped_mask(self, binary: bool = True, padding: int = 0) -> np.ndarray:
        """
        获取裁剪后的mask（只包含物体区域，带可选边距）。
        
        参数:
            binary: 是否返回二值mask
            padding: 边界框周围的边距（像素）
        
        返回:
            裁剪后的mask数组
        """
        x_min, y_min, x_max, y_max = self.bbox
        
        if x_max == 0 and y_max == 0:
            return np.array([])
        
        # 添加边距
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(self.shape[1], x_max + padding + 1)
        y_max = min(self.shape[0], y_max + padding + 1)
        
        mask_to_crop = self.binary_mask if binary else self.mask
        cropped = mask_to_crop[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def __repr__(self) -> str:
        """返回对象的字符串表示。"""
        return f"ObjectMask(id={self.object_id}, area={self.area}, bbox={self.bbox})"


class MaskExtractor:
    """
    2D Mask提取器类，用于从多物体mask图像中提取所有物体实例。
    
    属性:
        original_mask_agentview: agentview视角的原始多物体mask图像
        original_mask_eye_in_hand: eye_in_hand视角的原始多物体mask图像
        background_value: 背景像素值（可以是单个值或数组）
        objects: 提取出的物体实例列表
    """
    
    def __init__(self, background_value: Union[int, List[int], np.ndarray] = 0):
        """
        初始化mask提取器。
        
        参数:
            background_value: 背景像素值，可以是单个整数、整数列表或numpy数组，默认为0
                            例如: 0, [0, 255], np.array([0, 1, 2])
        """
        self.original_mask_agentview: Optional[np.ndarray] = None
        self.original_mask_eye_in_hand: Optional[np.ndarray] = None
        self.background_value = background_value
        self.objects: List[TwoDInstance] = []
        self.depth_agentview: Optional[np.ndarray] = None
        self.depth_eye_in_hand: Optional[np.ndarray] = None
    
    def load(self, path: str, frame_id: int) -> bool:
        """
        从文件加载mask图像。
        
        参数:
            mask_path: mask图像路径
        
        返回:
            是否加载成功
        """
        file = h5py.File(path, 'r')
        mask_agentview = file["data/demo_0/obs/agentview_segmentation"][frame_id]
        mask_eye_in_hand = file["data/demo_0/obs/eye_in_hand_segmentation"][frame_id]
        depth_agentview = file["data/demo_0/obs/agentview_depth"][frame_id]
        depth_eye_in_hand = file["data/demo_0/obs/eye_in_hand_depth"][frame_id]
        ee_ori = file["data/demo_0/obs/ee_ori"][frame_id]
        ee_pos = file["data/demo_0/obs/ee_pos"][frame_id]
        self.original_mask_agentview = mask_agentview
        self.original_mask_eye_in_hand = mask_eye_in_hand
        self.depth_agentview = depth_agentview
        self.depth_eye_in_hand = depth_eye_in_hand  
        self.ee_ori = ee_ori
        self.ee_pos = ee_pos
        return True
    
    def extract(self, binary: bool = True, view_type: str = 'agentview') -> List[TwoDInstance]:
        """
        从原始mask中提取所有物体实例。
        
        参数:
            binary: 是否生成二值mask
            view_type: 使用的视图类型，'agentview' 或 'eye_in_hand'，默认为 'agentview'
        
        返回:
            物体实例列表
        """
        # 选择要使用的mask
        if view_type == 'agentview':
            if self.original_mask_agentview is None:
                raise ValueError("请先加载mask图像（使用load()方法）")
            original_mask = self.original_mask_agentview
        elif view_type == 'eye_in_hand':
            if self.original_mask_eye_in_hand is None:
                raise ValueError("请先加载mask图像（使用load()方法）")
            original_mask = self.original_mask_eye_in_hand
        else:
            raise ValueError(f"不支持的视图类型: {view_type}，请使用 'agentview' 或 'eye_in_hand'")
        
        # 获取mask中所有唯一的像素值，这些值代表不同的物体ID
        unique_values = np.unique(original_mask)
        
        # 过滤背景值：支持单个值或数组
        # 如果background_value是数组，使用np.isin()检查元素是否在背景值数组中
        if isinstance(self.background_value, (list, np.ndarray, tuple)):
            # 多个背景值：使用np.isin()检查元素是否不在背景值数组中
            background_array = np.asarray(self.background_value)
            object_ids = unique_values[~np.isin(unique_values, background_array)]
        else:
            # 单个背景值：使用!=比较
            object_ids = unique_values[unique_values != self.background_value]
        
        self.objects = []
        
        for obj_id in object_ids:
            # 创建原始mask（物体保持原ID，其他为0）
            obj_original_mask = np.zeros_like(original_mask)
            obj_original_mask[original_mask == obj_id] = obj_id
            
            # 创建二值mask（物体为255，其他为0）
            binary_mask = np.zeros_like(original_mask, dtype=np.uint8)
            binary_mask[original_mask == obj_id] = 255
            
            #作用mask于深度图，得到深度图的物体区域
            obj_depth_agentview = self.depth_agentview[obj_original_mask > 0]
            obj_depth_eye_in_hand = self.depth_eye_in_hand[obj_original_mask > 0]
            
            # 创建物体实例
            obj = TwoDInstance(obj_id, obj_original_mask, binary_mask, obj_depth_agentview, obj_depth_eye_in_hand)
            self.objects.append(obj)
        
        return self.objects
    
    def ee_pos_ori_to_extrinsic_matrix(self, ee_pos: np.ndarray, ee_ori: np.ndarray) -> np.ndarray:
        """
        将ee_pos和ee_ori转换为extrinsic矩阵。
        
        参数:
            ee_pos: ee_pos
            ee_ori: ee_ori
        """
        ee_pos = self.ee_pos
        ee_ori = self.ee_ori
        extrinsic_matrix = np.linalg.inv(np.hstack((ee_ori, ee_pos.reshape(-1, 1))))
        return extrinsic_matrix
    
    def save_all(self, output_dir: str, base_name: str = "mask", 
                 format: str = 'png', binary: bool = True) -> List[str]:
        """
        保存所有物体的mask到指定目录。
        
        参数:
            output_dir: 输出目录
            base_name: 输出文件的基础名称
            format: 输出格式 ('png', 'jpg', 'jpeg')
            binary: 是否保存二值mask
        
        返回:
            保存的文件路径列表
        """
        if len(self.objects) == 0:
            print("警告: 没有物体需要保存，请先调用extract()方法")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for obj in self.objects:
            filename = f"{base_name}_object_{obj.object_id}.{format}"
            output_path = os.path.join(output_dir, filename)
            
            if obj.save(output_path, binary):
                saved_paths.append(output_path)
                print(f"已保存: {output_path}")
            else:
                print(f"警告: 保存失败: {output_path}")
        
        return saved_paths
    
    def get_object_by_id(self, object_id: int) -> Optional[TwoDInstance]:
        """
        根据物体ID获取物体实例。
        
        参数:
            object_id: 物体ID
        
        返回:
            物体实例，如果不存在则返回NoneS
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def __len__(self) -> int:
        """返回物体数量。"""
        return len(self.objects)
    
    def __iter__(self):
        """使对象可迭代。"""
        return iter(self.objects)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示。"""
        return f"MaskExtractor(background={self.background_value}, objects={len(self.objects)})"
