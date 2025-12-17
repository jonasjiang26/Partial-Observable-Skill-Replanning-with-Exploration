#!/usr/bin/env python3
import numpy as np
import h5py
import cv2
from typing import Optional, Tuple, Union, List, Dict
import os

from utils.mask_utils import TwoDInstance

_CAMERA_EXTRINSICS = {
    'agentview': {
        'pos': np.array([0.6586131746834771, 0.0, 1.6103500240372423]),  # (x, y, z)
        'quat': np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])  # (w, x, y, z)
    },
    'eye_in_hand': None  # uses ee_pos and ee_ori
}


_CAMERA_INTRINSICS = {
    'agentview': None,  # 如果为None，则根据视场角计算
    'eye_in_hand': None
}

# 末端执行器 -> eye_in_hand 相机的固定变换（如果未设置则假设重合）
_EE_TO_EYE_IN_HAND_TRANSFORM = None

def set_camera_extrinsic(view_type: str, pos: np.ndarray, quat: np.ndarray):

    global _CAMERA_EXTRINSICS
    if view_type not in ['agentview', 'eye_in_hand']:
        raise ValueError(f"view type not supported: {view_type}")
    _CAMERA_EXTRINSICS[view_type] = {
        'pos': np.array(pos),
        'quat': np.array(quat)
    }

def set_camera_intrinsic(view_type: str, fx: float, fy: float, cx: float, cy: float):

    global _CAMERA_INTRINSICS
    if view_type not in ['agentview', 'eye_in_hand']:
        raise ValueError(f"view type not supported: {view_type}")
    _CAMERA_INTRINSICS[view_type] = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def set_ee_to_eye_in_hand_transform(pos: np.ndarray, quat: np.ndarray):
    """
    设置末端执行器到eye_in_hand相机的固定变换 T_ee_cam。
    参数:
        pos: 相机相对于末端执行器的位置 (x, y, z)，单位：米
        quat: 相机相对于末端执行器的姿态四元数 (w, x, y, z)
    """
    global _EE_TO_EYE_IN_HAND_TRANSFORM
    _EE_TO_EYE_IN_HAND_TRANSFORM = {
        'pos': np.array(pos),
        'quat': np.array(quat)
    }


def get_ee_to_eye_in_hand_transform() -> Optional[Dict]:
    """获取末端执行器到eye_in_hand相机的固定变换。"""
    return _EE_TO_EYE_IN_HAND_TRANSFORM

def opengl_depth_to_linear(depth_buffer: np.ndarray, near: float = 0.01, far: float = 10.0) -> np.ndarray:
    """
    将OpenGL深度缓冲值转换为线性深度值。
    
    OpenGL深度缓冲使用非线性映射：
    z_buffer = (1/z_linear - 1/near) / (1/far - 1/near)
    
    反算公式：
    z_linear = 1 / (z_buffer * (1/far - 1/near) + 1/near)
    或者：
    z_linear = near * far / (far - z_buffer * (far - near))
    
    参数:
        depth_buffer: OpenGL深度缓冲值（范围通常在[0, 1]或归一化值）
        near: 近裁剪平面距离（米），默认0.01
        far: 远裁剪平面距离（米），默认10.0
    
    返回:
        线性深度值（米）
    """
    depth_buffer = np.clip(depth_buffer, 0.0, 1.0)
    
    # OpenGL深度缓冲到线性深度的转换
    # 方法1：使用标准公式
    # z_linear = 1 / (z_buffer * (1/far - 1/near) + 1/near)
    
    # 方法2：使用另一种等价形式（数值更稳定）
    # z_linear = near * far / (far - z_buffer * (far - near))
    
    # 使用方法2，因为它对near和far的差异更稳定
    denominator = far - depth_buffer * (far - near)
    # 避免除零
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    z_linear = near * far / denominator
    
    return z_linear

def statistical_outlier_removal(points: np.ndarray, 
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0) -> np.ndarray:
    """
    统计滤波：移除距离邻居点平均距离超过阈值的点（离群点）。
    
    参数:
        points: 点云 (N, 3)
        nb_neighbors: 用于计算平均距离的邻居点数量，默认20
        std_ratio: 标准差倍数阈值，默认2.0（超过平均距离+2*标准差的点被移除）
    
    返回:
        过滤后的点云
    """
    if len(points) < nb_neighbors + 1:
        return points
    
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("警告: scipy未安装，无法使用统计滤波，返回原始点云")
        return points
    
    # 构建KD树用于快速最近邻搜索
    tree = cKDTree(points)
    
    # 计算每个点到其k个最近邻的平均距离
    distances, _ = tree.query(points, k=nb_neighbors + 1)  # +1因为包含自己
    mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己
    
    # 计算全局平均距离和标准差
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # 保留在阈值内的点
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances < threshold
    
    return points[mask]


def radius_outlier_removal(points: np.ndarray,
                           radius: float = 0.05,
                           min_neighbors: int = 5) -> np.ndarray:
    """
    半径滤波：移除周围邻居点数量少于阈值的点。
    
    参数:
        points: 点云 (N, 3)
        radius: 搜索半径（米），默认0.05
        min_neighbors: 最小邻居点数量，默认5
    
    返回:
        过滤后的点云
    """
    if len(points) == 0:
        return points
    
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("警告: scipy未安装，无法使用半径滤波，返回原始点云")
        return points
    
    # 构建KD树
    tree = cKDTree(points)
    
    # 对每个点，统计半径内的邻居点数量
    neighbor_counts = tree.query_ball_point(points, radius, return_length=True)
    
    # 保留邻居点数量足够的点
    mask = neighbor_counts >= min_neighbors
    return points[mask]


def z_value_filter(points: np.ndarray,
                   z_std_ratio: float = 3.0) -> np.ndarray:
    """
    基于Z值的滤波：移除Z值异常的点（可能是深度图噪声）。
    
    参数:
        points: 点云 (N, 3)
        z_std_ratio: Z值标准差倍数阈值，默认3.0
    
    返回:
        过滤后的点云
    """
    if len(points) == 0:
        return points
    
    z_values = points[:, 2]
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)
    
    # 保留在阈值内的点
    z_min = z_mean - z_std_ratio * z_std
    z_max = z_mean + z_std_ratio * z_std
    mask = (z_values >= z_min) & (z_values <= z_max)
    
    return points[mask]


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat

    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-8:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R

class ThreeDInstance(TwoDInstance):
    def __init__(self, object_id: int, original_mask: np.ndarray, binary_mask: np.ndarray, 
                 depth_agentview: np.ndarray, depth_eye_in_hand: np.ndarray,
                 ee_pos: Optional[np.ndarray] = None, ee_ori: Optional[np.ndarray] = None,
                 depth_agentview_full: Optional[np.ndarray] = None,
                 depth_eye_in_hand_full: Optional[np.ndarray] = None):
        """
        初始化3D实例。
        
        参数:
            object_id: 物体ID
            original_mask: 原始mask
            binary_mask: 二值mask
            depth_agentview: agentview深度值（可以是1D数组或完整2D图）
            depth_eye_in_hand: eye_in_hand深度值（可以是1D数组或完整2D图）
            ee_pos: 末端执行器位置（可选）
            ee_ori: 末端执行器姿态axis-angle（可选）
            depth_agentview_full: 完整的agentview深度图（2D数组，用于投影）
            depth_eye_in_hand_full: 完整的eye_in_hand深度图（2D数组，用于投影）
        """
        super().__init__(object_id, original_mask, binary_mask, depth_agentview, depth_eye_in_hand)
        self.ee_pos = ee_pos
        self.ee_ori = ee_ori
        # 保存完整的深度图（用于投影）
        self._depth_agentview_full = depth_agentview_full
        self._depth_eye_in_hand_full = depth_eye_in_hand_full
        # 缓存不同视图的变换矩阵
        self._T_w_cam = {}  # 摄像机坐标系 -> 世界坐标系 (按view_type缓存)
    
    def load(self, path: str, frame_id: int) -> bool:
        file = h5py.File(path, 'r')
        ee_ori = file["data/demo_0/obs/ee_ori"][frame_id]
        ee_pos = file["data/demo_0/obs/ee_pos"][frame_id]
        self.ee_ori = ee_ori
        self.ee_pos = ee_pos
        self._T_w_cam = {}
        return True
    
    def axisangle_to_rotation_matrix(self) -> np.ndarray:
        """
        将轴角(axis-angle)转换为旋转矩阵（使用Rodrigues公式）。
        
        返回:
            3x3旋转矩阵
        """
        if self.ee_ori is None:
            raise ValueError("ee_ori未设置，请先调用load()方法")
        
        theta = np.linalg.norm(self.ee_ori)
        if theta < 1e-10:
            return np.eye(3)
        
        axis = self.ee_ori / theta
        
        # Rodrigues旋转公式: R = I + sin(θ)*K + (1-cos(θ))*K²
        # 其中K是反对称矩阵
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        rotation_matrix = (np.eye(3) + 
                          np.sin(theta) * K + 
                          (1 - np.cos(theta)) * np.dot(K, K))
        return rotation_matrix

    def get_camera_extrinsic(self, view_type: str = 'agentview') -> np.ndarray:
        """
        根据view_type获取摄像机到世界坐标系的变换矩阵 T_cam_w。
        
        这个函数返回摄像机坐标系到世界坐标系的变换矩阵。
        给定摄像机中的一点 p_cam，可以通过以下方式得到世界坐标系中的点：
        p_world = T_cam_w @ p_cam_homo
        
        参数:
            view_type: 'agentview' 或 'eye_in_hand'
        
        返回:
            4x4齐次变换矩阵 T_cam_w（摄像机坐标系 -> 世界坐标系）
        """
        if view_type in self._T_w_cam:
            return self._T_w_cam[view_type]
        
        # T_cam_w = [R_cam_w | t_cam_w]
        # 其中 R_cam_w 是从摄像机到世界的旋转
        # t_cam_w 是摄像机在世界坐标系中的位置
        T_cam_w = np.eye(4)
        
        if view_type == 'agentview':
            # 使用固定的agentview相机外参
            if _CAMERA_EXTRINSICS['agentview'] is None:
                raise ValueError("agentview相机外参未配置，请使用 set_camera_extrinsic() 设置")
            
            cam_pos = _CAMERA_EXTRINSICS['agentview']['pos']  # 摄像机在世界坐标系中的位置
            cam_quat = _CAMERA_EXTRINSICS['agentview']['quat']
            
            # 四元数转旋转矩阵（这是摄像机的方向）
            # 注意：根据四元数的定义，这给出的可能是世界->摄像机的旋转
            # 我们需要的是摄像机->世界的旋转，所以需要转置
            R_w_cam = quaternion_to_rotation_matrix(cam_quat)  # 世界->摄像机
            R_cam_w = R_w_cam.T  # 摄像机->世界（通过转置得到逆矩阵）
            
            T_cam_w[:3, :3] = R_cam_w
            T_cam_w[:3, 3] = cam_pos
            
            # 调试信息：检查变换矩阵
            print(f"  Agentview相机外参:")
            print(f"    摄像机位置（世界坐标系）: {cam_pos}")
            print(f"    四元数: {cam_quat}")
            print(f"    R_w_cam (世界->摄像机):\n{R_w_cam}")
            print(f"    R_cam_w (摄像机->世界):\n{R_cam_w}")
            print(f"    T_cam_w (摄像机->世界):\n{T_cam_w}")
            
        elif view_type == 'eye_in_hand':
            # eye_in_hand 相机固定在末端执行器上：T_cam_w = (T_w_ee @ T_ee_cam)^-1
            if self.ee_pos is None or self.ee_ori is None:
                raise ValueError("ee_pos和ee_ori未设置，请先调用load()方法")
            
            # 1) 世界->末端执行器
            R_w_ee = self.axisangle_to_rotation_matrix()  # 世界->ee
            T_w_ee = np.eye(4)
            T_w_ee[:3, :3] = R_w_ee
            T_w_ee[:3, 3] = self.ee_pos.reshape(3,)
            
            # 2) 末端执行器->相机（如果未设置，默认单位变换）
            if _EE_TO_EYE_IN_HAND_TRANSFORM is None:
                T_ee_cam = np.eye(4)
                print("  警告: 未设置末端执行器到相机的变换，假设相机与末端执行器重合")
            else:
                ee_to_cam_pos = _EE_TO_EYE_IN_HAND_TRANSFORM['pos']
                ee_to_cam_quat = _EE_TO_EYE_IN_HAND_TRANSFORM['quat']
                R_ee_cam = quaternion_to_rotation_matrix(ee_to_cam_quat)
                T_ee_cam = np.eye(4)
                T_ee_cam[:3, :3] = R_ee_cam
                T_ee_cam[:3, 3] = ee_to_cam_pos
            
            # 3) 世界->相机
            T_w_cam_full = T_w_ee @ T_ee_cam  # 世界->相机
            T_cam_w = np.linalg.inv(T_w_cam_full)  # 相机->世界
            
            # 调试信息
            print("  Eye-in-hand相机外参计算:")
            print(f"    末端执行器位置(ee_pos): {self.ee_pos}")
            print(f"    末端执行器姿态(ee_ori axis-angle): {self.ee_ori}")
            if _EE_TO_EYE_IN_HAND_TRANSFORM is not None:
                print(f"    相机相对末端执行器位置: {ee_to_cam_pos}")
                print(f"    相机相对末端执行器姿态(四元数): {ee_to_cam_quat}")
            print(f"    T_w_ee (世界->末端执行器):\n{T_w_ee}")
            print(f"    T_ee_cam (末端执行器->相机):\n{T_ee_cam}")
            print(f"    T_cam_w (相机->世界):\n{T_cam_w}")
        else:
            raise ValueError(f"不支持的视图类型: {view_type}")
        
        self._T_w_cam[view_type] = T_cam_w
        return T_cam_w
    
    def get_camera_to_world_transform(self, view_type: str = 'agentview') -> np.ndarray:
        """
        获取相机坐标系到世界坐标系的变换矩阵 T_cam_w。
        用于将相机坐标系下的点投影到世界坐标系。
        
        参数:
            view_type: 'agentview' 或 'eye_in_hand'
        
        返回:
            4x4齐次变换矩阵 T_cam_w（相机坐标系 -> 世界坐标系）
        """
        # get_camera_extrinsic() 现在直接返回 T_cam_w
        return self.get_camera_extrinsic(view_type)
    def get_intrinsic_matrix(self, view_type: str = 'agentview') -> np.ndarray:
        """
        获取相机内参矩阵。
        
        如果配置了内参，则使用配置的值；否则根据视场角计算。
        
        注意：fx和fy的单位应该是"像素/米"（如果深度值单位是米）。
        反投影公式：X = (u - cx) * Z / fx
        - u, cx 单位：像素
        - Z 单位：米
        - X 单位：米
        - 因此 fx 单位必须是：像素/米
        
        返回:
            3x3内参矩阵，fx和fy的单位是"像素/米"（如果深度值单位是米）
        """
        # 如果配置了内参，直接使用
        if _CAMERA_INTRINSICS.get(view_type) is not None:
            config = _CAMERA_INTRINSICS[view_type]
            fx = config['fx']
            fy = config['fy']
            cx = config['cx']
            cy = config['cy']
            print(f"  使用配置的内参矩阵 ({view_type}):")
            print(f"    fx={fx:.2f} 像素/米, fy={fy:.2f} 像素/米")
            print(f"    cx={cx:.2f} 像素, cy={cy:.2f} 像素")
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # 否则根据视场角计算
        # 使用mask的shape，因为mask和深度图应该匹配
        if len(self.shape) == 2:
            H, W = self.shape
        elif len(self.shape) == 3:
            H, W = self.shape[:2]
        else:
            raise ValueError(f"不支持的mask形状: {self.shape}，期望2D (H, W)")
        
        if view_type == "eye_in_hand":
            
            fovy = 75.0
        elif view_type == "agentview":
            fovy = 45.0
        else:
            raise ValueError(f"viewtype not supported:{view_type}")
        
        fovy_rad = fovy * np.pi / 180.0

        
        # 计算焦距（单位：像素/米）
        # 注意：这个公式假设传感器物理尺寸与像素数成正比
        # 对于192x192图像，75度视场角：
        # 如果假设传感器高度为 h 米，则 f = h / (2 * tan(fovy/2)) (米)
        # fx = f * (W / sensor_width) = f * (W / h) = W / (2 * tan(fovy/2)) (像素/米)
        # 这个公式假设传感器是方形的（sensor_width = sensor_height = h）
        fy = H / (2 * np.tan(fovy_rad / 2))  # 单位：像素/米
        fx = W / (2 * np.tan(fovy_rad / 2))  # 单位：像素/米
        
        # 主点坐标（像素）
        cx = W / 2.0
        cy = H / 2.0
        
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        print(f"  内参矩阵计算（基于视场角）:")
        print(f"    图像尺寸: {W}x{H} 像素")
        print(f"    视场角: {fovy}度")
        print(f"    fx={fx:.2f} 像素/米, fy={fy:.2f} 像素/米")
        print(f"    cx={cx:.2f} 像素, cy={cy:.2f} 像素")
        print(f"    警告: 这个计算假设传感器物理尺寸与像素数成正比")
                
        return intrinsic_matrix

    def backproject_depth_to_camera_frame(self, view_type: str = 'agentview', 
                                          depth_unit: str = 'm',
                                          depth_scale: float = 1.0,
                                          convert_opengl_depth: bool = True,
                                          opengl_near: float = 0.001,
                                          opengl_far: float = 50.0) -> np.ndarray:
        """
        将深度图反投影到相机坐标系（末端执行器坐标系）。
        
        参数:
            view_type: 'agentview' 或 'eye_in_hand'
            depth_unit: 深度值单位，'m'（米）或 'cm'（厘米），默认为 'm'
            depth_scale: 深度值缩放因子，默认为1.0（不缩放）
            convert_opengl_depth: 是否将OpenGL深度缓冲值转换为线性深度，默认为True
            opengl_near: OpenGL近裁剪平面距离（米），默认0.001
            opengl_far: OpenGL远裁剪平面距离（米），默认50.0
        
        返回:
            (N, 3) 的numpy数组，每行是相机坐标系下的3D点 [X, Y, Z]（单位：米）
        """
        # 需要完整的深度图，而不是索引后的1D数组
        if view_type == 'agentview':
            if hasattr(self, '_depth_agentview_full') and self._depth_agentview_full is not None:
                depth = self._depth_agentview_full
            else:
                raise ValueError("需要完整的agentview深度图，请确保在创建实例时传入完整的深度图")
        elif view_type == 'eye_in_hand':
            if hasattr(self, '_depth_eye_in_hand_full') and self._depth_eye_in_hand_full is not None:
                depth = self._depth_eye_in_hand_full
            else:
                raise ValueError("需要完整的eye_in_hand深度图，请确保在创建实例时传入完整的深度图")
        else:
            raise ValueError(f"不支持的视图类型: {view_type}")
        
        # 确保深度图是2D的
        if len(depth.shape) == 3:
            # 如果是3D (H, W, C)，取第一个通道或平均
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]
            else:
                # 多个通道，取平均或第一个通道
                depth = depth[:, :, 0]  # 或者使用 np.mean(depth, axis=2)
        elif len(depth.shape) != 2:
            raise ValueError(f"不支持的深度图形状: {depth.shape}，期望2D (H, W) 或 3D (H, W, C)")
        
        K = self.get_intrinsic_matrix(view_type)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 调试信息：检查内参矩阵
        print(f"  内参矩阵 K:")
        print(f"    fx={fx:.2f}, fy={fy:.2f}")
        print(f"    cx={cx:.2f}, cy={cy:.2f}")
        print(f"    图像尺寸: {depth.shape}")
        
        # 获取mask区域内的像素坐标
        ys, xs = np.where(self.binary_mask > 0)
        
        # 确保索引在有效范围内
        ys = np.clip(ys, 0, depth.shape[0] - 1)
        xs = np.clip(xs, 0, depth.shape[1] - 1)
        
        zs = depth[ys, xs]
        
        # 过滤无效深度值（0或NaN）
        valid_mask = (zs > 0) & np.isfinite(zs)
        if not np.any(valid_mask):
            raise ValueError("没有有效的深度值")
        
        xs = xs[valid_mask]
        ys = ys[valid_mask]
        zs = zs[valid_mask]
        
        # 转换OpenGL深度缓冲值为线性深度
        if convert_opengl_depth:
            print(f"  转换OpenGL深度缓冲值为线性深度:")
            print(f"    原始深度值范围: [{zs.min():.6f}, {zs.max():.6f}]")
            print(f"    OpenGL near: {opengl_near}, far: {opengl_far}")
            
            # 如果深度值不在[0,1]范围内，可能需要归一化
            # 检查深度值是否已经归一化
            if zs.max() > 1.0:
                # 深度值可能已经是线性深度（米），不需要转换
                print(f"    警告: 深度值范围超出[0,1]，可能已经是线性深度，跳过OpenGL转换")
                zs_linear = zs
            else:
                # 假设深度值是OpenGL深度缓冲值，转换为线性深度
                zs_linear = opengl_depth_to_linear(zs, near=opengl_near, far=opengl_far)
                print(f"    转换后线性深度范围: [{zs_linear.min():.6f}, {zs_linear.max():.6f}]")
        else:
            zs_linear = zs
        
        # 应用深度值缩放
        zs_scaled = zs_linear * depth_scale
        
        # 调试信息：显示深度值缩放信息（仅在缩放因子不为1时显示）
        if depth_scale != 1.0:
            print(f"  深度值缩放信息:")
            print(f"    缩放因子: {depth_scale}")
            print(f"    缩放后深度值范围: [{zs_scaled.min():.6f}, {zs_scaled.max():.6f}]")
        
        # 转换深度值单位
        if depth_unit == 'cm':
            # 从米转换为厘米
            zs_converted = zs_scaled * 100.0
            # 内参矩阵也需要相应调整（fx, fy的单位是像素/米，需要转换为像素/厘米）
            fx_converted = fx / 100.0
            fy_converted = fy / 100.0
        else:
            # 保持米为单位
            zs_converted = zs_scaled
            fx_converted = fx
            fy_converted = fy
        
        # 反投影到相机坐标系
        # 注意：深度值zs_converted是相机坐标系下的Z值（距离相机的距离）
        X = (xs - cx) * zs_converted / fx_converted
        Y = (ys - cy) * zs_converted / fy_converted
        Z = zs_converted
        
        points_cam = np.stack([X, Y, Z], axis=-1)  # (N, 3)
        
        # 调试信息：检查相机坐标系下的点云
        if len(points_cam) > 0:
            unit_label = "cm" if depth_unit == 'cm' else "m"
            print(f"  相机坐标系点云范围（单位：{unit_label}）:")
            print(f"    X: [{points_cam[:, 0].min():.4f}, {points_cam[:, 0].max():.4f}], "
                  f"范围: {points_cam[:, 0].max() - points_cam[:, 0].min():.4f} {unit_label}")
            print(f"    Y: [{points_cam[:, 1].min():.4f}, {points_cam[:, 1].max():.4f}], "
                  f"范围: {points_cam[:, 1].max() - points_cam[:, 1].min():.4f} {unit_label}")
            print(f"    Z: [{points_cam[:, 2].min():.4f}, {points_cam[:, 2].max():.4f}], "
                  f"范围: {points_cam[:, 2].max() - points_cam[:, 2].min():.4f} {unit_label}")
        
        return points_cam
    
    def denoise_pointcloud(self, points: np.ndarray,
                          method: str = 'radius',
                          **kwargs) -> np.ndarray:
        """
        对点云进行去噪处理。
        
        参数:
            points: 点云 (N, 3)
            method: 去噪方法，'statistical'（统计滤波）、'radius'（半径滤波）、
                   'z_value'（Z值滤波）或'combined'（组合），默认'statistical'
            **kwargs: 去噪方法的参数
                - statistical: nb_neighbors=20, std_ratio=2.0
                - radius: radius=0.05, min_neighbors=5
                - z_value: z_std_ratio=3.0
        
        返回:
            去噪后的点云
        """
        if len(points) == 0:
            return points
        
        original_count = len(points)
        
        if method == 'statistical':
            nb_neighbors = kwargs.get('nb_neighbors', 20)
            std_ratio = kwargs.get('std_ratio', 2.0)
            filtered_points = statistical_outlier_removal(points, nb_neighbors, std_ratio)
        elif method == 'radius':
            radius = kwargs.get('radius', 0.05)
            min_neighbors = kwargs.get('min_neighbors', 5)
            filtered_points = radius_outlier_removal(points, radius, min_neighbors)
        elif method == 'z_value':
            z_std_ratio = kwargs.get('z_std_ratio', 3.0)
            filtered_points = z_value_filter(points, z_std_ratio)
        elif method == 'combined':
            # 先Z值滤波，再统计滤波
            z_std_ratio = kwargs.get('z_std_ratio', 3.0)
            filtered_points = z_value_filter(points, z_std_ratio)
            nb_neighbors = kwargs.get('nb_neighbors', 20)
            std_ratio = kwargs.get('std_ratio', 2.0)
            filtered_points = statistical_outlier_removal(filtered_points, nb_neighbors, std_ratio)
        else:
            print(f"警告: 未知的去噪方法 '{method}'，返回原始点云")
            return points
        
        filtered_count = len(filtered_points)
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"  点云去噪 ({method}): 原始点数={original_count}, 去噪后={filtered_count}, "
                  f"移除={removed_count} ({removed_count/original_count*100:.1f}%)")
        
        return filtered_points
    
    def project_to_world_coordinates(self, view_type: str = 'agentview', 
                                     depth_unit: str = 'm',
                                     depth_scale: float = 1.0,
                                     convert_opengl_depth: bool = True,
                                     opengl_near: float = 0.01,
                                     opengl_far: float = 10.0,
                                     denoise: bool = True,
                                     denoise_method: str = 'z_value',
                                     denoise_params: Optional[dict] = None) -> np.ndarray:
        """
        将深度图投影到世界坐标系，得到3D instance点云。
        
        参数:
            view_type: 'agentview' 或 'eye_in_hand'
            depth_unit: 深度值单位，'m'（米）或 'cm'（厘米），默认为 'm'
            depth_scale: 深度值缩放因子，默认为1.0（不缩放）
            convert_opengl_depth: 是否将OpenGL深度缓冲值转换为线性深度，默认为True
            opengl_near: OpenGL近裁剪平面距离（米），默认0.01
            opengl_far: OpenGL远裁剪平面距离（米），默认10.0
            denoise: 是否对点云进行去噪，默认True
            denoise_method: 去噪方法，'statistical'、'radius'、'z_value'或'combined'，默认'statistical'
            denoise_params: 去噪参数字典，如果为None则使用默认值
        
        返回:
            (N, 3) 的numpy数组，每行是世界坐标系下的3D点 [X, Y, Z]（单位：米）
        """
        # 1. 反投影到相机坐标系
        points_cam = self.backproject_depth_to_camera_frame(
            view_type, 
            depth_unit=depth_unit, 
            depth_scale=depth_scale,
            convert_opengl_depth=convert_opengl_depth,
            opengl_near=opengl_near,
            opengl_far=opengl_far
        )
        
        # 2. 转换到齐次坐标
        N = points_cam.shape[0]
        points_cam_homo = np.hstack([points_cam, np.ones((N, 1))])  # (N, 4)
        
        # 3. 使用T_cam_w变换到世界坐标系
        T_cam_w = self.get_camera_to_world_transform(view_type)
        
        # 调试信息：检查变换矩阵
        if points_cam.shape[0] > 0:
            print(f"  变换矩阵 T_cam_w ({view_type} 相机->世界):")
            print(f"    旋转部分:\n{T_cam_w[:3, :3]}")
            print(f"    平移部分: {T_cam_w[:3, 3]}")
        
        points_world_homo = (T_cam_w @ points_cam_homo.T).T  # (N, 4)
        
        # 4. 转换回3D坐标
        points_world = points_world_homo[:, :3]  # (N, 3)
        
        # 5. 点云去噪
        if denoise and len(points_world) > 0:
            if denoise_params is None:
                denoise_params = {}
            points_world = self.denoise_pointcloud(points_world, method=denoise_method, **denoise_params)
        
        # 调试信息：检查世界坐标系下的点云
        if len(points_world) > 0:
            unit_label = "cm" if depth_unit == 'cm' else "m"
            print(f"  世界坐标系点云范围（单位：{unit_label}）:")
            print(f"    X: [{points_world[:, 0].min():.4f}, {points_world[:, 0].max():.4f}], "
                  f"范围: {points_world[:, 0].max() - points_world[:, 0].min():.4f} {unit_label}")
            print(f"    Y: [{points_world[:, 1].min():.4f}, {points_world[:, 1].max():.4f}], "
                  f"范围: {points_world[:, 1].max() - points_world[:, 1].min():.4f} {unit_label}")
            print(f"    Z: [{points_world[:, 2].min():.4f}, {points_world[:, 2].max():.4f}], "
                  f"范围: {points_world[:, 2].max() - points_world[:, 2].min():.4f} {unit_label}")
        
        return points_world
    
    def get_world_bbox3d(self, view_type: str = 'agentview', 
                        depth_unit: str = 'm',
                        depth_scale: float = 1.0,
                        convert_opengl_depth: bool = True,
                        opengl_near: float = 0.01,
                        opengl_far: float = 10.0,
                        denoise: bool = True,
                        denoise_method: str = 'z_value',
                        denoise_params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取3D instance在世界坐标系中的包围盒。
        
        参数:
            view_type: 'agentview' 或 'eye_in_hand'
            depth_unit: 深度值单位，'m'（米）或 'cm'（厘米），默认为 'm'
            depth_scale: 深度值缩放因子，默认为1.0（不缩放）
            convert_opengl_depth: 是否将OpenGL深度缓冲值转换为线性深度，默认为True
            opengl_near: OpenGL近裁剪平面距离（米），默认0.01
            opengl_far: OpenGL远裁剪平面距离（米），默认10.0
            denoise: 是否对点云进行去噪，默认True
            denoise_method: 去噪方法，默认'statistical'
            denoise_params: 去噪参数字典，如果为None则使用默认值
        
        返回:
            (xyz_min, xyz_max) 元组，每个是(3,)数组（单位：米）
        """
        points_world = self.project_to_world_coordinates(
            view_type, 
            depth_unit=depth_unit, 
            depth_scale=depth_scale,
            convert_opengl_depth=convert_opengl_depth,
            opengl_near=opengl_near,
            opengl_far=opengl_far,
            denoise=denoise,
            denoise_method=denoise_method,
            denoise_params=denoise_params
        )
        if points_world.shape[0] == 0:
            return np.zeros(3), np.zeros(3)
        
        xyz_min = points_world.min(axis=0)
        xyz_max = points_world.max(axis=0)
        return xyz_min, xyz_max


def merge_multi_frame_pointclouds(instances: List[ThreeDInstance],
                                  view_type: str = 'agentview',
                                  depth_unit: str = 'm',
                                  depth_scale: float = 1.0,
                                  convert_opengl_depth: bool = True,
                                  opengl_near: float = 0.01,
                                  opengl_far: float = 10.0,
                                  denoise: bool = True,
                                  denoise_method: str = 'statistical',
                                  denoise_params: Optional[Dict] = None,
                                  return_per_frame: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    合并多帧图像中同一个物体的点云到世界坐标系。
    
    参数:
        instances: ThreeDInstance对象列表，每个对应一帧图像中的同一个物体
        view_type: 'agentview' 或 'eye_in_hand'，默认'agentview'
        depth_unit: 深度值单位，'m'（米）或 'cm'（厘米），默认为 'm'
        depth_scale: 深度值缩放因子，默认为1.0（不缩放）
        convert_opengl_depth: 是否将OpenGL深度缓冲值转换为线性深度，默认为True
        opengl_near: OpenGL近裁剪平面距离（米），默认0.01
        opengl_far: OpenGL远裁剪平面距离（米），默认10.0
        denoise: 是否对点云进行去噪，默认True
        denoise_method: 去噪方法，默认'statistical'
        denoise_params: 去噪参数字典，如果为None则使用默认值
    
    返回:
        合并后的点云 (N, 3)，单位：米
    """
    if len(instances) == 0:
        return np.empty((0, 3))
    
    # 检查所有实例是否具有相同的object_id
    object_ids = [inst.object_id for inst in instances]
    if len(set(object_ids)) > 1:
        print(f"警告: 合并的点云来自不同的物体ID: {object_ids}")
    
    # 对每一帧投影到世界坐标系并收集点云
    all_points = []
    for i, instance in enumerate(instances):
        points_world = instance.project_to_world_coordinates(
            view_type=view_type,
            depth_unit=depth_unit,
            depth_scale=depth_scale,
            convert_opengl_depth=convert_opengl_depth,
            opengl_near=opengl_near,
            opengl_far=opengl_far,
            denoise=denoise,
            denoise_method=denoise_method,
            denoise_params=denoise_params
        )
        if len(points_world) > 0:
            all_points.append(points_world)
            print(f"  帧 {i+1}: 添加了 {len(points_world)} 个点")
    
    # 合并所有点云
    if len(all_points) == 0:
        return np.empty((0, 3))
    
    merged_points = np.vstack(all_points)
    print(f"\n合并后的点云:")
    print(f"  总点数: {len(merged_points)}")
    unit_label = "cm" if depth_unit == 'cm' else "m"
    print(f"  世界坐标系点云范围（单位：{unit_label}）:")
    print(f"    X: [{merged_points[:, 0].min():.4f}, {merged_points[:, 0].max():.4f}], "
          f"范围: {merged_points[:, 0].max() - merged_points[:, 0].min():.4f} {unit_label}")
    print(f"    Y: [{merged_points[:, 1].min():.4f}, {merged_points[:, 1].max():.4f}], "
          f"范围: {merged_points[:, 1].max() - merged_points[:, 1].min():.4f} {unit_label}")
    print(f"    Z: [{merged_points[:, 2].min():.4f}, {merged_points[:, 2].max():.4f}], "
          f"范围: {merged_points[:, 2].max() - merged_points[:, 2].min():.4f} {unit_label}")
    
    return merged_points


def register_and_merge_pointclouds(instances: List[ThreeDInstance],
                                   view_type: str = 'agentview',
                                   depth_unit: str = 'm',
                                   depth_scale: float = 1.0,
                                   convert_opengl_depth: bool = True,
                                   opengl_near: float = 0.01,
                                   opengl_far: float = 10.0,
                                   denoise: bool = True,
                                   denoise_method: str = 'statistical',
                                   denoise_params: Optional[Dict] = None,
                                   voxel_size: float = 0.01,
                                   icp_max_corr_factor: float = 1.5,
                                   icp_max_iter: int = 50,
                                   return_per_frame: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """
    使用Open3D对多帧同一物体的点云进行ICP配准并融合。
    如果未安装open3d，则退化为简单合并。
    """
    if len(instances) == 0:
        return np.empty((0, 3))
    
    try:
        import open3d as o3d
    except ImportError:
        print("警告: 未安装open3d，使用简单合并")
        merged = merge_multi_frame_pointclouds(
            instances=instances,
            view_type=view_type,
            depth_unit=depth_unit,
            depth_scale=depth_scale,
            convert_opengl_depth=convert_opengl_depth,
            opengl_near=opengl_near,
            opengl_far=opengl_far,
            denoise=denoise,
            denoise_method=denoise_method,
            denoise_params=denoise_params,
        )
        return (merged, [merged]) if return_per_frame else merged

    # 投影每帧点云到世界坐标系
    points_per_frame = []
    for i, instance in enumerate(instances):
        pts = instance.project_to_world_coordinates(
            view_type=view_type,
            depth_unit=depth_unit,
            depth_scale=depth_scale,
            convert_opengl_depth=convert_opengl_depth,
            opengl_near=opengl_near,
            opengl_far=opengl_far,
            denoise=denoise,
            denoise_method=denoise_method,
            denoise_params=denoise_params
        )
        if len(pts) == 0:
            continue
        points_per_frame.append(pts)
        center = pts.mean(axis=0)
        print(f"  帧 {i+1}: 点数={len(pts)}, 中心={center}")

    if len(points_per_frame) == 0:
        return np.empty((0, 3))
    if len(points_per_frame) == 1:
        return (points_per_frame[0], points_per_frame) if return_per_frame else points_per_frame[0]

    # 转为Open3D点云并下采样
    pcds = []
    for pts in points_per_frame:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        pcds.append(pcd)

    # 以第一帧为参考，逐帧ICP对齐
    ref = pcds[0]
    aligned_pcds = [ref]
    max_corr = voxel_size * icp_max_corr_factor
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter)

    for idx in range(1, len(pcds)):
        src = pcds[idx]
        result = o3d.pipelines.registration.registration_icp(
            src, ref, max_corr, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria
        )
        src_aligned = o3d.geometry.PointCloud(src)
        src_aligned.transform(result.transformation)
        aligned_pcds.append(src_aligned)
        print(f"  帧 {idx+1}: ICP fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.4f}")

    # 融合
    merged_points = np.vstack([np.asarray(p.points) for p in aligned_pcds])

    # 统计信息
    unit_label = "cm" if depth_unit == 'cm' else "m"
    print(f"\nICP融合后的点云: 总点数={len(merged_points)}")
    print(f"  X范围: [{merged_points[:,0].min():.4f}, {merged_points[:,0].max():.4f}] {unit_label}")
    print(f"  Y范围: [{merged_points[:,1].min():.4f}, {merged_points[:,1].max():.4f}] {unit_label}")
    print(f"  Z范围: [{merged_points[:,2].min():.4f}, {merged_points[:,2].max():.4f}] {unit_label}")

    if return_per_frame:
        aligned_np = [np.asarray(p.points) for p in aligned_pcds]
        return merged_points, aligned_np
    return merged_points