# 多帧深度图投影问题诊断指南

## 问题描述
从两帧深度图像提取的同一实例投影到世界坐标后，两个点云距离很远。

## 可能的原因分析

### 1. **摄像机外参错误（最可能）**
- **问题**: `_CAMERA_EXTRINSICS['agentview']` 中硬编码的摄像机位置和姿态可能不准确
- **症状**: 所有帧都投影到错误的全局位置
- **检查方法**:
  ```python
  # 运行诊断脚本查看摄像机变换矩阵
  python script/debug_projection.py --load_path data.h5 --frame_ids 0 1
  ```

### 2. **相机坐标系变换理解错误**
- **问题**: T_w_cam 的定义可能理解反了
- **当前代码**:
  ```python
  # T_w_cam 表示：世界坐标系 -> 摄像机坐标系的变换
  # 但变换矩阵中存储的是摄像机在世界坐标系中的位置和姿态
  ```
- **可能的问题**: 旋转矩阵应该是 R_cam_w（摄像机到世界），而不是 R_w_cam

### 3. **OpenGL深度值转换错误**
- **问题**: OpenGL深度缓冲值的转换公式可能不正确
- **检查**: 深度值是否已经是线性的（不需要转换）

### 4. **Eye-in-hand摄像机位置变化**
- **如果使用eye_in_hand摄像机**: 每帧的ee_pos和ee_ori都在变化
- **症状**: 即使是同一物体，从不同的ee位置看，投影结果会有偏移
- **预期行为**: 这是正常的！需要通过摄像机外参来对齐

### 5. **坐标系定义不一致**
- **问题**: 世界坐标系的定义可能不清楚
- **需要检查**:
  - 世界坐标系原点在哪里？
  - X、Y、Z轴的方向？
  - 旋转轴是否为身体固定坐标系还是欧拉角？

## 诊断步骤

### 步骤1：运行诊断脚本
```bash
cd /home/jing/POSRE
python script/debug_projection.py \
  --load_path <path_to_h5_file> \
  --frame_ids 0 1 \
  --object_id 118 \
  --view_type agentview
```

### 步骤2：检查输出信息

#### 2.1 摄像机变换矩阵
- 所有帧的 T_cam_w 应该**相同**（对于agentview）
- 如果不同，说明摄像机位置在变化

#### 2.2 点云范围
- 检查 X, Y, Z 的平均值是否相差很远
- 如果相差很远，说明投影出现了问题

#### 2.3 帧之间的距离
- 计算点云中心之间的距离
- 如果距离远超过物体大小，说明变换有问题

### 步骤3：检查摄像机内参
```
内参矩阵 K:
  fx = XXX.XX, fy = XXX.XX (像素/米)
  cx = XXX.XX, cy = XXX.XX (像素)
```

- fx, fy 应该在合理范围内（通常100-500像素/米）
- 如果值过大或过小，可能视场角参数设置错误

## 常见问题及解决方案

### 问题A：摄像机变换矩阵不同
**症状**: 每帧的 T_cam_w 都不同

**原因**: 
- agentview摄像机不是固定的（不应该发生）
- eye_in_hand摄像机，ee_pos/ee_ori每帧都不同（正常）

**解决方案**:
- 对于agentview: 检查摄像机是否真的固定，如果固定则使用第一帧的变换
- 对于eye_in_hand: 这是正常的，继续诊断

### 问题B：点云中心相差很远
**症状**: 同一物体不同帧投影后中心相差超过0.1米

**原因最可能的排序**:
1. 摄像机外参错误（位置、方向）
2. 旋转矩阵的行列对应错误
3. 深度值单位转换错误

**检查清单**:
```python
# 检查1: 旋转矩阵是否正交
T = get_camera_to_world_transform()
R = T[:3, :3]
# det(R) 应该接近1
# R @ R.T 应该接近单位矩阵
print(f"det(R) = {np.linalg.det(R)}")
print(f"R @ R.T =\n{R @ R.T}")

# 检查2: 深度值范围
# 应该在0-10米之间
# 如果超出这个范围，可能需要调整 opengl_near 和 opengl_far

# 检查3: 投影公式
# 使用已知的3D点，反投影回2D，看是否正确
```

### 问题C：点云距离过远，无法合并
**症状**: 运行 merge_multi_frame_pointclouds 后点云分布不集中

**解决方案**:
1. 首先运行诊断脚本定位问题
2. 根据具体原因修复：
   - 修改 `set_camera_extrinsic()` 参数
   - 修复坐标系变换公式
   - 调整 OpenGL 深度转换参数
3. 重新测试

## 坐标系变换公式详解

### 当前代码的理解
```
世界坐标系 ----T_w_cam----> 摄像机坐标系
(World Frame)              (Camera Frame)

p_cam = T_w_cam @ p_world
```

### 但通常的定义
```
相机坐标系 ----T_cam_w----> 世界坐标系
(Camera Frame)            (World Frame)

p_world = T_cam_w @ p_cam
```

### 代码中的实现
```python
# 当前代码：
T_w_cam = get_camera_extrinsic()  # 这个变量名容易误导
T_cam_w = np.linalg.inv(T_w_cam)  # 通过求逆得到

# 应该改为更清晰的命名
```

## 完整的诊断清单

- [ ] 运行诊断脚本，收集详细日志
- [ ] 检查摄像机变换矩阵是否一致（对于固定摄像机）
- [ ] 检查点云中心距离
- [ ] 验证旋转矩阵的正交性
- [ ] 检查深度值范围是否合理
- [ ] 对比原始2D mask位置和3D投影结果
- [ ] 如果是eye_in_hand，检查ee_pos/ee_ori是否正确加载
- [ ] 手动测试一个已知的3D点，看投影是否正确

## 需要提供的信息

当您运行诊断脚本后，请收集并提供以下信息：

1. **诊断脚本的完整输出**
2. **生成的可视化图像** (`debug_multiframe_pointcloud.png`)
3. **您知道的以下信息**:
   - 物体在世界坐标系中的实际位置/大小
   - 摄像机与世界坐标系原点的真实距离
   - 使用的摄像机类型（agentview/eye_in_hand）
   - 深度图的实际范围和单位

## 快速修复建议

### 如果摄像机外参是问题源
```python
# 在 mask_viewer.py 的 main() 函数中添加
from utils.depth_projection_utils import set_camera_extrinsic

# 根据您的实际摄像机参数更新
set_camera_extrinsic(
    'agentview',
    pos=np.array([x, y, z]),  # 真实位置
    quat=np.array([w, x, y, z])  # 真实四元数
)
```

### 如果是坐标系变换问题
```python
# 检查 get_camera_to_world_transform 中的变换方向
# 可能需要改为：
T_cam_w = T_w_cam  # 而不是求逆
# 或者
T_cam_w = np.linalg.inv(T_w_cam)  # 具体取决于变量定义
```

---

**建议**: 先运行诊断脚本，根据输出结果反馈给我具体的数值，这样可以更快定位问题。
