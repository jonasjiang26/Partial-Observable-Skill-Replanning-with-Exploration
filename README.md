# POTRe - 物体分割Mask提取工具

这个工具用于从包含多个物体分割mask的图片中，提取每个物体的mask并单独保存。采用面向对象设计，将物体作为实例，分割出的掩码作为实例的属性。

## 功能

- 从多物体mask图像中自动识别所有物体
- 将每个物体的mask提取为独立的二值图像
- 面向对象设计：物体作为实例，mask作为实例属性
- 支持获取物体的面积、边界框、中心点等属性
- 支持自定义输出目录和格式

## 使用方法

### 命令行工具

```bash
# 基本用法（输出到输入文件所在目录）
python script/extract_masks.py <input_mask_image>

# 指定输出目录
python script/extract_masks.py mask.png --output_dir ./masks

# 指定输出格式
python script/extract_masks.py mask.png --format png

# 自定义背景值
python script/extract_masks.py mask.png --background 0
```

### Python API（面向对象方式）

#### 基本使用

```python
from utils.mask_utils import MaskExtractor

# 创建mask提取器实例
extractor = MaskExtractor(background_value=0)

# 加载mask图像
extractor.load('mask.png')

# 提取所有物体实例
objects = extractor.extract(binary=True)

# 访问物体实例
for obj in objects:
    print(f"物体ID: {obj.object_id}")
    print(f"面积: {obj.area} 像素")
    print(f"边界框: {obj.bbox}")
    print(f"中心点: {obj.center}")
    
    # 保存单个物体的mask
    obj.save(f'object_{obj.object_id}.png', binary=True)
```

#### 批量保存

```python
from utils.mask_utils import MaskExtractor

extractor = MaskExtractor(background_value=0)
extractor.load('mask.png')
extractor.extract()

# 批量保存所有物体
saved_paths = extractor.save_all(
    output_dir='./masks',
    base_name='mask',
    format='png',
    binary=True
)
```

#### 从numpy数组加载

```python
import numpy as np
from utils.mask_utils import MaskExtractor

# 假设你有一个numpy数组形式的mask
mask_array = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 2, 2, 0]])

extractor = MaskExtractor(background_value=0)
extractor.load_from_array(mask_array)
objects = extractor.extract()

# 访问特定物体
obj = extractor.get_object_by_id(1)
if obj:
    print(f"物体1的面积: {obj.area}")
    cropped = obj.get_cropped_mask(binary=True, padding=5)
```

#### 物体实例属性和方法

每个 `ObjectMask` 实例包含以下属性和方法：

**属性：**
- `object_id`: 物体ID
- `mask`: 原始mask（物体区域保持原ID值，其他为0）
- `binary_mask`: 二值mask（物体区域为255，其他为0）
- `shape`: mask的形状 (height, width)
- `area`: 物体面积（像素数量）
- `bbox`: 边界框 (x_min, y_min, x_max, y_max)
- `center`: 中心点坐标 (x, y)

**方法：**
- `save(output_path, binary=True)`: 保存mask到文件
- `get_cropped_mask(binary=True, padding=0)`: 获取裁剪后的mask（只包含物体区域）

## 类说明

### MaskExtractor 类

Mask提取器类，用于从多物体mask图像中提取所有物体实例。

**主要方法：**
- `load(mask_path)`: 从文件加载mask图像
- `load_from_array(mask)`: 从numpy数组加载mask
- `extract(binary=True)`: 提取所有物体实例
- `save_all(output_dir, base_name, format, binary)`: 批量保存所有物体
- `get_object_by_id(object_id)`: 根据ID获取物体实例

### ObjectMask 类

物体mask实例类，表示一个物体的分割掩码。

**主要属性：**
- `object_id`: 物体ID
- `mask`: 原始mask
- `binary_mask`: 二值mask
- `area`: 物体面积
- `bbox`: 边界框
- `center`: 中心点

## 参数说明

### 命令行参数

- `input_mask`: 输入的mask图像路径（必需）
- `--output_dir, -o`: 输出目录（可选，默认为输入文件所在目录）
- `--format, -f`: 输出图像格式，可选 png/jpg/jpeg（默认: png）
- `--background, -b`: 背景像素值（默认: 0）

### Python API参数

- `background_value`: 背景像素值，默认为0
- `binary`: 是否生成/保存二值mask，默认为True

## 输出

脚本会为每个物体生成一个独立的mask文件，命名格式为：
`<原文件名>_object_<物体ID>.<格式>`

例如：`mask.png` 会生成 `mask_object_1.png`, `mask_object_2.png` 等。

## 依赖

- Python 3.6+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## 示例

```python
from utils.mask_utils import MaskExtractor

# 创建提取器
extractor = MaskExtractor(background_value=0)

# 加载并提取
extractor.load('multi_object_mask.png')
objects = extractor.extract()

# 遍历所有物体实例
for obj in extractor:  # 支持迭代
    print(f"物体 {obj.object_id}:")
    print(f"  - 面积: {obj.area} 像素")
    print(f"  - 边界框: {obj.bbox}")
    print(f"  - 中心: {obj.center}")
    
    # 保存单个物体
    obj.save(f'object_{obj.object_id}.png')

# 或者批量保存
extractor.save_all('./output', base_name='mask', format='png')
```
