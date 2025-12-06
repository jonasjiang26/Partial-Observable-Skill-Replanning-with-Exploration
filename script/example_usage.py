#!/usr/bin/env python3
"""
面向对象API使用示例
"""

import sys
import os

# 添加utils目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mask_utils import MaskExtractor, ObjectMask


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("示例1: 基本使用")
    print("=" * 50)
    
    # 创建mask提取器
    extractor = MaskExtractor(background_value=0)
    
    # 加载mask图像（请替换为你的mask图像路径）
    mask_path = 'your_mask_image.png'
    if not os.path.exists(mask_path):
        print(f"提示: 请将 {mask_path} 替换为实际的mask图像路径")
        return
    
    extractor.load(mask_path)
    
    # 提取所有物体
    objects = extractor.extract(binary=True)
    
    print(f"找到 {len(objects)} 个物体:\n")
    
    # 访问每个物体实例的属性
    for obj in objects:
        print(f"物体ID: {obj.object_id}")
        print(f"  - 面积: {obj.area} 像素")
        print(f"  - 边界框: {obj.bbox} (x_min, y_min, x_max, y_max)")
        print(f"  - 中心点: {obj.center} (x, y)")
        print(f"  - Mask形状: {obj.shape}")
        print()


def example_save_individual():
    """单独保存每个物体示例"""
    print("=" * 50)
    print("示例2: 单独保存每个物体")
    print("=" * 50)
    
    extractor = MaskExtractor(background_value=0)
    extractor.load('your_mask_image.png')
    extractor.extract()
    
    # 为每个物体单独保存
    for obj in extractor:
        output_path = f'object_{obj.object_id}_mask.png'
        obj.save(output_path, binary=True)
        print(f"已保存物体 {obj.object_id} 到: {output_path}")


def example_batch_save():
    """批量保存示例"""
    print("=" * 50)
    print("示例3: 批量保存所有物体")
    print("=" * 50)
    
    extractor = MaskExtractor(background_value=0)
    extractor.load('your_mask_image.png')
    extractor.extract()
    
    # 批量保存
    saved_paths = extractor.save_all(
        output_dir='./output_masks',
        base_name='mask',
        format='png',
        binary=True
    )
    
    print(f"\n总共保存了 {len(saved_paths)} 个文件")


def example_get_specific_object():
    """获取特定物体示例"""
    print("=" * 50)
    print("示例4: 获取特定物体")
    print("=" * 50)
    
    extractor = MaskExtractor(background_value=0)
    extractor.load('your_mask_image.png')
    extractor.extract()
    
    # 获取ID为1的物体
    obj = extractor.get_object_by_id(1)
    if obj:
        print(f"找到物体ID 1:")
        print(f"  - 面积: {obj.area}")
        print(f"  - 边界框: {obj.bbox}")
        
        # 获取裁剪后的mask（只包含物体区域）
        cropped = obj.get_cropped_mask(binary=True, padding=10)
        print(f"  - 裁剪后mask形状: {cropped.shape}")
    else:
        print("未找到物体ID 1")


def example_from_array():
    """从numpy数组加载示例"""
    print("=" * 50)
    print("示例5: 从numpy数组加载")
    print("=" * 50)
    
    import numpy as np
    
    # 创建一个示例mask数组
    # 0是背景，1和2是两个物体
    mask_array = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
    ])
    
    extractor = MaskExtractor(background_value=0)
    extractor.load_from_array(mask_array)
    objects = extractor.extract()
    
    print(f"从数组中找到 {len(objects)} 个物体:")
    for obj in objects:
        print(f"  物体ID {obj.object_id}: 面积={obj.area}, 边界框={obj.bbox}")


def example_iterate_objects():
    """迭代物体示例"""
    print("=" * 50)
    print("示例6: 迭代所有物体")
    print("=" * 50)
    
    extractor = MaskExtractor(background_value=0)
    extractor.load('your_mask_image.png')
    extractor.extract()
    
    # 方式1: 直接迭代extractor
    print("方式1: 直接迭代extractor")
    for obj in extractor:
        print(f"物体 {obj.object_id}: 面积={obj.area}")
    
    # 方式2: 使用objects属性
    print("\n方式2: 使用objects属性")
    for obj in extractor.objects:
        print(f"物体 {obj.object_id}: 面积={obj.area}")
    
    # 方式3: 使用len()获取数量
    print(f"\n总共有 {len(extractor)} 个物体")


if __name__ == '__main__':
    print("面向对象API使用示例\n")
    
    # 运行示例（取消注释你想运行的示例）
    # example_basic_usage()
    # example_save_individual()
    # example_batch_save()
    # example_get_specific_object()
    example_from_array()  # 这个示例不需要实际文件
    # example_iterate_objects()
    
    print("\n提示: 取消注释其他示例函数来查看更多用法")

