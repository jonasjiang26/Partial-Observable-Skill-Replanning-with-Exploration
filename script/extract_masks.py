#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mask_utils import MaskExtractor


def main():
    parser = argparse.ArgumentParser(
        description='从多物体mask图像中提取每个物体的mask并单独保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_masks.py mask.png
  python extract_masks.py mask.png --output_dir ./masks --format png
  python extract_masks.py mask.png --background 0
        """
    )
    
    parser.add_argument('input_mask', type=str, help='输入的mask图像路径')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='输出目录（默认：输入文件所在目录）')
    parser.add_argument('--format', '-f', type=str, default='png',
                       choices=['png', 'jpg', 'jpeg'],
                       help='输出图像格式（默认: png）')
    parser.add_argument('--background', '-b', type=int, default=0,
                       help='背景像素值（默认: 0）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_mask):
        print(f"错误: 输入文件不存在: {args.input_mask}")
        return
    
    try:
        # 创建mask提取器实例
        extractor = MaskExtractor(background_value=args.background)
        
        # 加载mask图像
        print(f"正在加载mask图像: {args.input_mask}")
        extractor.load(args.input_mask)
        
        # 提取所有物体
        print("正在提取物体...")
        objects = extractor.extract(binary=True)
        
        if len(objects) == 0:
            print(f"警告: 在图像中未找到任何物体（除了背景值 {args.background}）")
            return
        
        print(f"找到 {len(objects)} 个物体:")
        for obj in objects:
            print(f"  - 物体ID {obj.object_id}: 面积={obj.area} 像素, 边界框={obj.bbox}, 中心={obj.center}")
        
        # 设置输出目录
        if args.output_dir is None:
            args.output_dir = os.path.dirname(args.input_mask) or '.'
        
        # 获取输入文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(args.input_mask))[0]
        
        # 保存所有物体的mask
        print(f"\n正在保存到目录: {args.output_dir}")
        saved_paths = extractor.save_all(
            output_dir=args.output_dir,
            base_name=base_name,
            format=args.format,
            binary=True
        )
        
        print(f"\n总共提取并保存了 {len(saved_paths)} 个物体的mask")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
