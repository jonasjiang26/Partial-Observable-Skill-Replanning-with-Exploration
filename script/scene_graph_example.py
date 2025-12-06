#!/usr/bin/env python3
"""
场景图分析示例：分析多个3D实例之间的空间关系
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mask_utils import MaskExtractor
from utils.depth_projection_utils import ThreeDInstance
from utils.scene_graph_utils import SceneGraphAnalyzer


def main():
    parser = argparse.ArgumentParser(description='分析场景中多个物体的空间关系')
    parser.add_argument('--load_path', type=str, required=True, help='h5文件路径')
    parser.add_argument('--frame_id', type=int, required=True, help='帧ID')
    parser.add_argument('--object_ids', type=int, nargs='+', required=True, 
                       help='要分析的物体ID列表（至少2个）')
    parser.add_argument('--view_type', type=str, default='agentview',
                       choices=['agentview', 'eye_in_hand'],
                       help='使用的视图类型（默认agentview）')
    args = parser.parse_args()
    
    if len(args.object_ids) < 2:
        print("错误: 至少需要2个物体ID")
        return
    
    # 加载数据
    background_value = [5, 84, 20, 24, 25, 53, 29, 31, 39, 45, 43, 63, 48, 49, 51, 53, 57, 58, 47, 65, 66, 71, 72, 70, 73, 75, 78]
    viewer = MaskExtractor(background_value=background_value)
    viewer.load(args.load_path, args.frame_id)
    viewer.extract(binary=True, view_type=args.view_type)
    
    # 创建3D实例列表
    instances_3d = []
    for obj_id in args.object_ids:
        obj_2d = viewer.get_object_by_id(obj_id)
        if obj_2d is None:
            print(f"警告: 未找到物体ID {obj_id}，跳过")
            continue
        
        # 创建3D实例
        obj_3d = ThreeDInstance(
            object_id=obj_2d.object_id,
            original_mask=obj_2d.original_mask,
            binary_mask=obj_2d.binary_mask,
            depth_agentview=obj_2d.depth_agentview,
            depth_eye_in_hand=obj_2d.depth_eye_in_hand,
            depth_agentview_full=viewer.depth_agentview,
            depth_eye_in_hand_full=viewer.depth_eye_in_hand,
            ee_pos=viewer.ee_pos,
            ee_ori=viewer.ee_ori
        )
        instances_3d.append(obj_3d)
    
    if len(instances_3d) < 2:
        print("错误: 至少需要2个有效的3D实例")
        return
    
    # 创建场景图分析器
    analyzer = SceneGraphAnalyzer(instances_3d)
    
    # 打印场景图
    print(f"\n分析 {len(instances_3d)} 个物体的空间关系（视图类型: {args.view_type}）")
    analyzer.print_scene_graph(view_type=args.view_type)
    
    # 详细分析每对物体的关系
    print(f"\n详细关系分析:")
    print("=" * 60)
    for i, obj1 in enumerate(instances_3d):
        for j, obj2 in enumerate(instances_3d):
            if i < j:  # 避免重复
                relations = analyzer.get_spatial_relations(obj1, obj2, view_type=args.view_type)
                true_relations = [rel for rel, value in relations.items() if value]
                if true_relations:
                    print(f"物体 {obj1.object_id} 与 物体 {obj2.object_id}:")
                    print(f"  关系: {', '.join(true_relations)}")
                    
                    # 显示包围盒信息
                    bbox1_min, bbox1_max = analyzer._get_world_bbox(obj1, args.view_type)
                    bbox2_min, bbox2_max = analyzer._get_world_bbox(obj2, args.view_type)
                    print(f"  物体 {obj1.object_id} 包围盒: min={bbox1_min}, max={bbox1_max}")
                    print(f"  物体 {obj2.object_id} 包围盒: min={bbox2_min}, max={bbox2_max}")
                    print()


if __name__ == '__main__':
    main()

