# Copyright (c) OpenMMLab. All rights reserved.
"""
旋转目标检测 - 单张图像推理演示脚本

功能：加载旋转目标检测模型，对单张图像进行推理，并可视化检测结果。
支持通过命令行参数指定模型配置、权重文件、输出路径等。

使用示例:
    python demo/image_demo.py demo/demo.jpg configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py \
        checkpoints/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth \
        --out-file demo_result.jpg
"""

# 导入命令行参数解析模块
from argparse import ArgumentParser

# 导入 mmcv 多媒体处理库（图像读写、转换等）
import mmcv
# 从 mmdet 导入推理和模型初始化接口
from mmdet.apis import inference_detector, init_detector

# 从 mmrotate 导入可视化器注册表和模块注册函数
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules


def parse_args():
    """
    解析命令行参数

    返回:
        argparse.Namespace: 包含所有解析后的命令行参数的命名空间对象
            - img: 输入图像文件路径
            - config: 模型配置文件路径
            - checkpoint: 模型权重文件路径
            - out_file: 输出文件路径（可选，默认直接显示图像）
            - device: 推理所用设备（默认 'cuda:0'）
            - palette: 可视化调色板（默认 'dota'）
            - score_thr: 检测框置信度阈值（默认 0.3）
    """
    # 创建参数解析器
    parser = ArgumentParser()
    # 位置参数：输入图像文件路径
    parser.add_argument('img', help='Image file')
    # 位置参数：模型配置文件路径（如 configs/oriented_rcnn/xxx.py）
    parser.add_argument('config', help='Config file')
    # 位置参数：模型权重文件路径（.pth 文件）
    parser.add_argument('checkpoint', help='Checkpoint file')
    # 可选参数：输出文件路径，不指定时会在窗口中直接显示结果
    parser.add_argument('--out-file', default=None, help='Path to output file')
    # 可选参数：推理设备，默认使用第一块 GPU
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # 可选参数：可视化调色板，支持 dota、sar、hrsc 等数据集预定义调色板或随机调色板
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    # 可选参数：检测框置信度阈值，低于此值的检测结果将被过滤
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # 解析并返回参数
    args = parser.parse_args()
    return args


def main(args):
    """
    主函数：执行完整的单张图像推理与可视化流程

    流程:
        1. 注册 mmrotate 所有模块到对应注册表
        2. 根据配置文件和权重文件构建模型
        3. 初始化可视化器
        4. 对输入图像进行推理
        5. 可视化检测结果并保存或显示

    参数:
        args (argparse.Namespace): 命令行解析后的参数
    """
    # ========== 步骤1: 注册所有 mmrotate 模块 ==========
    # 将 mmrotate 中的所有模型、数据集等模块注册到 mmengine 的注册表中
    # 这样后续可以通过配置文件中的名称来构建对应的模块
    register_all_modules()

    # ========== 步骤2: 构建模型 ==========
    # 从配置文件和权重文件初始化旋转目标检测器
    # palette 参数用于指定可视化时的颜色方案
    # device 参数指定模型加载到哪个设备（CPU/GPU）
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # ========== 步骤3: 初始化可视化器 ==========
    # 根据模型配置中的 visualizer 配置构建可视化器实例
    # VISUALIZERS 是 mmengine 的注册表，通过 build 方法根据配置字典创建对象
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # 将数据集的元信息（如类别名称、调色板等）传递给可视化器
    # dataset_meta 在 init_detector 中从模型权重文件加载
    visualizer.dataset_meta = model.dataset_meta

    # ========== 步骤4: 单张图像推理 ==========
    # 对输入图像进行旋转目标检测推理
    # result 包含检测框、类别标签和置信度分数等信息
    result = inference_detector(model, args.img)

    # ========== 步骤5: 可视化并显示/保存结果 ==========
    # 读取原始图像（OpenCV 默认读取为 BGR 格式）
    img = mmcv.imread(args.img)
    # 将图像从 BGR 色彩空间转换为 RGB 色彩空间
    # 可视化通常使用 RGB 格式以保证颜色正确显示
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    # 使用可视化器添加检测结果到图像上
    visualizer.add_datasample(
        'result',                    # 样本名称/标识符
        img,                         # 输入图像（RGB 格式）
        data_sample=result,          # 检测结果数据
        draw_gt=False,               # 不绘制真值标注（演示模式无真值）
        show=args.out_file is None,  # 如果没有指定输出文件，则在窗口中显示
        wait_time=0,                 # 显示窗口的等待时间（0 表示一直等待按键）
        out_file=args.out_file,      # 输出文件路径，指定后结果将保存到此路径
        pred_score_thr=args.score_thr)  # 预测框置信度阈值，低于此值的框不显示


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 执行主函数
    main(args)