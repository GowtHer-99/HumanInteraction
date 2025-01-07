import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader
from datasets.reconstruction_feature_data import Reconstruction_Feature_Data
import cv2
import numpy as np
import pyrender
import trimesh
from pyrender import Mesh, Node
from trimesh.creation import axis

"======================================渲染32帧========================================="
# def visualize_smpl_model(sample_data, smpl):
#     # 从 sample_data 提取 SMPL 参数
#     betas = sample_data['betas'].reshape(-1, 10)  # 形状参数
#     pose = sample_data['pose'].reshape(-1, 72)    # 姿态参数
#     trans = sample_data['gt_cam_t'].reshape(-1, 3)  # 平移参数
    
#     # 转换为 torch.Tensor
#     betas = torch.from_numpy(betas).float().to(smpl.device)
#     pose = torch.from_numpy(pose).float().to(smpl.device)
#     # trans = torch.to(smpl.device)
    
#     # # 打印数据检查
#     # print("Betas shape:", betas.shape)
#     # print("Pose shape:", pose.shape)
#     # print("Trans shape:", trans.shape)
#     # print("Betas sample:\n", betas[0])
#     # print("Pose sample:\n", pose[0])
#     # print("Trans sample:\n", trans[0])

#     # 使用SMPL模型计算顶点和关节点
#     with torch.no_grad():
#         verts, joints = smpl(betas, pose, trans)

#     # # 数据检查：打印顶点信息
#     # print("Verts shape:", verts.shape)
#     # print("Verts data sample:\n", verts[0][:5])  # 打印前5个顶点数据

#     # # 如果顶点无效（全零或NaN），直接返回
#     # if torch.isnan(verts).any() or torch.isinf(verts).any():
#     #     print("Warning: Verts contain NaN or Inf!")
#     #     # return
#     # if torch.max(verts) < 1e-6:
#     #     print("Warning: Verts contain very small values!")
#     #     # return

#     # 转换为 numpy 数组
#     verts = verts.detach().cpu().numpy()

#     trans[:, 2] += 2.0  # 模型沿 z 轴向远处平移

#     # 使用 pyrender 渲染 3D 模型
#     scene = pyrender.Scene()

#     rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
#     for vert in verts:
#         mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
#         mesh.apply_transform(rotation)
#         mesh = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh)
    
#     # 设置相机参数
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

#     camera_pose = np.array([
#         [1.0, 0, 0, 0],
#         [0, 1.0, 0, 0],
#         [0, 0, 1.0, -0.5],  # 当前相机在 z=3.0 位置
#         [0, 0, 0, 1.0]
#     ])

#     scene.add(camera, pose=camera_pose)

#     # 添加光源
#     light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
#     scene.add(light, pose=camera_pose)

#     # 使用 OffscreenRenderer 渲染
#     renderer = pyrender.OffscreenRenderer(512, 512)
#     color, _ = renderer.render(scene)

#     cv2.imshow("SMPL Model Visualization", color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
"===================================================================================="

"======================================渲染2帧, 每帧一个人物模型========================================="
# def visualize_smpl_model(sample_data, smpl):
#     # 从 sample_data 提取 SMPL 参数
#     betas = sample_data['betas'].reshape(-1, 10)  # 形状参数
#     pose = sample_data['pose'].reshape(-1, 72)    # 姿态参数
#     trans = sample_data['gt_cam_t'].reshape(-1, 3)  # 平移参数
    
#     # 转换为 torch.Tensor
#     betas = torch.from_numpy(betas).float().to(smpl.device)
#     pose = torch.from_numpy(pose).float().to(smpl.device)
#     # trans = torch.to(smpl.device)
    
#     # # 打印数据检查
#     # print("Betas shape:", betas.shape)
#     # print("Pose shape:", pose.shape)
#     # print("Trans shape:", trans.shape)
#     # print("Betas sample:\n", betas[0])
#     # print("Pose sample:\n", pose[0])
#     # print("Trans sample:\n", trans[0])

#     # 使用SMPL模型计算顶点和关节点
#     with torch.no_grad():
#         verts, joints = smpl(betas, pose, trans)

#     # # 数据检查：打印顶点信息
#     # print("Verts shape:", verts.shape)
#     # print("Verts data sample:\n", verts[0][:5])  # 打印前5个顶点数据

#     # # 如果顶点无效（全零或NaN），直接返回
#     # if torch.isnan(verts).any() or torch.isinf(verts).any():
#     #     print("Warning: Verts contain NaN or Inf!")
#     #     # return
#     # if torch.max(verts) < 1e-6:
#     #     print("Warning: Verts contain very small values!")
#     #     # return

#     # 转换为 numpy 数组
#     verts = verts.detach().cpu().numpy()

#     trans[:, 2] += 2.0  # 模型沿 z 轴向远处平移

#     # 使用 pyrender 渲染 3D 模型
#     scene = pyrender.Scene()

#     rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

#     # 只渲染第一个或指定帧，避免多帧叠加
#     vert = verts[21]  # 渲染第0帧
#     mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
#     mesh.apply_transform(rotation)
#     mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(mesh)

#     vert = verts[20]  # 渲染第0帧
#     mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
#     mesh.apply_transform(rotation)
#     mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(mesh)
    
#     # 设置相机参数
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

#     camera_pose = np.array([
#         [1.0, 0, 0, 0],
#         [0, 1.0, 0, 0],
#         [0, 0, 1.0, -0.5],  # 当前相机在 z=3.0 位置
#         [0, 0, 0, 1.0]
#     ])

#     scene.add(camera, pose=camera_pose)

#     # 添加光源
#     light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
#     scene.add(light, pose=camera_pose)

#     # 使用 OffscreenRenderer 渲染
#     renderer = pyrender.OffscreenRenderer(512, 512)
#     color, _ = renderer.render(scene)

#     cv2.imshow("SMPL Model Visualization", color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
"============================================================================================="

"======================================渲染smpl原始模型========================================"
# def visualize_smpl_model(sample_data, smpl):
#     # 生成零向量的 betas 和 pose
#     betas = torch.zeros(1, 10).to(smpl.device)
#     pose = torch.zeros(1, 72).to(smpl.device)
#     trans = torch.zeros(1, 3).to(smpl.device)

#     # 使用SMPL模型计算顶点和关节点
#     with torch.no_grad():
#         verts, joints = smpl(betas, pose, trans)
#     # 转换为 numpy 数组
#     verts = verts.detach().cpu().numpy()
#     scene = pyrender.Scene()

#     # 只渲染第一个或指定帧，避免多帧叠加
#     vert = verts[0]  # 渲染第0帧
#     mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
#     mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(mesh)
    
#     # 设置相机参数
#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

#     camera_pose = np.array([
#         [1.0, 0, 0, 0],
#         [0, 1.0, 0, -0.2],
#         [0, 0, 1.0, 2.5],  # 当前相机在 z=3.0 位置
#         [0, 0, 0, 1.0]
#     ])

#     scene.add(camera, pose=camera_pose)

#     # 添加光源
#     light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
#     scene.add(light, pose=camera_pose)

#     # 使用 OffscreenRenderer 渲染
#     renderer = pyrender.OffscreenRenderer(512, 512)
#     color, _ = renderer.render(scene)

#     cv2.imshow("SMPL Model Visualization", color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
"========================================================================================="

"========================== 渲染可调节smpl原始模型 键盘控制姿态==============================="
def visualize_smpl_model(smpl):
    # 初始参数
    betas = torch.zeros(1, 10).to(smpl.device)
    pose = torch.zeros(1, 72).to(smpl.device)
    trans = torch.zeros(1, 3).to(smpl.device)
    step_size = 0.5  # 每次调整的步长

    # 渲染函数
    def render_model(betas, pose):
        with torch.no_grad():
            verts, joints = smpl(betas, pose, trans)
        verts = verts.detach().cpu().numpy()

        scene = pyrender.Scene()
        vert = verts[0]
        mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

        # 相机参数
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, -0.2],  # 上移一些
            [0, 0, 1.0, 2.5],   # 拉远镜头
            [0, 0, 0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # 光源
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=camera_pose)

        # 渲染
        renderer = pyrender.OffscreenRenderer(512, 512)
        color, _ = renderer.render(scene)
        cv2.imshow("SMPL Model Visualization", color)

    render_model(betas, pose)

    while True:
        key = cv2.waitKey(0)
        
        # 退出键
        if key == 27:  # 按ESC退出
            break
        
        # 调整形态 (betas)
        if key == ord('w'):  # 增加身高
            betas[0, 0] += step_size
        elif key == ord('s'):  # 减小身高
            betas[0, 0] -= step_size
        elif key == ord('a'):  # 增加胖瘦
            betas[0, 1] += step_size
        elif key == ord('d'):  # 减小胖瘦
            betas[0, 1] -= step_size
        
        # 调整姿态 (pose)
        if key == ord('i'):  # 头部向上
            pose[0, 45] += step_size
        elif key == ord('k'):  # 头部向下
            pose[0, 45] -= step_size
        elif key == ord('u'):  # 正向旋转手腕
            pose[0, 60] += step_size
        elif key == ord('j'):  # 逆向旋转手腕
            pose[0, 60] -= step_size
        elif key == ord('y'):  # 正向旋转手腕
            pose[0, 63] += step_size
        elif key == ord('h'):  # 逆向旋转手腕
            pose[0, 63] -= step_size
        elif key == ord('t'):  # 正向旋转腿
            pose[0, 3] += step_size
        elif key == ord('g'):  # 逆向旋转腿
            pose[0, 3] -= step_size
        elif key == ord('r'):  # 正向旋转腿
            pose[0, 6] += step_size
        elif key == ord('f'):  # 逆向旋转腿
            pose[0, 6] -= step_size
        elif key == ord('o'):  # 正向旋转肘部
            pose[0, 56] += step_size
        elif key == ord('l'):  # 逆向旋转肘部
            pose[0, 56] -= step_size
        elif key == ord('p'):  # 正向旋转肘部
            pose[0, 59] += step_size
        elif key == ord(';'):  # 逆向旋转肘部
            pose[0, 59] -= step_size

        # 重新渲染
        render_model(betas, pose)

    cv2.destroyAllWindows()

def print_joint_connections(smpl):
    kintree_table = smpl.kintree_table  # 直接使用 kintree_table
    print("Joint Connections (Child -> Parent):")
    for child, parent in enumerate(kintree_table[0]):
        print(f"Joint {child} -> Parent Joint {parent}")

def main(**args):
    seed = 7
    g = set_seed(seed)

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
    
    # 从子数据集中提取样本
    sample_data = train_dataset.datasets[0].create_data(index=0)
    # visualize_smpl_model(sample_data, smpl)
    visualize_smpl_model(smpl)
    print("关节连接关系：")
    print_joint_connections(smpl)

if __name__ == "__main__":
    args = parse_config()

    main(**args)
