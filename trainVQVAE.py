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
import torch.nn.functional as F
from model.MotionVQVAE import MotionVQVAE
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from torchvision.utils import make_grid

def train_motion_vqvae(model, train_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        train_res_recon_error = []
        train_res_perplexity = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        for i, data in enumerate(train_loader):
            # 提取姿态数据用于训练 VQ-VAE
            data = data['pose'].to(device)  # 使用 SMPL pose 数据
            optimizer.zero_grad()

            B, T, J, D = data.shape 
            # 将时间维度 T 合并到批次维度 B
            data = data.view(B * T, J, D)  # reshape 成 (B*T, J, D)

            # 前向传播
            vq_loss, data_recon, perplexity = model(data)

            # 计算重建损失
            recon_error = F.mse_loss(data_recon, data)
            loss = recon_error + vq_loss
            loss.backward()

            # 更新模型参数
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            progress_bar.set_postfix({
                'Recon Loss': f"{np.mean(train_res_recon_error[-10:]):.4f}",
                'Perplexity': f"{np.mean(train_res_perplexity[-10:]):.4f}"
            })
            progress_bar.update(1)

            # 显示最近10次的重建损失和困惑度
            # tqdm.write(f"Iter {i + 1}/{len(train_loader)}, Recon Loss: {np.mean(train_res_recon_error[-10:]):.4f}, Perplexity: {np.mean(train_res_perplexity[-10:]):.4f}")

            # if (i + 1) % 100 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}]")
            #     print(f"Reconstruction Loss: {np.mean(train_res_recon_error[-100:])}")
            #     print(f"Perplexity: {np.mean(train_res_perplexity[-100:])}\n")
        
        progress_bar.close()
        print(f"Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}]")
        print(f"Reconstruction Loss: {np.mean(train_res_recon_error[-100:])}")
        print(f"Perplexity: {np.mean(train_res_perplexity[-100:])}\n")   

    plot_training_progress(train_res_recon_error, train_res_perplexity)
        # 保存最终模型
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'recon_loss': np.mean(train_res_recon_error),
    #     'perplexity': np.mean(train_res_perplexity),
    # }, os.path.join(save_dir, 'motion_vqvae_final.pth'))
    # print(f"Final model saved at epoch {epoch+1}")

def visualize_smpl_model(sample_data, smpl):
    # 从 sample_data 提取 SMPL 参数
    betas = sample_data['betas'].reshape(-1, 10)  # 形状参数
    pose = sample_data['pose'].reshape(-1, 72)    # 姿态参数
    trans = sample_data['gt_cam_t'].reshape(-1, 3).to(smpl.device)  # 平移参数
    
    # 转换为 torch.Tensor
    betas = torch.from_numpy(betas).float().to(smpl.device)

    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float().to(smpl.device)
    else:
        pose = pose.float().to(smpl.device)
    # trans.to(smpl.device)
    
    # # 打印数据检查
    print("Betas shape:", betas.shape)
    print("Pose shape:", pose.shape)
    print("Trans shape:", trans.shape)
    print("Betas sample:\n", betas[0])
    print("Pose sample:\n", pose[0])
    print("Trans sample:\n", trans[0])

    # 使用SMPL模型计算顶点和关节点
    with torch.no_grad():
        verts, joints = smpl(betas, pose, trans)

    # # 数据检查：打印顶点信息
    # print("Verts shape:", verts.shape)
    # print("Verts data sample:\n", verts[0][:5])  # 打印前5个顶点数据

    # # 如果顶点无效（全零或NaN），直接返回
    # if torch.isnan(verts).any() or torch.isinf(verts).any():
    #     print("Warning: Verts contain NaN or Inf!")
    #     # return
    # if torch.max(verts) < 1e-6:
    #     print("Warning: Verts contain very small values!")
    #     # return

    # 转换为 numpy 数组
    verts = verts.detach().cpu().numpy()

    # trans[:, 2] += 2.0  # 模型沿 z 轴向远处平移

    # 使用 pyrender 渲染 3D 模型
    scene = pyrender.Scene()

    rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    for vert in verts:
        mesh = trimesh.Trimesh(vertices=vert, faces=smpl.faces)
        mesh.apply_transform(rotation)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    
    # 设置相机参数
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    camera_pose = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, -0.5],  # 当前相机在 z=3.0 位置
        [0, 0, 0, 1.0]
    ])

    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=camera_pose)

    # 使用 OffscreenRenderer 渲染
    renderer = pyrender.OffscreenRenderer(512, 512)
    color, _ = renderer.render(scene)

    cv2.imshow("SMPL Model Visualization", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 绘制损失和困惑度曲线
def plot_training_progress(recon_error, perplexity, save_path='./vaeoutput/pic/training_progress.png'):
    recon_error_smooth = savgol_filter(recon_error, 201, 7)
    perplexity_smooth = savgol_filter(perplexity, 201, 7)

    f = plt.figure(figsize=(16, 8))
    
    # 绘制损失曲线
    ax = f.add_subplot(1, 2, 1)
    ax.plot(recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('Iteration')
    
    # 绘制困惑度曲线
    ax = f.add_subplot(1, 2, 2)
    ax.plot(perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('Iteration')

    # 自动创建保存路径
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 保存图片而不是显示
    plt.savefig(save_path, dpi=300)  # 保存为PNG格式，dpi为300
    print(f"Training progress plot saved at {save_path}")
    plt.close() 

# 模型评估与可视化
def evaluate_model(model, validation_set, validation_loader, smpl, device):
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            # 只选一个样本
            sample_data = validation_set.datasets[0].create_data(index=0)
            # 提取姿态数据
            pose = data['pose'].to(device)
            B, T, J, D = pose.shape
            pose = pose.view(B * T, J, D)

            pose_without_recontruct = pose.view(B, T, -1)[0]
            # 可视化原始数据
            sample_data['pose'] = pose_without_recontruct

            print("Visualizing Original Sample")
            visualize_smpl_model(sample_data, smpl)

            # 编码 + 量化 + 解码
            vq_output_eval = model.pre_vq_conv(model.encoder(pose))
            _, quantized, _, _ = model.vq_vae(vq_output_eval)
            reconstructions = model.decoder(quantized)
            
            reconstructed_pose = reconstructions.view(B, T, -1)[0]

            print(pose.shape)
            print(reconstructions.shape)
            print(reconstructions[0].shape)

            # # 可视化重建结果
            print("Visualizing Reconstructed Sample")
            sample_data['pose'] = reconstructed_pose
            visualize_smpl_model(sample_data, smpl)
            
def bone_length_loss(pred, target, bone_pairs):
    loss = 0.0
    
    for pair in bone_pairs:
        pred_bone = torch.norm(pred[:, :, pair[0]] - pred[:, :, pair[1]], dim=-1)
        target_bone = torch.norm(target[:, :, pair[0]] - target[:, :, pair[1]], dim=-1)
        loss += torch.mean((pred_bone - target_bone) ** 2)
    
    # 进行平均
    return loss / len(bone_pairs)

def velocity_loss(pred, target):
    # 计算相邻帧之间的关节速度差异
    pred_velocity = torch.diff(pred, dim=1)  # 生成的姿态速度
    target_velocity = torch.diff(target, dim=1)  # 真实姿态速度
    
    # 计算速度差的 L2 损失
    loss = torch.mean((pred_velocity - target_velocity) ** 2)
    
    return loss




def main(**args):
    seed = 7
    g = set_seed(seed)

    # 加载骨骼对和足部索引
    bone_pairs = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), 
        (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), 
        (9, 12), (9, 13), (9, 14), (12, 15), (13, 16),
        (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
        (20, 22), (21, 23)
    ]

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    out_dir = './vaeoutput'

    # Load loss function
    loss = LossLoader(smpl, device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, out_dir=out_dir, **args)

    # Create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)

    # 定义 Motion VQ-VAE 模型和优化器
    motion_vqvae = MotionVQVAE(
        num_hiddens=256,
        num_residual_layers=3,
        num_residual_hiddens=64,
        num_embeddings=1024,
        embedding_dim=128,
        commitment_cost=0.25
    ).to(device)

    optimizer = torch.optim.Adam(motion_vqvae.parameters(), lr=1e-4)

    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])
        
        # 定义 Motion VQ-VAE 模型和优化器
        # motion_vqvae = MotionVQVAE(
        #     num_hiddens=256,
        #     num_residual_layers=3,
        #     num_residual_hiddens=64,
        #     num_embeddings=1024,
        #     embedding_dim=128,
        #     commitment_cost=0.25
        # ).to(device)

        # optimizer = torch.optim.Adam(motion_vqvae.parameters(), lr=1e-4)

        # 训练 Motion VQ-VAE
        train_motion_vqvae(motion_vqvae, train_loader, optimizer, num_epoch, device)

        torch.save({
            'epoch': num_epoch,
            'model_state_dict': motion_vqvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(out_dir, 'motion_vqvae_final.pth'))

        print("Final model saved successfully.")

    elif mode == 'eval':
        # 评估模型（在训练完成后进行模型重建评估）
        test_dataset = dataset.load_testset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=batchsize, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
        # 加载刚刚保存的模型并直接评估
        checkpoint = torch.load(os.path.join(out_dir, 'motion_vqvae_final.pth'), map_location=device)
        motion_vqvae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1} for evaluation.")

        evaluate_model(motion_vqvae, test_dataset, test_loader, smpl, device)

    # # 从子数据集中提取样本并可视化
    # # sample_data = train_dataset.datasets[0].create_data(index=0)
    # # visualize_smpl_model(smpl)

if __name__ == "__main__":
    args = parse_config()

    main(**args)
