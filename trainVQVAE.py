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
from utils.renderer_pyrd import Renderer
from scipy.signal import savgol_filter
from torchvision.utils import make_grid
from utils.module_utils import tensorborad_add_video_xyz
from torch.utils.tensorboard import SummaryWriter
from modules import MPJPE
from utils.rotation_conversions import *

def train_motion_vqvae(model, train_loader, testloader, lossca, smpl, optimizer, num_epochs, logger, device):
    model.train()
    testlossbest = 1000
    out_dir = './vaeoutput/bestres'
    for epoch in range(num_epochs):
        train_res_recon_error = []
        train_res_perplexity1 = []
        train_res_perplexity2 = []
        train_res_total_loss = []
        train_res_velocity = []
        train_res_smoothness = []
        train_res_comit_error = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        for i, data in enumerate(train_loader):
            # 提取姿态数据用于训练 VQ-VAE
            # pose = data['pose'].to(device)  # 使用 SMPL pose 数据
            pose6D = data['pose_6d'].to(device)
            
            # print("pose6d shape: ", pose6D.shape)

            optimizer.zero_grad()

            # B, T, P, D = pose6D.shape 
            # 将时间维度 T 合并到批次维度 B
            # pose6D = pose6D.view(B * T, P, D)  # reshape 成 (B*T, J, D)

            # 前向传播
            vq_loss, data_recon, perplexity1, perplexity2 = model(pose6D)
            # print(pose6D.shape)
            # vq_loss, data_recon, perplexity = model(pose6D)
            # print(data_recon.shape)
            # 计算重建损失
            # data_recon = data_recon.view(B, T, P, D)  # 恢复时间维度
            # pose6D = pose6D.view(B, T, P, D) 
            recon_error = F.mse_loss(data_recon, pose6D)
            smooth_loss = smoothness_loss(data_recon)
            vel_loss = velocity_loss(data_recon, pose6D)

            lambda_vel = epoch / 25  # 训练前期影响较小，后期逐渐增加
            lambda_smooth = epoch / 25
            vq_weight = max(0.20, 4 - epoch / 50 )  # 逐渐减少 vq_loss 的影响
            recon_weight = 8 - epoch / 40

            loss = recon_error * recon_weight + vq_weight * vq_loss + lambda_vel * vel_loss + lambda_smooth * smooth_loss
            loss.backward()

            # 更新模型参数
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_comit_error.append(vq_loss.item())
            train_res_perplexity1.append(perplexity1.item())
            train_res_velocity.append(vel_loss.item())  # 新增
            train_res_smoothness.append(smooth_loss.item())
            train_res_total_loss.append(loss.item())
            train_res_perplexity2.append(perplexity2.item())
            progress_bar.set_postfix({
                'Recon Loss': f"{np.mean(train_res_recon_error[-10:]):.4f}",
                'Comit Loss': f"{np.mean(train_res_comit_error[-10:]):.4f}",
                'Vel Loss': f"{np.mean(train_res_velocity[-10:]):.6f}",  # 新增
                'Smooth Loss': f"{np.mean(train_res_smoothness[-10:]):.6f}",
                'Perplexity1': f"{np.mean(train_res_perplexity1[-10:]):.4f}",
                'Perplexity2': f"{np.mean(train_res_perplexity2[-10:]):.4f}"
            })
            progress_bar.update(1)

            # 显示最近10次的重建损失和困惑度
            # tqdm.write(f"Iter {i + 1}/{len(train_loader)}, Recon Loss: {np.mean(train_res_recon_error[-10:]):.4f}, Perplexity: {np.mean(train_res_perplexity[-10:]):.4f}")

            # if (i + 1) % 100 == 0:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}]")
            #     print(f"Reconstruction Loss: {np.mean(train_res_recon_error[-100:])}")
            #     print(f"Perplexity: {np.mean(train_res_perplexity[-100:])}\n")

        progress_bar.close()

        training_loss = np.mean(train_res_total_loss)
        if (epoch) % 1 == 0:
            testing_loss = evaluate_model(model, lossca, testloader, smpl, device)
            if testing_loss < testlossbest:
                testlossbest = testing_loss
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(out_dir, 'motion_vqvae_final.pth'))

            lr = 0.0002
            logger.append([int(epoch + 1), lr, training_loss, testing_loss])
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}]")
        print(f"Reconstruction Loss: {np.mean(train_res_recon_error[-100:])}")
        print(f"Commitment Loss: {np.mean(train_res_comit_error[-100:])}")
        print(f"Vel Loss: {np.mean(train_res_velocity[-100:])}")
        print(f"Smooth Loss: {np.mean(train_res_smoothness[-100:])}")
        print(f"Perplexity1: {np.mean(train_res_perplexity1[-100:])}\n") 
        print(f"Perplexity2: {np.mean(train_res_perplexity2[-100:])}\n")   
    #     # 保存最终模型
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'recon_loss': np.mean(train_res_recon_error),
    #     'perplexity1': np.mean(train_res_perplexity1),
    #     'perplexity2': np.mean(train_res_perplexity2)
    # }, os.path.join(save_dir, 'motion_vqvae_final.pth'))
    # print(f"Final model saved at epoch {epoch+1}")
    
    # plot_training_progress(train_res_recon_error, train_res_perplexity1)
    # plot_training_progress(train_res_recon_error, train_res_perplexity2)

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
def evaluate_model(model, loss_ca, validation_loader, smpl, device):
    model.eval()
    test_loss = 0.
    output = './vaeoutput'
    writer = SummaryWriter(output)
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            # 只选一个样本
            # sample_data = validation_set.datasets[0].create_data(index=0)
            # 提取姿态数据
            pose6D = data['pose_6d'].to(device)
            B, T, J, D = pose6D.shape
            # print(pose6D.shape)
            # print(pose.shape)
            # 恢复原始形状
            # poses = axis_angles.reshape()
            pose = rotation_6d_to_matrix(pose6D)  # 转换为旋转矩阵
            pose = matrix_to_axis_angle(pose)  # 转换为 axis-angle 表示
            pose = pose.reshape(B, 16, 2, 72)

            B, T, J, D = pose6D.shape
            # pose6D = pose6D.view(B * T, J, D)
            pred = {}

            # pose6D = pose6D.view(B, T, -1)

            # vq_output_eval = model.pre_vq_conv(model.encoder(pose))
            # _, quantized, _, _ = model.vq_vae(vq_output_eval)
            # reconstructions = model.decoder(quantized)
            # print(pose6D.shape)
            _,reconstructions,_,_= model.forward(pose6D)
            # print(reconstructions.shape)
            
            reconstructed_pose6D = reconstructions.view(B, T, -1)
            # print(reconstructed_pose6D.shape)
            reconstructed_pose = rotation_6d_to_matrix(reconstructed_pose6D)  # 转换为旋转矩阵
            reconstructed_pose = matrix_to_axis_angle(reconstructed_pose)  # 转换为 axis-angle 表示
            reconstructed_pose = reconstructed_pose.reshape(B, 16, 2, 72)

            pred['recon_x'] = reconstructed_pose
            # print("sb", pred['recon_x'].shape)

            betas = data['betas'].reshape(-1, 10)  # 形状参数
            pose = data['pose'].reshape(-1, 72)    # 姿态参数
            trans = data['gt_cam_t'].reshape(-1, 3).to(smpl.device)  # 平移参数
            # betas = torch.from_numpy(betas).float().to(smpl.device)
            
            pred_pose = reconstructed_pose.reshape(-1, 72)

            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float().to(smpl.device)
            else:
                pose = pose.float().to(smpl.device)
            # print("Betas shape:", betas.shape)
            # print("Pose shape:", pose.shape)
            # print("Pose shape:", pred_pose.shape)
            # print("Trans shape:", trans.shape)

            gt_verts, gt_joints = smpl(betas, pose, trans, halpe = True)
            data['smpl_verts'] = gt_verts
            if isinstance(pose, np.ndarray):
                pred_pose = torch.from_numpy(pred_pose).float().to(smpl.device)
            else:
                pred_pose = pred_pose.float().to(smpl.device)

            verts, pred_joints = smpl(betas, pred_pose, trans, halpe = True)
            pred['pred_joints'] = pred_joints 
            pred['pred_joints'] = pred['pred_joints'].view(B, 16, 2, 26, 3)
            pred['pred_verts'] = verts

            loss, loss_dict = loss_ca.calcul_testloss(pred, data)
            # pred['MPJPE'] = loss_dict['MPJPE']
            # loss_dict['MPJPE']
            # pred['MPJPE'] = torch.tensor(pred['MPJPE']).view(1, 1).repeat(2048, 1)

            # loss1, loss_dict2 = loss_ca.calcul_testloss(pred, data)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建 MPJPE 实例
            mpjpe = MPJPE(device=device)
            mpjpe_loss = mpjpe(pred['pred_joints'], data['gt_joints'], data['valid'], False)
            pred['MPJPE'] = mpjpe_loss
            pred['recon_x'] = pred['recon_x'].view(B, T, J, 72)

            if False: #loss.max() > 100:
                # print("begin")
                results = {}
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(pred_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['recon_x'].detach().cpu().numpy().astype(np.float32))  # 预测的姿态参数
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))  # 真实的姿态参数
                results.update(pred_shape=data['betas'].detach().cpu().numpy().astype(np.float32))  # 预测的形状参数
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))  # 真实的形状参数
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))  # 预测的顶点坐标
                results.update(gt_verts=data['smpl_verts'].detach().cpu().numpy().astype(np.float32))  # 真实的顶点坐标
                results.update(MPJPE=pred['MPJPE'].detach().cpu().numpy().astype(np.float32))
                # save_generated_interaction(results, i, 64, smpl)
                # save_generated_motion(results, i, 64, writer, output)

            loss_batch = loss.mean().detach() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(validation_loader), loss_batch), loss_dict)
            test_loss += loss_batch
            test_loss = test_loss / len(validation_loader)
            
            # Record the loss
            test_loss += loss.detach().item()
    
    return test_loss


def save_generated_interaction(results, iter, batchsize,model_smpl):
        output = './vaeoutput'
        output = os.path.join(output, 'images')
        if not os.path.exists(output):
            os.makedirs(output)

        results['pred_trans'] = results['pred_trans'].reshape(batchsize, -1, 2, 3)
        results['pred_pose'] = results['pred_pose'].reshape(batchsize, -1, 2, 72)
        results['pred_shape'] = results['pred_shape'].reshape(batchsize, -1, 2, 10)
        results['pred_verts'] = results['pred_verts'].reshape(batchsize, -1, 2, 6890, 3)
        results['gt_trans'] = results['gt_trans'].reshape(batchsize, -1, 2, 3)
        results['gt_pose'] = results['gt_pose'].reshape(batchsize, -1, 2, 72)
        results['gt_shape'] = results['gt_shape'].reshape(batchsize, -1, 2, 10)
        results['gt_verts'] = results['gt_verts'].reshape(batchsize, -1, 2, 6890, 3)

        if 'MPJPE' in results.keys():
            results['MPJPE'] = results['MPJPE'].reshape(batchsize, -1, 2)

        for batch, (pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts) in enumerate(zip(results['pred_trans'], results['pred_pose'], results['pred_shape'], results['pred_verts'], results['gt_trans'], results['gt_pose'], results['gt_shape'], results['gt_verts'])):
            if batch > 0:
                break
            for f, (p_trans, p_pose, p_shape, p_verts, g_trans, g_pose, g_shape, g_verts) in enumerate(zip(pred_trans, pred_pose, pred_shape, pred_verts, gt_trans, gt_pose, gt_shape, gt_verts)):

                data = {}
                data['pose'] = p_pose
                data['trans'] = p_trans
                data['betas'] = p_shape

                path = os.path.join(output, 'pred_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                # path = os.path.join(path, '%04d.pkl' %(f))
                # save_pkl(path, data)

                data = {}
                data['pose'] = g_pose
                data['trans'] = g_trans
                data['betas'] = g_shape

                path = os.path.join(output, 'gt_params/%05d' %batch)
                os.makedirs(path, exist_ok=True)
                # path = os.path.join(path, '%04d.pkl' %(f))
                # save_pkl(path, data)

                renderer = Renderer(focal_length=1000, center=(512, 512), img_w=1024, img_h=1024,
                                    faces= model_smpl.faces,
                                    same_mesh_color=True)
                
                p_verts[:, :, 1] *= -1  # 对 y 轴取反
                g_verts[:, :, 1] *= -1 
                p_verts = p_verts + np.array([0,0,3])
                g_verts = g_verts + np.array([0,0,3])

                pred_smpl_back = renderer.render_back_view(p_verts)
                pred_smpl_backside = renderer.render_backside_view(p_verts)
                pred = np.concatenate((pred_smpl_back, pred_smpl_backside), axis=1)

                gt_smpl_back = renderer.render_back_view(g_verts)
                gt_smpl_backside = renderer.render_backside_view(g_verts)
                gt = np.concatenate((gt_smpl_back, gt_smpl_backside), axis=1)

                rendered = np.concatenate((pred, gt), axis=0)

                background = np.zeros((1024*2, 1024, 3,), dtype=rendered.dtype)

                if 'MPJPE' in results.keys():
                    background = cv2.putText(background, 'MPJPE 00: ' + str(results['MPJPE'][batch][f][0]), (50,150),cv2.FONT_HERSHEY_COMPLEX,2,(105,170,255),5)
                    background = cv2.putText(background, 'MPJPE 01: ' + str(results['MPJPE'][batch][f][1]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

                rendered = np.concatenate((background, rendered), axis=1)

            # gt_smpl = renderer.render_front_view(gt_verts, bg_img_rgb=img.copy())
                renderer.delete()
                render_name = "seq%04d_%02d_smpl.jpg" % (iter * batchsize + batch, f)
                # rendered = cv2.flip(rendered, 0) 
                cv2.imwrite(os.path.join(output, render_name), rendered)
                print(os.path.join(output, render_name))
            # render_name = "%s_%02d_gt_smpl.jpg" % (name, iter * batchsize + index)
            # cv2.imwrite(os.path.join(output, render_name), gt_smpl)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_pred_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(pred_verts, mesh_name)

            # mesh_name = os.path.join(output, 'meshes/%s_%02d_gt_mesh.obj' %(name, iter * batchsize + index))
            # self.model_smpl_gpu.write_obj(gt_verts, mesh_name)
                
            # vis_img('pred_smpl_back', pred_smpl_back)
            # vis_img('pred_smpl_backside', pred_smpl_backside)
            # vis_img('gt_smpl', gt_smpl)


def save_generated_motion(results, iter, batchsize, writer, output):

        pose_pred = results['pred_pose']
        pose_gt = results['gt_pose']

        for ii, (pred, gt) in enumerate(zip(pose_pred, pose_gt)):
            if ii > 5:
                break
            print("gt shape:", gt.shape)
            tensorborad_add_video_xyz(writer, gt, iter, tag='%05d_%05d' %(iter, ii), nb_vis=1, title_batch=['test'], outname=[os.path.join(output, '%05d_gt.gif' %ii)])
            print("pred shape:", pred.shape)
            tensorborad_add_video_xyz(writer, pred, iter, tag='%05d_%05d' %(iter, ii), nb_vis=1, title_batch=['test'], outname=[os.path.join(output, '%05d_pred.gif' %ii)])



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

def smoothness_loss(pred):
    """
    平滑损失，约束相邻帧之间的加速度变化。
    Args:
        pred: 预测的运动序列，形状为 (batch_size, time_steps, num_joints, feature_dim)
    Returns:
        loss: 平均平滑损失
    """
    # 计算相邻帧的速度
    velocity = torch.diff(pred, dim=1)
    # 计算相邻帧速度的变化（加速度）
    acceleration = torch.diff(velocity, dim=1)
    # 平滑损失为加速度的 L2 范数
    loss = torch.mean(torch.square(acceleration))
    
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
        test_dataset = dataset.load_testset()
        test_loader = DataLoader(
            test_dataset,
            batch_size=batchsize, shuffle=False,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # 训练 Motion VQ-VAE
        train_motion_vqvae(motion_vqvae, train_loader, test_loader, loss, smpl, optimizer, num_epoch, logger, device)

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

        evaluate_model(motion_vqvae, loss, test_loader, smpl, device)

    # # 从子数据集中提取样本并可视化
    # # sample_data = train_dataset.datasets[0].create_data(index=0)
    # # visualize_smpl_model(smpl)

if __name__ == "__main__":
    args = parse_config()

    main(**args)
