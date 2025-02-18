'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
from tqdm import tqdm
from utils.smpl_torch_batch import SMPLModel

def merge_gt(data):
    batch_size, frame_length, agent_num = data['pose'].shape[:3]

    data['data_shape'] = data['pose'].shape[:3]
    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, 6890, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)
    data['x'] = data['x'].reshape(batch_size*frame_length*agent_num, -1)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid(data):
    batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

    data['data_shape'] = data['keypoints'].shape[:3]
    data['center'] = data['center'].reshape(batch_size*frame_length*agent_num, -1)
    data['scale'] = data['scale'].reshape(batch_size*frame_length*agent_num,)
    data['img_h'] = data['img_h'].reshape(batch_size*frame_length*agent_num,)
    data['img_w'] = data['img_w'].reshape(batch_size*frame_length*agent_num,)
    data['focal_length'] = data['focal_length'].reshape(batch_size*frame_length*agent_num,)
    data['valid'] = data['valid'].reshape(batch_size*frame_length*agent_num,)

    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, -1, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['keypoints'] = data['keypoints'].reshape(batch_size*frame_length*agent_num, 26, 3)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['norm_img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'].reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    # imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    # data['imgname'] = imgname.tolist()

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data

def reconstruction_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['keypoints'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        debug = False
        if debug:
            results = {}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            model.save_generated_interaction(results, i, batchsize)

        debug = False
        if debug:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        train_loss += loss_batch

    return train_loss/len_data

def reconstruction_test(model, loss_func, loader, epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['keypoints'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if i < 1: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                if 'MPJPE_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

            loss_batch = loss.mean().detach() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def reconstruction_eval(model, loader, loss_func, device=torch.device('cpu')):

    print('-' * 10 + 'model eval' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    output = {'pose':{}, 'shape':{}, 'trans':{}}
    gt = {'pose':{}, 'shape':{}, 'trans':{}, 'gender':{}, 'valid':{}}
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            # if i > 1:
            #     break
            batchsize = data['keypoints'].shape[0]
            seq_id = data['seq_id']
            frame_id = torch.cat(data['frame_id']).reshape(-1, batchsize)
            frame_id = frame_id.detach().cpu().numpy().T

            batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

            del data['seq_id']
            del data['frame_id']
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)

            pred_pose = pred['pred_pose'].reshape(batch_size, frame_length, agent_num, -1)
            pred_shape = pred['pred_shape'].reshape(batch_size, frame_length, agent_num, -1)
            pred_trans = pred['pred_cam_t'].reshape(batch_size, frame_length, agent_num, -1)

            pred_pose = pred_pose.detach().cpu().numpy()
            pred_shape = pred_shape.detach().cpu().numpy()
            pred_trans = pred_trans.detach().cpu().numpy()

            gt_pose = data['pose'].reshape(batch_size, frame_length, agent_num, -1)
            gt_shape = data['betas'].reshape(batch_size, frame_length, agent_num, -1)
            gt_trans = data['gt_cam_t'].reshape(batch_size, frame_length, agent_num, -1)
            gt_gender = data['gender'].reshape(batch_size, frame_length, agent_num)
            valid = data['valid'].reshape(batch_size, frame_length, agent_num)

            gt_pose = gt_pose.detach().cpu().numpy()
            gt_shape = gt_shape.detach().cpu().numpy()
            gt_trans = gt_trans.detach().cpu().numpy()
            gt_gender = gt_gender.detach().cpu().numpy()
            valid = valid.detach().cpu().numpy()

            for batch in range(batchsize):
                s_id = str(int(seq_id[batch]))
                for f in range(frame_length):

                    if s_id not in output['pose'].keys():
                        output['pose'][s_id] = [pred_pose[batch][f]]
                        output['shape'][s_id] = [pred_shape[batch][f]]
                        output['trans'][s_id] = [pred_trans[batch][f]]

                        gt['pose'][s_id] = [gt_pose[batch][f]]
                        gt['shape'][s_id] = [gt_shape[batch][f]]
                        gt['trans'][s_id] = [gt_trans[batch][f]]
                        gt['gender'][s_id] = [gt_gender[batch][f]]
                        gt['valid'][s_id] = [valid[batch][f]]
                    else:
                        output['pose'][s_id].append(pred_pose[batch][f])
                        output['shape'][s_id].append(pred_shape[batch][f])
                        output['trans'][s_id].append(pred_trans[batch][f])

                        gt['pose'][s_id].append(gt_pose[batch][f])
                        gt['shape'][s_id].append(gt_shape[batch][f])
                        gt['trans'][s_id].append(gt_trans[batch][f])
                        gt['gender'][s_id].append(gt_gender[batch][f])
                        gt['valid'][s_id].append(valid[batch][f])
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

        return output, gt


def motionVqvae_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu'), **kwargs):
    """
    Training function for Motion VQ-VAE model
    """
    print('-' * 10 + 'motion vqvae model training' + '-' * 10)
    model_smpl = SMPLModel(
                                device=torch.device('cuda'),
                                model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                                data_type=torch.float32,
                            )
    len_data = len(train_loader)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        # batchsize = data['pose'].shape[0]
        data = to_device(data, device) # Move pose data to the specified device
        pose = data['pose']
        # Flatten time dimension into batch dimension
        # B, T, J, D = pose.shape
        # pose = pose.view(B * T, J, D)

        # Zero the gradients
        model.optimizer.zero_grad()

        # Forward pass
        pred = model.model(pose)  # 返回一个字典

        # pred_pose = pred['x_recon'].reshape(-1, 72).contiguous()
        # betas = data['betas'].view(-1, 10).to(device) 
        # trans = data['gt_cam_t'].view(-1, 3).to(device)
        # trans = torch.zeros_like(data['gt_cam_t']).view(-1, 3).to(device)
        # g_pose = pose.reshape(-1, 72).contiguous()

        # verts, pred_joints = model_smpl(betas, pred_pose, trans, halpe=True)
        # pred_joints  = pred_joints .view(64, 16, 2, 26, 3)
        # print("Sample Prediction:", pred_joints[0])
        # print("pose:", pred['x_recon'][0] )
        # print("Gpose:", pose[0] )
        # print("Gbetas:", betas[0] )
        # print("Gtrans:", trans[0] )
        
        # Calculate loss
        loss, loss_dict = loss_func.calcul_trainloss(pred, data)

        # Backward pass and optimize
        loss.backward()
        model.optimizer.step()

        # Record the loss
        train_loss += loss.detach().item()

        print(f'epoch: {epoch}/{num_epoch}, batch: {i}/{len_data}, '
              f'loss: {loss.item():.6f}, recon_loss: {loss_dict.get("recon_loss", 0):.6f}, '
              f'vq_loss: {loss_dict.get("vq_loss", 0):.6f}, bone_length_loss: {loss_dict.get("bone_length_loss", 0):.6f}, '
              f'velocity_loss: {loss_dict.get("velocity_loss", 0):.6f}')

    return train_loss / len_data

def motionVqvae_test(model, loss_func, test_loader, epoch, device=torch.device('cpu'),**kwargs):
    """
    Testing function for Motion VQ-VAE model
    """
    print('-' * 10 + 'motion vqvae model testing' + '-' * 10)
    smpl = kwargs.get('smpl', None)  # 提取 smpl 参数，默认 None
    if smpl is None:
        raise ValueError("SMPL model must be provided for motionVqvae_train.")
    smpl = smpl.to(device)
    
    len_data = len(test_loader)
    test_loss = 0.

    model_smpl = SMPLModel(
                                device=torch.device('cuda'),
                                model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                                data_type=torch.float32,
                            )

    # Set model to evaluation mode
    model.model.eval()

    with torch.no_grad():  # Disable gradient calculations
        for i, data in enumerate(test_loader):
            # Move data to the specified device
            data = to_device(data, device)
            pose = data['pose']

            # Forward pass
            pred = model.model(pose)  # 返回一个字典
            print("pred :", pred['x_recon'][0] )
            pose = pose.view(-1, 72)
            pred_pose = pred['x_recon'].reshape(-1, 72).contiguous()
            betas = data['betas'].view(-1, 10).to(device) 
            trans = data['gt_cam_t'].view(-1, 3).to(device)
            trans = torch.zeros_like(data['gt_cam_t']).view(-1, 3).to(device)
            # print(f"betas.shape: {betas.shape}")
            # print("Gtrans:", trans[0] )
            # print("pred :", pred_pose[0] )
            # print("pose:", pose[0] )
            # print(f"pred['x_recon'].shape: {pred['x_recon'].shape}")

            # print(f"betas.device: {betas.device}")
            # print(f"thetas.device: {pred['x_recon'].device}")
            # print(f"trans.device: {trans.device}")
            # print(f"smpl.device: {next(smpl.parameters()).device}") 

            verts, pred_joints = model_smpl(betas, pred_pose, trans, halpe=True)
            pred['pred_joints'] = pred_joints 
            # Calculate loss
            pred['pred_joints'] = pred['pred_joints'].view(64, 16, 2, 26, 3)
            
            # print(f"pred_joints: {pred['pred_joints'].shape}")
            # print(f"data_gt_joints: {data['gt_joints'].shape}")
            
            # print("Pred Joints Range:", pred['pred_joints'].min().item(), pred['pred_joints'].max().item(), pred['pred_joints'].mean().item())
            # print("GT Joints Range:", data['gt_joints'].min().item(), data['gt_joints'].max().item(), data['gt_joints'].mean().item())
            # print("Sample Prediction:", pred['pred_joints'][0])
            # print("Sample Ground Truth:", data['gt_joints'][0])
            # break
            loss, loss_dict = loss_func.calcul_testloss(pred, data)


            loss_batch = loss.mean().detach() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(test_loader), loss_batch), loss_dict)
            test_loss += loss_batch
            test_loss = test_loss / len(test_loader)
            
            # Record the loss
            test_loss += loss.detach().item()

            print(f'epoch: {epoch}, batch: {i}/{len_data}, '
                  f'loss: {loss.item():.6f}')

    return test_loss