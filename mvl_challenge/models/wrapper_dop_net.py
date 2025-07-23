import json
import logging
import os
import sys
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import statistics
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from copy import deepcopy
from mvl_challenge.config.cfg import save_cfg
from mvl_challenge import ROOT_DIR
from mvl_challenge.datasets.mvl_dataset import MVImageLayout
from mvl_challenge.utils.io_utils import save_json_dict, print_cfg_information, create_directory
from mvl_challenge.data_loaders.mvl_dataloader import MVLDataLoader
from mvl_challenge.utils.eval_utils import eval_2d3d_iuo_from_tensors, compute_weighted_L1, compute_L1_loss
from mvl_challenge.models.DOPNet.utils.boundary import depth2boundaries
from mvl_challenge.models.DOPNet.utils.conversion import depth2pixel, uv2pixel,uv2depth,xyz2depth,pixel2uv, depth2xyz
from mvl_challenge.utils.spherical_utils import uv2phi_coords,phi_coords2xyz,phi_coords2uv, phi_coords2xyzTorch
from mvl_challenge.models.DOPNet.loss.grad_depth_loss import GradDepthLoss
from mvl_challenge.datasets.mvl_dataset import iter_mvl_room_scenes_train

import random

from PIL import Image
from mvl_challenge.utils.geometry_utils import extend_array_to_homogeneous
from mvl_challenge.utils.spherical_utils import phi_coords2xyz
from mvl_challenge.models.DOPNet.utils.conversion import depth2pixel, uv2pixel,uv2depth,xyz2depth,pixel2uv
from mvl_challenge.utils.spherical_utils import uv2phi_coords,phi_coords2xyz,phi_coords2uv
from torch.utils.data import Dataset

class SceneDataset(Dataset):
    def __init__(self, list_ly) -> None:
        super().__init__()
        self.list_ly = list_ly

    def __len__(self):
        return len(self.list_ly)
    
    def __getitem__(self, index):
        ly = self.list_ly[index]
        image_path = ly.img_fn

        img = np.array(Image.open(image_path), np.float32)[..., :3] / 255.
        
        ps_label_fn = ly.ps_label_fn
        std_fn = ly.std_fn
        stdr_fn = ly.stdr_fn
        ratio_fn = ly.ratio_fn
        gt_label_fn = ly.gt_label_fn

        if os.path.exists(ps_label_fn + ".npy"):
            ps_label = np.load(ps_label_fn + ".npy")
        elif os.path.exists(ps_label_fn + ".npz"):
            ps_label = np.load(ps_label_fn + ".npz")["phi_coords"]
        else:
            raise ValueError(f"Not found {ps_label_fn}")
        
        gt_label = None
        
        if gt_label_fn is not None:  
          if os.path.exists(gt_label_fn + ".npy"):
              gt_label = np.load(gt_label_fn + ".npy")
              gt_label = torch.FloatTensor(gt_label.copy())
          elif os.path.exists(gt_label_fn + ".npz"):
              gt_label = np.load(gt_label_fn + ".npz")["phi_coords"]
              gt_label = torch.FloatTensor(gt_label.copy())
          else:
            raise ValueError(f"Not found {gt_label}")
            
        if gt_label is None:
          # Random flip
          if ly.cfg.get('flip', False) and np.random.randint(2) == 0:
              img = np.flip(img, axis=1)
              ps_label = np.flip(ps_label, axis=len(label.shape) - 1)
              gt_label = np.flip(gt_label, axis=len(label.shape) - 1)
          '''
          # Random horizontal rotate
          if self.cfg.get('rotate', False):
              dx = np.random.randint(img.shape[1])
              img = np.roll(img, dx, axis=1)
              label = np.roll(label, dx, axis=len(label.shape) - 1)
          '''
  
          # Random gamma augmentation
          if ly.cfg.get('gamma', False):
              p = np.random.uniform(1, 2)
              if np.random.randint(2) == 0:
                  p = 1 / p
              img = img**p
        
        if os.path.exists(std_fn):
            std = np.load(std_fn)
        else:
            std = np.ones_like(ps_label)
            
        if os.path.exists(stdr_fn):
            stdr = np.load(stdr_fn)
        else:
            stdr = np.ones_like(ps_label)
        
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        ps_label = torch.FloatTensor(ps_label.copy())
        std = torch.FloatTensor(std.copy())
        stdr = torch.FloatTensor(stdr.copy())

        with open(ratio_fn, 'r') as f:
            datar = json.load(f)
        ratio = datar['ratio']
        y_bon_ref = ps_label
        
        uv_c, uv_f = phi_coords2uv(y_bon_ref)#uv是pixel的还是

        uv_c=uv_c.transpose(1,0)
        uv_f = uv_f.transpose(1, 0)
        uv_c=pixel2uv(uv_c)
        uv_f = pixel2uv(uv_f)
        c_depth=uv2depth(uv_c)#这个现在是pixelUV，数值是（1,1023）
        f_depth=uv2depth(uv_f)
        #c_depth  但误差大 舍弃
        c_depth=torch.tensor(c_depth)
        c_depth=c_depth.unsqueeze(0).unsqueeze(0)#linear only for 3D
        #c_depth=torch.nn.functional.interpolate(c_depth,  scale_factor=0.25, mode='linear', align_corners=False)
        c_depth=c_depth.squeeze(0).squeeze(0)
        c_depth=np.array(c_depth)
        #f_depth 误差小  选择
        f_depth = torch.tensor(f_depth)
        f_depth = f_depth.unsqueeze(0).unsqueeze(0)  # linear only for 3D
        #f_depth = torch.nn.functional.interpolate(f_depth, scale_factor=0.25, mode='linear',align_corners=False)
        f_depth = f_depth.squeeze(0).squeeze(0)
        f_depth = np.array(f_depth)
        depth = f_depth
        pose = ly.pose.SE3_scaled()
        ly.img = x
        ly.gt_label = gt_label
        ly.ps_label = ps_label
        ly.std = std
        ly.stdr = stdr
        ly.depth = depth
        ly.ratio = ratio
        #ly.pose_all = pose
        if gt_label is not None:
          sample = {
              "image": x,
              "depth": depth,
              "ps_label": ps_label,
              "ratio": ratio,
              "std": std,
              "stdr": stdr,
              "pose": pose,
              "gt_label": gt_label
          }
        else:
          sample = {
              "image": x,
              "depth": depth,
              "ps_label": ps_label,
              "ratio": ratio,
              "std": std,
              "stdr": stdr,
              "pose": pose,
          }
        return sample


class WrapperDOPNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_horizon_net_path()
        from mvl_challenge.models import utilsdop as dop_utils
        from mvl_challenge.models.DOPNet.models.my_layout_net import My_Layout_Net

        # ! Setting cuda-device
        self.device = torch.device(
            f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu"
        )

        # Loaded trained model
        assert os.path.isfile(cfg.model.ckpt), f"Not found {cfg.model.ckpt}"
        logging.info("Loading DOPNet...")
        self.net = dop_utils.load_trained_model(My_Layout_Net, cfg.model.ckpt).to(self.device)

        self.grad_depth_conv = GradDepthLoss().to(self.device) # ???这里应该怎么写  ??什么时候加self呐,这个需不需要加到init里面？？ 问题1



        logging.info(f"ckpt: {cfg.model.ckpt}")
        logging.info("DOPNet Wrapper Successfully initialized")

    @staticmethod
    def set_horizon_net_path():
        hn_dir = os.path.join(ROOT_DIR, "models", "DOPNet")#hn_dir:   MVL_Challenge/mvl_toolkitDOPNet/mvl_challenge/models/DOPNet/
        if hn_dir not in sys.path:
            sys.path.append(hn_dir)

    def estimate_within_list_ly(self, list_ly):
        """
        Estimates phi_coords (layout boundaries) for all ly defined in list_ly using the passed model instance
        """

        layout_dataloader = DataLoader(
            MVImageLayout([(ly.img_fn, ly.idx) for ly in list_ly]),
            batch_size=self.cfg.runners.mvl.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.mvl.num_workers,
            pin_memory=True if self.device != "cpu" else False,
            worker_init_fn=lambda x: np.random.seed(),
        )
        self.net.eval()
        evaluated_data = {}
        ratio_data={}
        for x in tqdm(layout_dataloader, desc=f"Estimating layout..."):
            with torch.no_grad():
                # print('在estimate_within_list_ly函数中，x["images"]的形状是 ', x["images"].shape) #torch.Size([10, 3, 512, 1024])
                depth, ratio = self.net(x["images"].to(self.device))
                # print("depth.shape   ", depth.shape)#torch.Size([10, 256])
                # print(type(depth[0].cpu().numpy()))
                b, w = depth.shape
                phi_c = []
                depth2=[]
                for ii in range(b):
                    # print('当前的ii是',ii)
                    # print("depth[ii]的shape是 ", depth[ii].shape)#torch.Size([256])
                    # print("ratio[ii]的shape是 ",ratio[ii].shape)#torch.Size([1])
                    floor_uv, ceiling_uv = depth2boundaries(ratio[ii].cpu().numpy(), depth[ii].cpu().numpy())

                    # floor生成
                    # print("floor_uv type:", type(floor_uv))#<class 'numpy.ndarray'>
                    # print("floor_uv shape:", floor_uv.shape)#(463, 2)
                    floor_uv_ps = uv2pixel(floor_uv)
                    # print("floor_uv_ps type:", type(floor_uv_ps))# <class 'numpy.ndarray'>
                    # print("floor_uv_ps shape:", floor_uv_ps.shape)#(463, 2) #这个每个batch是不固定的
                    # print('floor_uv_ps是 ',floor_uv_ps)
                    floor_uv_ps = np.transpose(floor_uv_ps, (1, 0))
                    # print("floor_uv_ps 转置后shape:", floor_uv_ps.shape)#(2, 463)
                    phi_c_floor = uv2phi_coords(floor_uv_ps, type_bound="floor")  # 1024
                    # print("phi_c_floor的类型", type(phi_c_floor))#<class 'numpy.ndarray'>
                    # print("phi_c_floor的形状", phi_c_floor.shape)#(1024,)

                    # ceiling生成
                    # print("ceiling_uv type:", type(ceiling_uv))#<class 'numpy.ndarray'>
                    # print("ceiling_uv shape:", ceiling_uv.shape)#(556, 2)
                    ceiling_uv_ps = uv2pixel(ceiling_uv)
                    # print("ceiling_uv_ps type:", type(ceiling_uv_ps))#<class 'numpy.ndarray'>
                    # print("ceiling_uv_ps shape:", ceiling_uv_ps.shape)#(556, 2),这个每个batch是不固定的
                    # print('ceiling_uv_ps是 ',ceiling_uv_ps)

                    ceiling_uv_ps = np.transpose(ceiling_uv_ps, (1, 0))
                    # print("ceiling_uv_ps 转置后shape:", ceiling_uv_ps.shape)#(2, 556)
                    phi_c_ceiling = uv2phi_coords(ceiling_uv_ps, type_bound="ceiling")  # 1024
                    # print("phi_c_ceiling的类型", type(phi_c_ceiling))#<class 'numpy.ndarray'>
                    # print("phi_c_ceiling的形状", phi_c_ceiling.shape)# (1024,)

                    data_phi = np.vstack((phi_c_ceiling, phi_c_floor))
                    # print('data_phi的形状是 ',data_phi.shape)# (2, 1024)
                    phi_c.append(data_phi)
                    #dataphi转回depth
                    # print('data_phi',data_phi)#这个是phi_coords的取值
                    ''' # (2, 1024)
                    [[-0.77312632 -0.77312632 -0.77312632 ... -0.77926224 -0.77926224 -0.77926224]
                    [ 0.81607778  0.81607778  0.81607778 ...  0.81607778  0.81607778 0.81607778]]
                    '''
                    uv_c_data_phi,uv_f_data_phi=phi_coords2uv(data_phi)
                    uv_c_data_phi=uv_c_data_phi.transpose(1,0)
                    uv_c_data_phi=pixel2uv(uv_c_data_phi)
                    # print('在uv2depth 之前的uv_c_data_phi',uv_c_data_phi)
                    '''
                                        [[4.88281250e-04 3.48632812e-01]
                                         [1.46484375e-03 3.48632812e-01]
                                         [2.44140625e-03 3.48632812e-01]
                                         ...
                                         [9.97558594e-01 3.48632812e-01]
                                         [9.98535156e-01 3.48632812e-01]
                                         [9.99511719e-01 3.48632812e-01]]
                                        '''
                    #print('uv_c_data_phi shape ',uv_c_data_phi.shape)

                    c_depth = uv2depth(uv_c_data_phi)
                    c_depth=torch.tensor(c_depth)
                    c_depth = c_depth.unsqueeze(0).unsqueeze(0)
                    c_depth = torch.nn.functional.interpolate(c_depth, scale_factor=0.25, mode='linear',
                                                              align_corners=False)
                    c_depth = c_depth.squeeze(0).squeeze(0)
                    c_depth = np.array(c_depth)
                    depth2.append(c_depth)
                # Difference_value=depth-depth2
                # print(Difference_value)


                    # uv_ps = depth2pixel(depth[ii].cpu().numpy())
                    # print("transpose之前的shape：", uv_ps.shape)
                    # uv_ps = np.transpose(uv_ps, (1,0))
                    ##print(uv_ps[0])
                    ##uv_ps = uv_ps.transpose(0, 1)
                    # print(uv_ps.shape)
                    # phi_coords_floor = uv2phi_coords(uv_ps, type_bound="floor")
                    # print(type(phi_coords_floor))
                    # phi_coords_ceiling = uv2phi_coords(uv_ps, type_bound="ceiling")
                    # data_phi = np.vstack((phi_coords_ceiling, phi_coords_floor))
                    # print(data_phi.shape)
                    # phi_c.append(data_phi)
                # (phi_coords_floor.shape)
                # print(phi_coords_ceiling.shape)
                # y_bon_, y_cor_ = net(x[0].to(device))
                # phi_c_array = np.array(phi_c)
            for y_, ratio_,idx, in zip(phi_c,ratio, x["idx"]):
                # print(type(y_))
                evaluated_data[idx] = y_
                ratio_data[idx]=ratio_

        [ly.set_ratio(ratio=float(ratio_data[ly.idx])) for ly in list_ly]
        [ly.set_phi_coords(phi_coords=evaluated_data[ly.idx]) for ly in list_ly]

    def set_optimizer(self):
        if self.cfg.model.optimizer == "SGD":
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                momentum=self.cfg.model.beta1,
                weight_decay=self.cfg.model.weight_decay,
            )
        elif self.cfg.model.optimizer == "Adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.cfg.model.lr,
                betas=(self.cfg.model.beta1, 0.999),
                weight_decay=self.cfg.model.weight_decay,
            )
        else:
            raise NotImplementedError()

    def set_scheduler(self):
        decayRate = self.cfg.model.lr_decay_rate
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=decayRate
        )

    def save_current_scores(self):
        # ! Saving current epoch data
        fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        save_json_dict(filename=fn, dict_data=self.curr_scores)
        # ! Save the best scores in a json file regardless of saving the model or not
        save_json_dict(
            dict_data=self.best_scores,
            filename=os.path.join(self.dir_ckpt, "best_score.json")
        )
    
    
    def train_iou_within_list_ly(self, mvl_train):
        if not self.is_training:
            logging.warning("Wrapper is not ready for training")
            return False
        # ! Freezing some layer    #要改  resnet50,34   dopnet有了分割  分割以及分割之前都freeze
        # 我们还要冻结更多层
        # ! Freezing some layer
        # for param in self.net.seg_head.parameters():
        #     param.requires_grad = False
        if self.cfg.model.freeze_earlier_blocks != -1:  # freeze_earlier_blocks: -1
            b0, b1, b2, b3, b4 = self.net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(self.cfg.model.freeze_earlier_blocks + 1):
                logging.warn('Freeze block %d' % i)
                for m in blocks[i]:
                    for param in m.parameters():
                        param.requires_grad = False

        if self.cfg.model.bn_momentum != 0:  # bn_momentum: 0
            for m in self.net.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = self.cfg.model.bn_momentum

        #print_cfg_information(self.cfg)

        self.net.train()
        self.iterations = 0        
        for list_ly in iter_mvl_room_scenes_train(dataset=mvl_train):
          batch_size = self.cfg.runners.train.batch_size
          # for debug set batch size 1
          dataset = SceneDataset(list_ly)
          dataloader = DataLoader(
              dataset=dataset,
              shuffle=True,
              batch_size=self.cfg.runners.train.batch_size,
              num_workers=self.cfg.runners.train.num_workers,
              drop_last=True,
              pin_memory=True,
          )
          for i, sample in enumerate(dataloader):
            print("迭代次数{}/当前轮数{}".format(self.iterations, self.current_epoch))
            #self.iterations = self.iterations + 1
            image = sample["image"]
            std = sample["std"]
            stdr = sample["stdr"]
            depth = sample["depth"]
            gtratio = sample["ratio"]
            pose = sample["pose"]
            phi_bon = sample["ps_label"]
            
            depth_ly = torch.tensor(depth)
            
            
            self.iterations += 1
                
            est_depth, est_ratio= self.net(image.to(self.device), depth.to(self.device), pose.to(self.device)) #可以depth，ratio，再用depth和ration转成boundary
            
            est_depth = est_depth.unsqueeze(1)
            
            est_depth = torch.nn.functional.interpolate(est_depth, scale_factor=4, mode='linear',
                                                          align_corners=False)
            est_depth = est_depth.squeeze(1)
            
            #print(est_depth.shape)
            floor_xyz = depth2xyz(est_depth)
            #return xyz_unit_sphere
            est_ceiling_xyz = floor_xyz
            est_ceiling_xyz[:, :, 1] *= -est_ratio
            ceiling_xyz = est_ceiling_xyz
            norm = torch.norm(ceiling_xyz, dim=-1, keepdim=True)  # 计算范数
            ceiling_xyz = ceiling_xyz / norm  # 转换到单
            #print(ceiling_xyz)
            '''
            ref_ceiling_depth = []
            for i in range(est_depth.shape[0]):            
              uv_c, uv_f = phi_coords2uv(phi_bon[i])#uv是pixel的还是
              uv_c=uv_c.transpose(1,0)
              uv_c=pixel2uv(uv_c)
              c_depth=uv2depth(uv_c)#这个现在是pixelUV，数值是（1,1023）
              ref_ceiling_depth.append(c_depth)
            ref_ceiling_depth = torch.tensor(ref_ceiling_depth)
            ref_ceiling_xyz = depth2xyz(ref_ceiling_depth, plan_y=-1)
            #return xyz_unit_sphere
            gt_ceiling_xyz = ref_ceiling_xyz
            #est_ceiling_xyz[:, 1] *= -est_ratio
            '''
            ref_ceiling_depth = []
            for i in range(est_depth.shape[0]):
              tmp_ceiling_xyz = phi_coords2xyz(phi_bon[i,0,:]) 
              #print(tmp_ceiling_xyz.shape)
              tmp_ceiling_xyz = tmp_ceiling_xyz.transpose(1, 0)
              #print(tmp_ceiling_xyz.shape)           
              ref_ceiling_depth.append(tmp_ceiling_xyz)
            gt_ceiling_xyz = torch.tensor(ref_ceiling_depth)
            
            #print(gt_ceiling_xyz)
            

            #depth_ly = torch.tensor(depth)  # 转换为张量对象
            std = torch.mean(std, dim=1)

            normal_loss,grad_loss=self.grad_depth_conv(depth_ly.clone().detach().to(self.device),est_depth.to(self.device), std.clone().detach().to(self.device), self.cfg.model.min_std)#返回值用一个列表包裹起来了  问题2
            
            stdd= stdr
            #stdr = #stdr.unsqueeze(1).repeat(1, 1024)
            #print(stdr.shape)
            std_xyz = std.unsqueeze(2).repeat(1,1,3)
            #print(std_xyz.shape)
            #print(std_xyz.shape)
            #print(ceiling_xyz.shape)
            #print(gt_ceiling_xyz.shape)
            stdmean = std.mean()
            ratio_loss = compute_weighted_L1(est_ratio.to(self.device), gtratio.clone().detach().to(self.device),stdmean.clone().detach().to(self.device), self.cfg.model.min_std)
            depth_loss = compute_weighted_L1(depth_ly.clone().detach().to(self.device), est_depth.to(self.device), std.clone().detach().to(self.device), self.cfg.model.min_std)
            relace_ratio_loss = compute_weighted_L1(gt_ceiling_xyz.clone().detach().to(self.device), ceiling_xyz.to(self.device), std_xyz.clone().detach().to(self.device), self.cfg.model.min_std)
            #seg_loss = SegLoss()(seg.clone().detach()..to(self.device), est_seg)
            total_loss=0.1*normal_loss+0.1*grad_loss+0.9*depth_loss+0.08*relace_ratio_loss#+0.01*seg_loss 
            # total_loss = normal_loss + grad_loss+depth_loss
            loss=total_loss
            #print(ceiling_xyz.to(self.device) - gt_ceiling_xyz.to(self.device))
            ##print(gt_ceiling_xyz.to(self.device))
            #print("seg_loss", seg_loss.item())
            print("ratio_loss", ratio_loss.item())
            print("relace_ratio_loss", relace_ratio_loss.item())
            print("depth_loss", depth_loss.item())
            print("total_loss", total_loss.item())
            if loss.item() is np.NAN:
                raise ValueError("something is wrong")
            self.tb_writer.add_scalar(
                "train/loss", loss.item(), self.iterations)
            self.tb_writer.add_scalar(
                "train/lr", self.lr_scheduler.get_last_lr()[0], self.iterations)

            # back-prop
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.net.parameters(), 3.0, norm_type="inf")
            self.optimizer.step()

        self.lr_scheduler.step()

        # Epoch finished
        self.current_epoch += 1

        # ! Saving model
        if self.cfg.model.get("save_every") > 0:
            if self.current_epoch % self.cfg.model.get("save_every", 5) == 0:
                self.save_model(f"model_at_{self.current_epoch}.pth")

        if self.current_epoch > self.cfg.model.epochs:
            self.is_training = False

        # # ! Saving current epoch data
        # fn = os.path.join(self.dir_ckpt, f"valid_eval_{self.current_epoch}.json")
        # save_json_dict(filename=fn, dict_data=self.curr_scores)

        return self.is_training
        
        
    def valid_iou_within_list_ly(self, mvl_val, only_val=False):
        print_cfg_information(self.cfg)
        self.iterations = 0
        self.net.eval()
        iterator_valid_iou = 0#创建了一个迭代器iterator_valid_iou，用于遍历self.valid_iou_loader的数据。
        total_eval = {}#定义了一个空字典total_eval，用于存储评估结果
        invalid_cnt = 0#初始化了一个变量invalid_cnt，用于计数无效的结果数量
        for list_ly in iter_mvl_room_scenes_train(dataset=mvl_val):
          batch_size = self.cfg.runners.train.batch_size
          if len(list_ly) % batch_size != 0:
            # 计算最后一个批次的大小
            last_batch_size = len(list_ly) % batch_size
            # 分离最后一个批次的数据
            last_batch_data = list_ly[-last_batch_size:]
            # 取出其他的数据
            other_data = list_ly[:-last_batch_size]
            # 计算需要重复的数据的数量
            num_to_repeat = batch_size - last_batch_size
            # 从other_data中随机选择数据进行重复
            repeated_data = random.choices(other_data, k=num_to_repeat)
            # 将随机选择的数据和最后一个批次的数据添加到list_ly的末尾
            list_ly += repeated_data + last_batch_data

          '''
          if len(list_ly) % batch_size != 0:
            num_to_repeat = batch_size - (len(list_ly) % batch_size)
            repeated_data = random.choices(list_ly, k=num_to_repeat)  # 随机选择数据进行重复
            list_ly += repeated_data  # 将重复的数据添加到原始数据中
          '''
          # for debug set batch size 1
          dataset = SceneDataset(list_ly)
          dataloader = DataLoader(
              dataset=dataset,
              shuffle=False,
              batch_size=self.cfg.runners.valid_iou.batch_size,
              num_workers=self.cfg.runners.valid_iou.num_workers,
              drop_last=True,
              pin_memory=True,
          )
          for i, sample in enumerate(dataloader):
            print("迭代次数{}/当前轮数{}".format(self.iterations, self.current_epoch))
            iterator_valid_iou = iterator_valid_iou+1
            self.iterations = self.iterations + 1
            image = sample["image"]
            std = sample["std"]
            label = sample["gt_label"]
            ps_label = sample["ps_label"]
            depth = sample["depth"]
            #gtratio = sample["ratio"]
            pose = sample["pose"]
            x, y_bon_ref, std = image, label, std#在每次循环中，next(iterator_valid_iou)从迭代器中获取下一个元素
            with torch.no_grad():
                #y_bon_est, _ = self.net(x.to(self.device))  #改
                depth, ratio = self.net(x.to(self.device), depth.to(self.device), pose.to(self.device))
                #print('depth shape ',depth.shape)
                b, w = depth.shape
                phi_c = []
                depth2_c = []
                depth2_f = []
                for ii in range(b):
                    floor_uv, ceiling_uv = depth2boundaries(ratio[ii].cpu().numpy(), depth[ii].cpu().numpy())

                    floor_uv_ps = uv2pixel(floor_uv)
                    floor_uv_ps = np.transpose(floor_uv_ps, (1, 0))
                    phi_c_floor = uv2phi_coords(floor_uv_ps, type_bound="floor")  # 1024

                    ceiling_uv_ps = uv2pixel(ceiling_uv)
                    ceiling_uv_ps = np.transpose(ceiling_uv_ps, (1, 0))
                    phi_c_ceiling = uv2phi_coords(ceiling_uv_ps, type_bound="ceiling")  # 1024,)

                    data_phi = np.vstack((phi_c_ceiling, phi_c_floor))
                    phi_c.append(data_phi)
                    
                y_bon_est = phi_c
                #y_bon_est = np.array(y_bon_est)
                y_bon_est = torch.Tensor(y_bon_est)
                #print(y_bon_est.shape)
                #print(y_bon_ref.shape)
                # print('y_bon_est shape  ',y_bon_est.shape)#y_bon_est shape   (4, 2, 1024)
                # print('y_bon_ref type',type(y_bon_ref))#<class 'torch.Tensor'>
                # print('y_bon_est type', type(y_bon_est))#<class 'numpy.ndarray'>

                true_eval = {"2DIoU": [], "3DIoU": []}#定义了一个名为true_eval的字典，用于存储评估指标的结果
                for gt, est in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy()):#在y_bon_ref和y_bon_est这两个numpy数组上进行迭代。zip函数将两个数组的对应元素打包成元组，并在循环中进行迭代。
                    eval_2d3d_iuo_from_tensors(est[None], gt[None], true_eval, )#函数可能是用于计算目标估计结果和真实结果之间的2D和3D IoU评估指标，并将结果存储到true_eval中。
                #GPT:这个函数可能计算了权重化的L1损失，即对y_bon_est和y_bon_ref之间的差异进行加权求和，权重由std确定。
                local_eval = dict(loss=compute_weighted_L1(y_bon_est.to(self.device), y_bon_ref.to(self.device), std.to(self.device)))

                #2D和3D IoU的平均值，并将结果存储到local_eval字典中的相应键中
                local_eval["2DIoU"] = torch.FloatTensor(
                    [true_eval["2DIoU"]]).mean()
                local_eval["3DIoU"] = torch.FloatTensor(
                    [true_eval["3DIoU"]]).mean()
            try:
                for k, v in local_eval.items():#for循环遍历local_eval字典的键值对
                    if v.isnan():#个键值对，首先检查值v是否为NaN（Not a Number）。如果是NaN，则跳过后续代码，继续下一个迭代。
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item() * x.size(0)#如果值v是一个number，那么通过total_eval.get(k, 0)获取total_eval字典中键k对应的值。如果该键不存在，返回默认值0。将局部评估结果乘以输入x的大小x.size(0)，并将结果加到total_eval[k]上，实现了累积操作。
            except:
                invalid_cnt += 1
                pass

        if only_val:  #scaler_value表示缩放因子的值  计算3D和2D IoU得分
            scaler_value = self.cfg.runners.valid_iou.batch_size * \
                           (iterator_valid_iou - invalid_cnt)
            curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
            curr_score_2d_iou = total_eval["2DIoU"] / scaler_value
            logging.info(f"3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"2D-IoU score: {curr_score_2d_iou:.4f}")
            return {"2D-IoU": curr_score_2d_iou, "3D-IoU": curr_score_3d_iou}

        scaler_value = self.cfg.runners.valid_iou.batch_size * \
                       (iterator_valid_iou - invalid_cnt) #缩放因子的计算方法是将批量大小乘以总批次数与无效结果数量之差，即 batch_size * (总批次数 - 无效结果数量)
        for k, v in total_eval.items():
            k = "valid_IoU/%s" % k
            self.tb_writer.add_scalar(
                k, v / scaler_value, self.current_epoch)

        # Save best validation loss model
        curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
        curr_score_2d_iou = total_eval["2DIoU"] / scaler_value

        # ! Saving current score
        #用于保存最佳验证损失模型的。代码首先从total_eval字典中获取键为"3DIoU"和"2DIoU"的值，并将它们除以scaler_value后分别赋值给变量curr_score_3d_iou和curr_score_2d_iou
        self.curr_scores['iou_valid_scores'] = dict(
            best_3d_iou_score=curr_score_3d_iou,
            best_2d_iou_score=curr_score_2d_iou
        )

        #这段代码主要是用于记录最佳的3D-IoU和2D-IoU分数。
        # 首先，它通过self.best_scores.get("best_iou_valid_score")来检查self.best_scores字典中是否已经存在了"best_iou_valid_score"的项。如果不存在，说明当前还没有记录最佳的分数。
        # 接下来，在日志中输出当前的3D-IoU和2D-IoU分数，使用logging.info()方法来输出这两个分数。
        # 最后，它会将最佳的分数以字典的形式存储在self.best_scores字典中的"best_iou_valid_score"项中。存储的内容包括当前的3D-IoU分数和2D-IoU分数，分别以best_3d_iou_score和best_2d_iou_score作为键。
        if self.best_scores.get("best_iou_valid_score") is None:
            logging.info(f"Best 3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"Best 2D-IoU score: {curr_score_2d_iou:.4f}")
            self.best_scores["best_iou_valid_score"] = dict(
                best_3d_iou_score=curr_score_3d_iou,
                best_2d_iou_score=curr_score_2d_iou
            )
        #这段代码是在已经存在最佳分数的情况下，对当前分数与最佳分数进行比较，并更新最佳分数。
        else:
            best_3d_iou_score = self.best_scores["best_iou_valid_score"]['best_3d_iou_score']
            best_2d_iou_score = self.best_scores["best_iou_valid_score"]['best_2d_iou_score']

            logging.info(
                f"3D-IoU: Best: {best_3d_iou_score:.4f} vs Curr:{curr_score_3d_iou:.4f}")
            logging.info(
                f"2D-IoU: Best: {best_2d_iou_score:.4f} vs Curr:{curr_score_2d_iou:.4f}")

            if best_3d_iou_score < curr_score_3d_iou:
                logging.info(
                    f"New 3D-IoU Best Score {curr_score_3d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_3d_iou_score'] = curr_score_3d_iou
                self.save_model("best_3d_iou_valid.pth")

            if best_2d_iou_score < curr_score_2d_iou:
                logging.info(
                    f"New 2D-IoU Best Score {curr_score_2d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_2d_iou_score'] = curr_score_2d_iou
                self.save_model("best_2d_iou_valid.pth")

    def valid_iou_loop(self, only_val=False):
        print_cfg_information(self.cfg)
        self.net.eval()
        iterator_valid_iou = iter(self.valid_iou_loader)#创建了一个迭代器iterator_valid_iou，用于遍历self.valid_iou_loader的数据。
        total_eval = {}#定义了一个空字典total_eval，用于存储评估结果
        invalid_cnt = 0#初始化了一个变量invalid_cnt，用于计数无效的结果数量

        for _ in trange(len(iterator_valid_iou), desc="IoU Validation epoch %d" % self.current_epoch):#for循环来遍历一个迭代器
            x, y_bon_ref, std = next(iterator_valid_iou)#在每次循环中，next(iterator_valid_iou)从迭代器中获取下一个元素
            # print('y_bon_ref的形状是 ',y_bon_ref.shape) # torch.Size([4, 2, 1024])


            with torch.no_grad():
                #y_bon_est, _ = self.net(x.to(self.device))  #改
                depth, ratio = self.net(x.to(self.device))
                print('depth shape ',depth.shape)
                b, w = depth.shape
                phi_c = []
                depth2_c = []
                depth2_f = []
                for ii in range(b):
                    floor_uv, ceiling_uv = depth2boundaries(ratio[ii].cpu().numpy(), depth[ii].cpu().numpy())

                    floor_uv_ps = uv2pixel(floor_uv)
                    floor_uv_ps = np.transpose(floor_uv_ps, (1, 0))
                    phi_c_floor = uv2phi_coords(floor_uv_ps, type_bound="floor")  # 1024

                    ceiling_uv_ps = uv2pixel(ceiling_uv)
                    ceiling_uv_ps = np.transpose(ceiling_uv_ps, (1, 0))
                    phi_c_ceiling = uv2phi_coords(ceiling_uv_ps, type_bound="ceiling")  # 1024,)

                    data_phi = np.vstack((phi_c_ceiling, phi_c_floor))
                    phi_c.append(data_phi)
                    
                y_bon_est = phi_c
                #y_bon_est = np.array(y_bon_est)
                y_bon_est = torch.Tensor(y_bon_est)
                # print('y_bon_est shape  ',y_bon_est.shape)#y_bon_est shape   (4, 2, 1024)
                # print('y_bon_ref type',type(y_bon_ref))#<class 'torch.Tensor'>
                # print('y_bon_est type', type(y_bon_est))#<class 'numpy.ndarray'>

                true_eval = {"2DIoU": [], "3DIoU": []}#定义了一个名为true_eval的字典，用于存储评估指标的结果
                for gt, est in zip(y_bon_ref.cpu().numpy(), y_bon_est.cpu().numpy()):#在y_bon_ref和y_bon_est这两个numpy数组上进行迭代。zip函数将两个数组的对应元素打包成元组，并在循环中进行迭代。
                    eval_2d3d_iuo_from_tensors(est[None], gt[None], true_eval, )#函数可能是用于计算目标估计结果和真实结果之间的2D和3D IoU评估指标，并将结果存储到true_eval中。
                #GPT:这个函数可能计算了权重化的L1损失，即对y_bon_est和y_bon_ref之间的差异进行加权求和，权重由std确定。
                local_eval = dict(loss=compute_weighted_L1(y_bon_est.to(self.device), y_bon_ref.to(self.device), std.to(self.device)))

                #2D和3D IoU的平均值，并将结果存储到local_eval字典中的相应键中
                local_eval["2DIoU"] = torch.FloatTensor(
                    [true_eval["2DIoU"]]).mean()
                local_eval["3DIoU"] = torch.FloatTensor(
                    [true_eval["3DIoU"]]).mean()
            try:
                for k, v in local_eval.items():#for循环遍历local_eval字典的键值对
                    if v.isnan():#个键值对，首先检查值v是否为NaN（Not a Number）。如果是NaN，则跳过后续代码，继续下一个迭代。
                        continue
                    total_eval[k] = total_eval.get(k, 0) + v.item() * x.size(0)#如果值v是一个number，那么通过total_eval.get(k, 0)获取total_eval字典中键k对应的值。如果该键不存在，返回默认值0。将局部评估结果乘以输入x的大小x.size(0)，并将结果加到total_eval[k]上，实现了累积操作。
            except:
                invalid_cnt += 1
                pass

        if only_val:  #scaler_value表示缩放因子的值  计算3D和2D IoU得分
            scaler_value = self.cfg.runners.valid_iou.batch_size * \
                           (len(iterator_valid_iou) - invalid_cnt)
            curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
            curr_score_2d_iou = total_eval["2DIoU"] / scaler_value
            logging.info(f"3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"2D-IoU score: {curr_score_2d_iou:.4f}")
            return {"2D-IoU": curr_score_2d_iou, "3D-IoU": curr_score_3d_iou}

        scaler_value = self.cfg.runners.valid_iou.batch_size * \
                       (len(iterator_valid_iou) - invalid_cnt) #缩放因子的计算方法是将批量大小乘以总批次数与无效结果数量之差，即 batch_size * (总批次数 - 无效结果数量)
        for k, v in total_eval.items():
            k = "valid_IoU/%s" % k
            self.tb_writer.add_scalar(
                k, v / scaler_value, self.current_epoch)

        # Save best validation loss model
        curr_score_3d_iou = total_eval["3DIoU"] / scaler_value
        curr_score_2d_iou = total_eval["2DIoU"] / scaler_value

        # ! Saving current score
        #用于保存最佳验证损失模型的。代码首先从total_eval字典中获取键为"3DIoU"和"2DIoU"的值，并将它们除以scaler_value后分别赋值给变量curr_score_3d_iou和curr_score_2d_iou
        self.curr_scores['iou_valid_scores'] = dict(
            best_3d_iou_score=curr_score_3d_iou,
            best_2d_iou_score=curr_score_2d_iou
        )

        #这段代码主要是用于记录最佳的3D-IoU和2D-IoU分数。
        # 首先，它通过self.best_scores.get("best_iou_valid_score")来检查self.best_scores字典中是否已经存在了"best_iou_valid_score"的项。如果不存在，说明当前还没有记录最佳的分数。
        # 接下来，在日志中输出当前的3D-IoU和2D-IoU分数，使用logging.info()方法来输出这两个分数。
        # 最后，它会将最佳的分数以字典的形式存储在self.best_scores字典中的"best_iou_valid_score"项中。存储的内容包括当前的3D-IoU分数和2D-IoU分数，分别以best_3d_iou_score和best_2d_iou_score作为键。
        if self.best_scores.get("best_iou_valid_score") is None:
            logging.info(f"Best 3D-IoU score: {curr_score_3d_iou:.4f}")
            logging.info(f"Best 2D-IoU score: {curr_score_2d_iou:.4f}")
            self.best_scores["best_iou_valid_score"] = dict(
                best_3d_iou_score=curr_score_3d_iou,
                best_2d_iou_score=curr_score_2d_iou
            )
        #这段代码是在已经存在最佳分数的情况下，对当前分数与最佳分数进行比较，并更新最佳分数。
        else:
            best_3d_iou_score = self.best_scores["best_iou_valid_score"]['best_3d_iou_score']
            best_2d_iou_score = self.best_scores["best_iou_valid_score"]['best_2d_iou_score']

            logging.info(
                f"3D-IoU: Best: {best_3d_iou_score:.4f} vs Curr:{curr_score_3d_iou:.4f}")
            logging.info(
                f"2D-IoU: Best: {best_2d_iou_score:.4f} vs Curr:{curr_score_2d_iou:.4f}")

            if best_3d_iou_score < curr_score_3d_iou:
                logging.info(
                    f"New 3D-IoU Best Score {curr_score_3d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_3d_iou_score'] = curr_score_3d_iou
                self.save_model("best_3d_iou_valid.pth")

            if best_2d_iou_score < curr_score_2d_iou:
                logging.info(
                    f"New 2D-IoU Best Score {curr_score_2d_iou: 0.4f}")
                self.best_scores["best_iou_valid_score"]['best_2d_iou_score'] = curr_score_2d_iou
                self.save_model("best_2d_iou_valid.pth")

    def save_model(self, filename):
        if not self.cfg.model.get("save_ckpt", True):
            return

        # ! Saving the current model
        # state_dict = OrderedDict(
        #     {
        #         "args": self.cfg,
        #         "kwargs": {
        #             "backbone": self.net.backbone,
        #             "use_rnn": self.net.use_rnn,
        #         },
        #         "state_dict": self.net.state_dict(),
        #     }
        # )
        state_dict=self.net.state_dict()
        torch.save(state_dict, os.path.join(
            self.dir_ckpt, filename))

    def prepare_for_training(self):
        self.is_training = True
        self.current_epoch = 0
        self.iterations = 0
        self.best_scores = dict()
        self.curr_scores = dict()
        self.set_optimizer()
        self.set_scheduler()
        self.set_train_dataloader()
        self.set_log_dir()
        save_cfg(os.path.join(self.dir_ckpt, 'cfg.yaml'), self.cfg)

    def set_log_dir(self):
        output_dir = os.path.join(self.cfg.output_dir, self.cfg.id_exp)#/opt/data/private/MVL_Challenge/mvl_toolkitDOPNet/mvl_challenge/assets/data/mvl_data/labels/zjsTestBaseLine/mlc__dop_mp3d__scene_list__warm_up_training_set
        create_directory(output_dir, delete_prev=True)
        logging.info(f"Output directory: {output_dir}")
        self.dir_log = os.path.join(output_dir, 'log')
        self.dir_ckpt = os.path.join(output_dir, 'ckpt')
        os.makedirs(self.dir_log, exist_ok=True)
        os.makedirs(self.dir_ckpt, exist_ok=True)

        self.tb_writer = SummaryWriter(log_dir=self.dir_log)

    def set_train_dataloader(self):
        logging.info("Setting Training Dataloader")
        self.train_loader = DataLoader(
            MVLDataLoader(self.cfg.runners.train,mode='train',),
            batch_size=self.cfg.runners.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.runners.train.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed(),
        )

    def set_valid_dataloader(self,):
        logging.info("Setting IoU Validation Dataloader")
        self.valid_iou_loader = DataLoader(
            MVLDataLoader(self.cfg.runners.valid_iou,mode='valid'),
            batch_size=self.cfg.runners.valid_iou.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.runners.valid_iou.num_workers,
            pin_memory=True if self.device != 'cpu' else False,
            worker_init_fn=lambda x: np.random.seed())
