import os
import json
import numpy as np

from mvl_challenge.utils.geometry_utils import extend_array_to_homogeneous
from mvl_challenge.utils.spherical_utils import phi_coords2xyz
from mvl_challenge.models.DOPNet.utils.conversion import depth2pixel, uv2pixel,uv2depth,xyz2depth,pixel2uv
from mvl_challenge.utils.spherical_utils import uv2phi_coords,phi_coords2xyz,phi_coords2uv

from .cam_pose import CAM_REF
from imageio import imread
#import numpy as np
import pathlib
import torch.utils.data as data
from PIL import Image
#import json
import torch
import logging
from imageio import imread


class Layout:
    @property
    def phi_coords(self):
        return self.__phi_coords

    @phi_coords.setter
    def phi_coords(self, value):
        self.__phi_coords = value
        if value is None:
            return
        self.recompute_ly_geometry()

    def set_phi_coords(self, phi_coords):
        self.phi_coords = phi_coords

    def set_ratio(self, ratio):
        self.ratio = ratio


    def __init__(self, cfg):
        self.cfg = cfg

        self.boundary_floor = None
        self.boundary_ceiling = None

        self.ratio=None

        self.cam2boundary = None
        self.cam2boundary_mask = None

        self.bearings_floor = None
        self.bearings_ceiling = None

        self.img_fn = ""
        self.pose = np.eye(4)
        self.idx = ""
        #for training 
        self.img_fn = ""
        self.gt_label_fn = ""
        self.ps_label_fn = ""
        self.std_fn = ""
        self.stdr_fn = ""
        self.ratio_fn = ""
        
        self.img = None
        self.gt_label = None
        self.ps_label = None
        self.std = None
        self.depth = None
        #self.ratio = ratio_fn

        self.phi_coords = None
        self.cam_ref = CAM_REF.CC
        self.ceiling_height = None  # ! Must be None by default
        self.camera_height = 1
        self.primary = False
        self.scale = 1

        # ! data for normalize boundaries
        self.bound_scale = 1
        self.bound_center = np.zeros((3, 1))

    def apply_vo_scale(self, scale):

        if self.cam_ref == CAM_REF.WC_SO3:
            self.boundary_floor = self.boundary_floor + (
                scale / self.pose.vo_scale
            ) * np.ones_like(self.boundary_floor) * self.pose.t.reshape(3, 1)

            self.boundary_ceiling = self.boundary_ceiling + (
                scale / self.pose.vo_scale
            ) * np.ones_like(self.boundary_ceiling) * self.pose.t.reshape(3, 1)

            self.cam_ref = CAM_REF.WC

        elif self.cam_ref == CAM_REF.WC:
            delta_scale = scale - self.pose.vo_scale
            self.boundary_floor = self.boundary_floor + (
                delta_scale / self.pose.vo_scale
            ) * np.ones_like(self.boundary_floor) * self.pose.t.reshape(3, 1)

            self.boundary_ceiling = self.boundary_ceiling + (
                delta_scale / self.pose.vo_scale
            ) * np.ones_like(self.boundary_ceiling) * self.pose.t.reshape(3, 1)

        self.pose.vo_scale = scale

        return True

    def estimate_height_ratio(self):
        """
        Estimates the height ratio that describes the distance ratio of camera-ceboundary_ceiling over theboundary_ceiling-ceiling distance.
        This information is important to recover the 3D
        structure of the predicted Layout
        """

        floor = np.abs(self.ly_data[1, :])
        ceiling = np.abs(self.ly_data[0, :])

        ceiling[ceiling > np.radians(80)] = np.radians(80)
        ceiling[ceiling < np.radians(5)] = np.radians(5)
        floor[floor > np.radians(80)] = np.radians(80)
        floor[floor < np.radians(5)] = np.radians(5)

        self.height_ratio = np.mean(np.tan(ceiling) / np.tan(floor))

    def compute_cam2boundary(self):
        """
        Computes the horizontal distance for every boundary point w.r.t camera pose.
        The boundary can be in any reference coordinates
        """
        if self.cam_ref == CAM_REF.WC_SO3 or self.cam_ref == CAM_REF.CC:
            # ! Boundary reference still in camera reference
            self.cam2boundary = np.linalg.norm(self.boundary_floor[(0, 2), :], axis=0)

        else:
            assert self.cam_ref == CAM_REF.WC
            pcl = np.linalg.inv(self.pose.SE3_scaled())[
                :3, :
            ] @ extend_array_to_homogeneous(self.boundary_floor)
            self.cam2boundary = np.linalg.norm(pcl[(0, 2), :], axis=0)

        # self.cam2boundary_mask = np.zeros_like(self.cam2boundary)
        # self.cam2boundary_mask = self.cam2boundary < np.quantile(self.cam2boundary, 0.25)

    def recompute_ly_geometry(self):

        # ! Compute bearings
        self.bearings_ceiling = phi_coords2xyz(phi_coords=self.phi_coords[0, :])
        self.bearings_floor = phi_coords2xyz(phi_coords=self.phi_coords[1, :])

        #print('self.bearings_ceiling的形状是 ',self.bearings_ceiling.shape)#(3,1024)
        #print('self.bearings_floor的形状是 ', self.bearings_floor.shape)

        # ! Compute floor boundary
        ly_scale = self.camera_height / self.bearings_floor[1, :]
        pcl = ly_scale * self.bearings_floor * self.scale
        self.cam_ref = CAM_REF.WC
        self.boundary_floor = self.pose.SE3_scaled()[
            :3, :
        ] @ extend_array_to_homogeneous(pcl)

        #print('self.boundary_floor的形状是 ', self.boundary_floor.shape)


        # from mlc.utils.vispy_utils.vispy_utils import plot_pcl
        # ! Compute ceiling boundary
        if self.ceiling_height is None:
            # ! forcing consistency between floor and ceiling
            scale_ceil = np.linalg.norm(pcl[(0, 2), :], axis=0) / np.linalg.norm(
                self.bearings_ceiling[(0, 2), :], axis=0
            )
            pcl = scale_ceil * self.bearings_ceiling
            # plot_pcl(pcl)
        else:
            ly_scale = (
                self.ceiling_height - self.camera_height
            ) / self.bearings_ceiling[1, :]
            pcl = ly_scale * self.bearings_ceiling * self.scale

        self.boundary_ceiling = self.pose.SE3_scaled()[
            :3, :
        ] @ extend_array_to_homogeneous(pcl)
        #print('self.boundary_ceiling的形状是 ', self.boundary_ceiling.shape)
        self.compute_cam2boundary()

    def transform_to_WC_SO3(self):

        self.boundary_ceiling = self.boundary_ceiling - self.pose.t.reshape(3, 1)
        self.boundary_floor = self.boundary_floor - self.pose.t.reshape(3, 1)

        self.cam_ref = CAM_REF.WC_SO3

    def normalize_boundaries(self, scale, center):
        assert (
            scale > 0
        ), "Zero or negative scale is not allowed to normalize a layout bondary"
        if self.bound_scale != scale:
            self.bound_scale = scale
        if np.linalg.norm(self.bound_center) != np.linalg.norm(center):
            self.bound_center = center

        self.boundary_floor = self.apply_normalization(self.bearings_floor)
        self.boundary_ceiling = self.apply_normalization(self.bearings_ceiling)

    def get_rgb(self, mode):
        if mode == "train":
          filename = os.path.splitext(self.idx)[0]
          ps_label_fn = os.path.join(self.cfg.runners.train.data_dir.labels_dir, self.cfg.runners.train.label, f"{filename}")
              
          std_fn = os.path.join(self.cfg.runners.train.data_dir.labels_dir, 'std', f"{filename}.npy")
          stdr_fn = os.path.join(self.cfg.runners.train.data_dir.labels_dir, 'room_ratio', f"{filename}.npy")
          image_fn = os.path.join(self.cfg.runners.train.data_dir.img_dir, f"{filename}")
  
          ratio_fn=os.path.join(self.cfg.runners.train.data_dir.ratio_newdir,f"{filename}.json")
  
          
          if os.path.exists(image_fn + '.jpg'):
              image_fn += '.jpg'
          elif os.path.exists(image_fn + '.png'):
              image_fn += '.png'
          self.img_fn = image_fn
          self.gt_label_fn = None
          self.ps_label_fn = ps_label_fn
          self.std_fn = std_fn
          self.stdr_fn = stdr_fn
          self.ratio_fn = ratio_fn
        if mode == "val":
          filename = os.path.splitext(self.idx)[0]
          ps_label_fn = os.path.join(self.cfg.runners.valid_iou.data_dir.pseudo_labels_dir, self.cfg.runners.valid_iou.pseudo_label, f"{filename}")
          std_fn = os.path.join(self.cfg.runners.train.data_dir.labels_dir, 'std', f"{filename}.npy")
          image_fn = os.path.join(self.cfg.runners.train.data_dir.img_dir, f"{filename}")
  
          ratio_fn=os.path.join(self.cfg.runners.train.data_dir.ratio_dir,f"{filename}.json")
  
          
          if os.path.exists(image_fn + '.jpg'):
              image_fn += '.jpg'
          elif os.path.exists(image_fn + '.png'):
              image_fn += '.png'
          gt_label_fn = os.path.join(self.cfg.runners.valid_iou.data_dir.labels_dir, self.cfg.runners.valid_iou.label, f"{filename}")
          self.img_fn = image_fn
          self.gt_label_fn = gt_label_fn
          self.ps_label_fn = ps_label_fn
          self.std_fn = std_fn
          self.ratio_fn = ratio_fn

    def apply_normalization(self, xyz):
        return (xyz - self.bound_center)/self.bound_scale