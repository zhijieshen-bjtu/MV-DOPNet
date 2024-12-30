import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from imageio import imwrite
import sys
sys.path.append("../..")
from mvl_challenge.models.models_utils import load_layout_model
from mvl_challenge.datasets.mvl_dataset import iter_mvl_room_scenes
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, MVLDataset_val, MVLDataset_train
from mvl_challenge import (
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    SCENE_LIST_DIR,
    DEFAULT_TRAINING_DIR
)

MLC_TUTORIAL_DIR=os.path.dirname(__file__)

def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    if cfg.pass_args:   # params in the yaml will be replaced by the passed arguments
        cfg.mvl_dir = args.scene_dir  #mvl_challenge/assets/data/mvl_data
        cfg.training_scene_list = args.training_scene_list #mvl_challenge/data/scene_list/scene_list__challenge_phase_training_set.json
        cfg.pilot_scene_list = args.pilot_scene_list #mvl_challenge/data/scene_list/scene_list__warm_up_pilot_set.json
        cfg.output_dir = args.output_dir #mvl_challenge/assets/data/mvl_training_results
        cfg.ckpt = args.ckpt #MVL_Challenge/mvl_toolkitDOPNet/mvl_challenge/assets/ckpt/dop_mp3d.pkl
        cfg.cuda_device = args.cuda_device
        cfg.id_exp = "mlc__dop_zindfine__scene_list__challenge_phase_training_set"#f"mlc__{Path(cfg.ckpt).stem}__{Path(args.training_scene_list).stem}"#??maybe    mlc__dop_mp3d__scene_list__challenge_phase_training_set
        cfg.id_exp_val = "mlc__dop_zindfine__scene_list__warm_up_pilot_set"#f"mlc__{Path(cfg.ckpt).stem}__{Path(args.training_scene_list).stem}"#??maybe    mlc__dop_mp3d__scene_list__challenge_phase_training_set
    return cfg

def main(args):
    # ! Reading configuration
    cfg = get_cfg_from_args(args)

    #model = load_layout_model(cfg)
    
    mvl_train = MVLDataset_train(cfg, mode="train")
    mvl_val = MVLDataset_val(cfg, mode="val")

    model = load_layout_model(cfg)
    model.prepare_for_training()
    
    model.valid_iou_within_list_ly(mvl_val)
    
    while model.is_training:
        model.train_iou_within_list_ly(mvl_train)
        model.valid_iou_within_list_ly(mvl_val)
        model.save_current_scores()
        
def get_passed_args():
    parser = argparse.ArgumentParser()
    
    default_cfg = f"{MLC_TUTORIAL_DIR}/train_mlc.yaml"   #MVL_Challenge/mvl_toolkitDOPNet/tutorial/train_360_mlc/

    parser.add_argument(
        '--cfg',
        default=default_cfg,
        help=f'Config File. Default {default_cfg}')
    
    parser.add_argument(
        "--training_scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__challenge_phase_training_set.json",#mvl_challenge/data/scene_list/scene_list__challenge_phase_training_set.json         #scene_list__warm_up_training_set.json
        help="Training scene list.",
    )
    
    parser.add_argument(
        "--pilot_scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_pilot_set.json",#mvl_challenge/data/scene_list/...json
        help="Pilot scene list",
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{DEFAULT_TRAINING_DIR}",#mvl_challenge/assets/data/mvl_training_results
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "-d",
        "--scene_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}",##mvl_challenge/assets/data/mvl_data    #这个里面有未来labels的输出，有数据集img，有几何信息geometry_info
        help="MVL dataset directory.",
    )
    
    parser.add_argument(
        "--ckpt",
        default=f"{ASSETS_DIR}/ckpt/dop_zind.pkl",#hn_mp3d.pth",        #MVL_Challenge/mvl_toolkitDOPNet/mvl_challenge/assets/ckpt/dop_mp3d.pkl
        help="Path to ckpt pretrained model (Default: mp3d)",
    )
    
    parser.add_argument("--cuda_device", default=0, type=int, help="Cuda device. (Default: 0)")

    args = parser.parse_args()
    return args
         
if __name__ == "__main__":
    args = get_passed_args()
    main(args)