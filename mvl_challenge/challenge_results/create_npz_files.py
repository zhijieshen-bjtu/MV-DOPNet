import argparse
import sys
sys.path.append("../..")
from mvl_challenge import (
    DATA_DIR,
    ROOT_DIR,
    CFG_DIR,
    EPILOG,
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    DEFAULT_NPZ_DIR,
    SCENE_LIST_DIR,
)

from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge.datasets.mvl_dataset import MVLDataset, iter_mvl_room_scenes
from mvl_challenge.utils.vispy_utils import plot_list_ly
from mvl_challenge.utils.image_utils import draw_boundaries_phi_coords, add_caption_to_image
from imageio import imwrite
from mvl_challenge.models.wrapper_horizon_net import WrapperHorizonNet
from mvl_challenge.utils.io_utils import (
    create_directory,
    get_scene_room_from_scene_room_idx,
    save_compressed_phi_coords,
)
from mvl_challenge.utils.image_utils import plot_image
import numpy as np
import os
from pathlib import Path


def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    cfg.scene_dir = args.scene_dir
    cfg.scene_list = args.scene_list
    cfg.ckpt = args.ckpt
    cfg.cuda_device = args.cuda_device
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    mvl = MVLDataset(cfg)
    hn = WrapperHorizonNet(cfg)

    # ! Join the output_dir and the scene_list
    output_dir = create_directory(
        os.path.join(args.output_dir, Path(args.scene_list).stem), delete_prev=False
    )
    
    if args.vis:   
        output_dir_vis = create_directory(
            os.path.join(args.output_dir, Path(args.scene_list).stem + "_vis"), delete_prev=False
        )
        
    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        for ly in list_ly:
            fn = os.path.join(output_dir, ly.idx)
            # ! IMPORTANT: Use ALWAYS save_compressed_phi_coords()
            save_compressed_phi_coords(ly.phi_coords, fn)

            if args.vis:
                # ! Save visualizations
                img = ly.get_rgb()
                draw_boundaries_phi_coords(
                    img, phi_coords=np.vstack([ly.phi_coords[0], ly.phi_coords[1]])
                )
                img = add_caption_to_image(img, ly.idx)

                fn = os.path.join(output_dir_vis, f"{ly.idx}.jpg")
                imwrite(fn, img)
                


def get_argparse():
    desc = (
        "This script saves the evaluations (phi_coords) of HorizonNet given a MVL dataset, scene list and cfg file. "
        + "The scene directory is where the MVL data is stored. "
        + "The scene list is the list of scene in scene_room_idx format. "
        + "The cfg file is the yaml configuration with all hyperparameters set to default values."
    )

    parser = argparse.ArgumentParser(description=desc, epilog=EPILOG)

    parser.add_argument(
        "-d",
        "--scene_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}",
        help="MVL dataset directory.",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{CFG_DIR}/eval_mvl_dataset.yaml",
        help=f"Config file to load a MVL dataset. For this script model cfg most be defined in the cfg file too. (Default: {CFG_DIR}/eval_mvl_dataset.yaml)",
    )

    parser.add_argument(
        "-f",
        "--scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__challenge_phase_testing_set.json",#scene_list__warm_up_pilot_set.json",
        help="Scene_list of mvl scenes in scene_room_idx format.",
    )

    parser.add_argument(
        "--ckpt",
        default=f"{ASSETS_DIR}/ckpt/best_3d_iou_valid.pth",#hn_mp3d.pth",
        help="Pretrained model ckpt (Default: mp3d)",
    )

    parser.add_argument("--cuda_device", default=0, type=int, help="Cuda device. (Default: 0)")

    parser.add_argument(
        "-o",
        "--output_dir",
        default=f"{DEFAULT_NPZ_DIR}",
        help="Output directory where to store phi_coords estimations.",
    )

    parser.add_argument(
        "--vis",
        action="store_true",
        help="Output directory where to store phi_coords estimations.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    main(args)
