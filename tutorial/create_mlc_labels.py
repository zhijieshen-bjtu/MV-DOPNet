import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
# from mlc.mlc import compute_pseudo_labels
import sys
sys.path.append("../..")
from mvl_challenge.models.MLC360.mlc.mlc import compute_pseudo_labels

from mvl_challenge.datasets.mvl_dataset import iter_mvl_room_scenes
from mvl_challenge.config.cfg import read_omega_cfg
from mvl_challenge import DEFAULT_MVL_DIR
from mvl_challenge.utils.spherical_utils import xyz2uv
from mvl_challenge.utils.io_utils import create_directory, save_compressed_phi_coords
from mvl_challenge.datasets.mvl_dataset import MVLDataset
from mvl_challenge.models.wrapper_dop_net import WrapperDOPNet
from mvl_challenge.utils.image_utils import (
    draw_boundaries_uv,
    draw_uncertainty_map,
    COLOR_MAGENTA, COLOR_CYAN)
from mvl_challenge import (
    ASSETS_DIR,
    DEFAULT_MVL_DIR,
    SCENE_LIST_DIR,
)

MLC_TUTORIAL_DIR = os.path.dirname(__file__)


def create_mlc_label_dirs(cfg):
    """
    Create MLC pseudo label the directories
    """
    create_directory(os.path.join(cfg.output_dir, cfg.id_exp), delete_prev=True)
    create_directory(cfg.mlc_dir.phi_coords, delete_prev=True)
    create_directory(cfg.mlc_dir.std, delete_prev=True)
    create_directory(cfg.mlc_dir.vis, delete_prev=True)


def save_visualization(fn, img_boundaries, sigma_map):
    plt.figure(0, dpi=200)
    plt.clf()
    plt.subplot(211)
    plt.suptitle(Path(fn).stem)
    plt.imshow(img_boundaries)
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(sigma_map)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def compute_and_save_mlc_labels(list_ly):
    for ref in tqdm(list_ly, desc="Estimating MLC Labels"):
        uv_ceiling_ps, uv_floor_ps, std_ceiling, std_floor, _ = compute_pseudo_labels(
            list_frames=list_ly,
            ref_frame=ref,
        )

        # ! Saving pseudo labels
        v = np.vstack((uv_ceiling_ps[1], uv_floor_ps[1]))
        std = np.vstack((std_ceiling, std_floor))
        phi_coords = (v / 512 - 0.5) * np.pi

        # ! NOTE: 360-MLC expects npy files as pseudo labels
        fn = os.path.join(ref.cfg.mlc_dir.phi_coords, f"{ref.idx}")
        np.save(fn, phi_coords)
        fn = os.path.join(ref.cfg.mlc_dir.std, f"{ref.idx}")
        np.save(fn, std)

        uv_ceiling_hat = xyz2uv(ref.bearings_ceiling)
        uv_floor_hat = xyz2uv(ref.bearings_floor)

        img = ref.get_rgb()
        
        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_ceiling_hat,
            color=COLOR_CYAN
        )

        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_floor_hat,
            color=COLOR_CYAN
        )

        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_ceiling_ps,
            color=COLOR_MAGENTA
        )

        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_floor_ps,
            color=COLOR_MAGENTA
        )

        sigma_map = draw_uncertainty_map(
            peak_boundary=np.hstack((uv_ceiling_ps, uv_floor_ps)),
            sigma_boundary=np.hstack((std_ceiling, std_floor))
        )

        fn = os.path.join(ref.cfg.mlc_dir.vis, f"{ref.idx}.jpg")
        #new avg ratio写入的部分
        new_gemo_info_fn=os.path.join(ref.cfg.mvl_dir,'geometry_info')
        waited_write_json=os.path.join(new_gemo_info_fn,f"{ref.idx}.json")
        # print('waited_write_json',waited_write_json)
        # print('ref.ratio',ref.ratio)
        with open(waited_write_json,'r') as f:
            content = json.load(f)
        newratio={"ratio": ref.ratio}

        content.update(newratio)
        with open(waited_write_json,'w') as f_new:
            json.dump(content,f_new)
        #准确的 ratio写入的部分
        # new_gemo_info_fn = os.path.join(ref.cfg.mvl_dir, 'geometry_info_accnew')
        # waited_write_json = os.path.join(new_gemo_info_fn, f"{ref.idx}.json")
        # with open(waited_write_json, 'r') as f:
        #     content = json.load(f)
        # newratio = {"accuratio": ref.ratio}
        # content.update(newratio)
        # with open(waited_write_json, 'w') as f_new:
        #     json.dump(content, f_new)


        save_visualization(fn, img, sigma_map)


def get_cfg_from_args(args):
    cfg = read_omega_cfg(args.cfg)
    if cfg.pass_args:  # params in the yaml will be replaced by the passed arguments
        cfg.mvl_dir = args.scene_dir
        cfg.scene_list = args.scene_list
        cfg.output_dir = args.output_dir
        cfg.ckpt = args.ckpt
        cfg.cuda_device = args.cuda_device
        cfg.id_exp = f"mlc__{Path(cfg.ckpt).stem}__{Path(args.scene_list).stem}"
    return cfg


def main(args):
    # ! Reading configuration
    cfg = get_cfg_from_args(args)

    mvl = MVLDataset(cfg)
    hn = WrapperDOPNet(cfg)

    mvl.print_mvl_data_info()
    create_mlc_label_dirs(cfg)

    for list_ly in iter_mvl_room_scenes(model=hn, dataset=mvl):
        compute_and_save_mlc_labels(list_ly)


def get_passed_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        default=f"{MLC_TUTORIAL_DIR}/create_mlc_labels.yaml",
        help=f'Config File. Default {MLC_TUTORIAL_DIR}/create_mlc_labels.yaml')

    parser.add_argument(
        "-f",
        "--scene_list",
        type=str,
        default=f"{SCENE_LIST_DIR}/scene_list__warm_up_pilot_set.json",
        # scene_list__challenge_phase_training_set.json#scene_list__warm_up_training_set.json
        help="Scene_list of mvl scenes in scene_room_idx format.",
    )

    parser.add_argument(
        "-d",
        "--scene_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}",
        help="MVL dataset directory.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=f"{DEFAULT_MVL_DIR}/labels",
        help="MVL dataset directory.",
    )

    parser.add_argument(
        "--ckpt",
        default=f"{ASSETS_DIR}/ckpt/dop_mp3d.pkl",
        help="Path to ckpt pretrained model (Default: mp3d)",
    )

    parser.add_argument("--cuda_device", default=0, type=int, help="Cuda device. (Default: 0)")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_passed_args()
    main(args)
