import numpy as np
import logging
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from mlc.scale_recover.scale_recover import ScaleRecover
from mlc.utils.image_utils import draw_boundaries_uv, draw_uncertainty_map
from mlc.utils.color_utils import *
from mlc.utils.layout_utils import filter_out_noisy_layouts
from mlc.utils.geometry_utils import extend_array_to_homogeneous, stack_camera_poses
from mlc.utils.projection_utils import uv2xyz, uv2sph, xyz2uv
from mlc.utils.bayesian_utils import apply_kernel
from mlc.models.utils import load_layout_model
from mlc.datasets.utils import load_mvl_dataset


#
# def reproject_ly_boundaries_Align(list_ly, ref_ly, shape=(512, 1024)):
#     pose_W2ref = np.linalg.inv(ref_ly.pose.SE3_scaled())
#     boundaries_xyz = []
#     [(boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_floor)),
#       boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_ceiling)))
#      for ly in list_ly]
#     uv = xyz2uv(np.hstack(boundaries_xyz), shape)
#     return uv



'''
��δ��붨����һ����Ϊ reproject_ly_boundaries �ĺ���������һ�����֣�layout���б� list_ly ����ͶӰ���ο����� ref_ly ��λ�ã������ظò��ֵı߽��� shape ��״��ͼ���ϵ�����λ�á�
'''
def reproject_ly_boundaries(list_ly, ref_ly, shape=(512, 1024)):
    print('ȷ��һ���ǲ��������mlc.py')
    """
    Reproject a list of layouts into the passed reference using comprehension lists
    """
    '''
    ��δ����Ŀ���Ǽ���ο����֣�ref_ly���������������ϵ�ı任�����������Ա㽫�������֣�ly��ͶӰ���ο����ֵ�λ�á�
    ������һ�� ref_ly.pose.SE3_scaled()����������ķ���ֵ�ǲο������������������ϵ�ı任����
    �����˵��������һ�� SE3 ���󣬱�ʾ�ο����ִ���������ϵ��ԭ�㣨�������ο��㣩��ʼ��ƽ�ƺ���ת�任��
    SE3 ������һ�� 4x4 �ľ��󣬿���������ʾ��ά����ϵ�еı任������ǰ��������ת�������һ����ƽ��������
    Ȼ��np.linalg.inv ������ numpy ���е�һ�����������ڼ���һ������������
    ���ԣ�np.linalg.inv(ref_ly.pose.SE3_scaled()) �����þ�������ο������������������ϵ�ı任�������������������ں���Ĵ��������ڽ���������ͶӰ���ο����ֵ�λ���ϡ�
    '''
    pose_W2ref = np.linalg.inv(ref_ly.pose.SE3_scaled())#����ο������������������ϵ�ı任����������

    boundaries_xyz = []#����һ�����б�׼���洢�����
    #ʹ���б��Ƶ�ʽ���� list_ly �е�ÿ������ ly������ײ��Ͷ����߽�������ת��Ϊ�ο���������ϵ�еĵ㣬
    # Ȼ��������ӵ� boundaries_xyz �б��С�ע�⵽���д���ʵ����ִ�е���һ��ѭ��������ʹ�����б��Ƶ�ʽ���﷨��
    '''
    [(..., ...) for ly in list_ly]������һ���б��Ƶ�ʽ���﷨����ʾҪ�� list_ly �е�ÿ��Ԫ�� ly ���в���������ʡ�Ժ� <...> ��ʾ�� ly ���еĲ�����
    pose_W2ref[:3, :]��������ʽ��һ�� 3x4 �ľ��󣬱�ʾ�ο������еĵ�����������ϵ�е�����ֵ��pose_W2ref ����ǰ�����Ĳο������������������ϵ�ı任����������
    pose_W2ref[:3, :] ȡ������ǰ���У���ʾ��ת�����ƽ������������ת������Ӷ��ܹ�����������ͶӰ���ο������е�λ���ϡ�
    ly.boundary_floor �� ly.boundary_ceiling �ֱ��ʾ���ֵײ��Ͷ����ı߽�㼯�ϣ���һ�� N x 3 �ľ������� N ��ʾ���������ÿ��������������ֵ��ʾ��
    extend_array_to_homogeneous ������ά��Ϊ d ���������䵽 d+1 ά��������һά����ֵ��Ϊ 1�����ڽ��㼯��ʾΪ������꣬�Ӷ�ʹ����˷��ܹ����㼯�任����ȷ��λ���ϡ�
    @ �������ʾ����˷������ο���������ϵ�еĵ�ת��Ϊ��������ϵ�еĵ㡣
    ��ת����������ֱ���ӵ� boundaries_xyz �б��С������������߽�㣨�ײ��Ͷ������������������ append ������
    ���գ�boundaries_xyz �б���������в��ֵı߽�����������ϵ�е�ֵ�����Խ���Щ��ͶӰ����άƽ���ϵõ��������ꡣ
    '''
    [(boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_floor)),
      boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_ceiling)))
     for ly in list_ly]

    # [(boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(
    #     ly.boundary_floor[:, ly.cam2boundary < np.quantile(ly.cam2boundary, 0.5)])),
    #   boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(
    #       ly.boundary_ceiling[:, ly.cam2boundary < np.quantile(ly.cam2boundary, 0.5)])))
    #  for ly in list_ly]

    # [boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_ceiling))
    #  for ly in list_ly]

    #�� boundaries_xyz �б��е����е�ϲ���һ�����飬Ȼ��ʹ�� xyz2uv ��������Щ�����ά�ռ�ͶӰ����άͼ��ƽ���ϡ��õ��� uv ��һ����ά���飬��ʾ�� shape ��״��ͼ���ϵ�����λ�á�
    '''
    np.hstack(boundaries_xyz)����������Ƚ�֮ǰ����õ������б߽�����������ϵ�е�ֵ��ϳ�һ�������顣
    boundaries_xyz ��һ���б�����ÿ��Ԫ�ض���һ�����󣬱�ʾһ�������еı߽�㼯�ϡ�np.hstack �������б��е����о��������ӣ������ŵڶ���ά����ϣ����γ�һ��һά���顣
    xyz2uv ������������������������е��һά���飬��ͼ��Ĵ�С shape��һ����Ԫ�飬��ʾ�����������������Ὣ���е����ά����ϵ��ͶӰ����άͼ��ƽ���ϣ�������ͶӰ����������ꡣע�⣬������û���ṩ xyz2uv ������ʵ�֣���˲���������ʵ�ַ�ʽ��
    uv = xyz2uv(np.hstack(boundaries_xyz), shape)����������е��һά�����ͼ���С���ݸ� xyz2uv ��������ͶӰ�����õ��Ľ���洢��һ����Ϊ uv �ı����С�uv ��һ����ά���飬��ʾ���б߽���ڶ�άͼ��ƽ���ϵ���������λ�á�
    '''
    uv = xyz2uv(np.hstack(boundaries_xyz), shape)
    print('��mlc������uv shape',uv.shape)
    print('��mlc������uv',uv)

    #uv2depth�õ�ÿ��֡�����depth,�Ż����depth

    return uv

#����compute_pseudo_labels��������������list_frames��һ����֡��ɵ��б�ref_frame�Ǳ��ο���֡��shape��ָ����״��Ԫ�飬���ڶ����������״��
# �ú��������б��е�֡�Ͳο�֡ref_frame����α��ǩ
def compute_pseudo_labels(list_frames, ref_frame, shape=(512, 1024)):
    """
    Computes the pseudo labels based on the passed list of frames wrt the ref_frame
    """
    cfg = ref_frame.cfg
    # ����reproject_ly_boundaries()��������list_frames�еı߽��ͶӰ���ο�֡ref_frame��ͼ������ϵ�У��������ڱ���uv�С�
    uv = reproject_ly_boundaries(list_frames, ref_frame, shape)#�����������Ҫ�ҵ�0626

    # Ȼ��uv�е����е����������ڱ�ǩ��״��Χ��
    uv[0] = np.clip(uv[0], 0, shape[1] - 1)#np.clip(uv[0], 0, shape[1] - 1)����uv����ĵ�һ��ά�ȣ���uv[0]���е�ֵ������0��shape[1] - 1֮�䡣�������ȷ����uv[0]�е�ֵ���ᳬ�������ı߽緶Χ��
    uv[1] = np.clip(uv[1], 0, shape[0] - 1)#np.clip(uv[1], 0, shape[0] - 1)����uv����ĵڶ���ά�ȣ���uv[1]���е�ֵ������0��shape[0] - 1֮�䡣ͬ�����������ȷ����uv[1]�е�ֵ���ᳬ�������ı߽緶Χ��

    #���ţ���uv����ͬ����ĵ㰴�ճ��ִ���ͳ�Ƶ�һ��ֱ��ͼ�У������ظ�ֱ��ͼ����������Ϣ��
    '''
    ��δ���ʵ�������¹��ܣ�
    ���ȣ�ʹ��np.unique����������uv�е�Ψһ������������ִ����ֱ�洢��uv��counts�С�
    ������������һ����״Ϊshape��ȫ������prj_map����Ϊ���յ�ͶӰӳ�䡣
    ʹ��uv�е��к�����������counts�е�ֵ����prj_map��Ӧλ���ϵ�Ԫ�ء�������prj_map�еķ���Ԫ�ر�ʾ��ԭʼӳ����ÿ������ֵ��ͶӰƽ���ϵĳ��ִ�����
    ���ͨ��copy��������_prj_map������prj_map�ĸ�����
    ��δ�����Ҫ��������ͶӰӳ�䣬����ÿ������ֵ�ĳ��ִ�����ʾ����ԭʼӳ���ж�Ӧ����ֵ�����ص���Ŀ
    '''
    # ! Aggregating projected points into a histogram
    uv, counts = np.unique(list(uv.T), return_counts=True, axis=0)

    prj_map = np.zeros(shape)
    prj_map[uv[:, 1], uv[:, 0]] = counts

    _prj_map = prj_map.copy()

    '''
    ֮�󣬽�ֱ��ͼ�е����ݸ���ͼ��ĵײ��Ͷ������зָ�ֱ������ײ��Ͷ����ı�ǩ��
    ���ڵײ���ǩ�����Ƚ�ֱ��ͼ�°벿����ȡ������Ȼ���ڸò������ҵ����ֵ��λ�ã���λ�ñ�ʾ�ײ���ǩ�Ĵ�ֱλ�ã�ͬʱ�����λ�õķ��
    ���ڶ�����ǩ��ͬ�����Ƚ�ֱ��ͼ�ϰ벿����ȡ������Ȼ���ҵ��ò������ֵ��λ�ã������λ�õķ��������м������õı�ǩλ�úͷ�����Ϣ�Լ�ֱ��ͼ���ء�
    '''

    '''
    ��δ�����һ��������ִ�����²�����
    �������floor_map��ceiling_map���ֱ��ʾ�ذ���컨���ӳ�䡣
    ʹ��np.argmax������floor_map��ceiling_map��ÿһ�����ҵ����ֵ����������ʾ�ذ���컨��Ĵ�ֱλ�á�
    ����get_std��������ذ���컨��ӳ��ı�׼�
    ����ƽ���ֲ���ˮƽ����u��
    ���صذ���컨��Ĵ�ֱλ�ã��ذ���컨��ı�׼���Լ�ԭʼӳ��_prj_map��  
    '''
    # ! Floor
    floor_map = _prj_map[512//2:, :]#�����_prj_map��ѡȡ��һ������Ϊfloor_map��������˵��floor_map�Ǵ�_prj_map����ĵ�512//2�У����ӵذ�λ�ÿ�ʼ����ʼ�����һ�е������У�����ʾ�ذ�Ĳ���
    v_floor = np.argmax(floor_map, axis=0)
    std_floor = get_std(floor_map, v_floor, cfg.runners.mvl.std_kernel)
    v_floor += 512//2

    # ! Ceiling
    ceiling_map = _prj_map[:512//2, :]
    v_ceiling = np.argmax(ceiling_map, axis=0)
    std_ceiling = get_std(ceiling_map, v_ceiling, cfg.runners.mvl.std_kernel)

    u = np.linspace(0, ceiling_map.shape[1]-1, ceiling_map.shape[1]).astype(int)
    return np.vstack((u, v_ceiling)), np.vstack((u, v_floor)), std_ceiling, std_floor, _prj_map


def get_std(hw_map, peak, kernel):
    """
    https://www.statology.org/histogram-standard-deviation/
    https://math.stackexchange.com/questions/857566/how-to-get-the-standard-deviation-of-a-given-histogram-image
    """
    # ! To avoid zero STD
    hw_map = apply_kernel(hw_map.copy(), size=(kernel[0], kernel[1]), sigma=kernel[2])
    m = np.linspace(0, hw_map.shape[0]-1, hw_map.shape[0]) + 0.5
    miu = np.repeat(peak.reshape(1, -1), hw_map.shape[0], axis=0)
    mm = np.repeat(m.reshape(-1, 1), hw_map.shape[1], axis=1)
    N = np.sum(hw_map, axis=0)
    std = np.sqrt(np.sum(hw_map*(mm-miu)**2, axis=0)/N)
    return std / hw_map.shape[0]


def median(c):
    px_val = c[c > 0]
    med_val = np.median(px_val)
    std_val = np.std(px_val)
    v_val = np.argmin(abs(c - med_val))
    return v_val, std_val


def iterator_room_scenes(cfg):
    """
    Creates a generator which yields a list of layout from a defined 
    dataset.
    """
    model = load_layout_model(cfg)
    dataset = load_mvl_dataset(cfg)
    scale_recover = ScaleRecover(cfg)
    dataset.load_imgs = True
    dataset.load_gt_labels = False
    dataset.load_npy = False

    for scene in tqdm(dataset.list_scenes, desc="Reading MVL scenes..."):
        cfg._room_scene = scene
        logging.info(f"Scene Name: {scene}")
        logging.info(f"Experiment ID: {cfg.id_exp}")
        logging.info(f"Output_dir: {cfg.output_dir}")
        logging.info(f"Iteration: {dataset.list_scenes.index(scene)}/{dataset.list_scenes.__len__()}")
        list_ly = dataset.get_list_ly(scene_name=scene)

        # ! Overwrite phi_coord by the estimates
        model.estimate_within_list_ly(list_ly)
        filter_out_noisy_layouts(
            list_ly=list_ly,
            max_room_factor_size=cfg.runners.mvl.max_room_factor_size
        )
        if cfg.runners.mvl.apply_scale_recover:
            scale_recover.fully_vo_scale_estimation(list_ly=list_ly)

        yield list_ly


def compute_and_store_mlc_labels(list_ly, save_vis=False):
    _output_dir = list_ly[0].cfg.output_dir
    _cfg = list_ly[0].cfg
    for ref in tqdm(list_ly, desc="Estimating MLC Labels"):
        uv_ceiling_ps, uv_floor_ps, std_ceiling, std_floor, prj_map = compute_pseudo_labels(
            list_frames=list_ly,
            ref_frame=ref,
        )

        # ! Saving pseudo labels
        uv = np.vstack((uv_ceiling_ps[1], uv_floor_ps[1]))
        std = np.vstack((std_ceiling, std_floor))
        phi_bon = (uv / 512 - 0.5) * np.pi
        np.save(os.path.join(_output_dir, "label", _cfg.runners.mvl.label, ref.idx), phi_bon)
        np.save(os.path.join(_output_dir, "label", "std", ref.idx), std)

        if not save_vis:
            return

        uv_ceiling_hat = xyz2uv(ref.bearings_ceiling)
        uv_floor_hat = xyz2uv(ref.bearings_floor)
        img = ref.img.copy()

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

        plt.figure(0, dpi=200)
        plt.clf()
        plt.subplot(211)
        plt.suptitle(ref.idx)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(sigma_map)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(os.path.join(_output_dir, "vis",
                    f"{ref.idx}.jpg"), bbox_inches='tight')
