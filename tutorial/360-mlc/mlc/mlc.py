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
这段代码定义了一个名为 reproject_ly_boundaries 的函数。它将一个布局（layout）列表 list_ly 重新投影到参考布局 ref_ly 的位置，并返回该布局的边界在 shape 形状的图像上的像素位置。
'''
def reproject_ly_boundaries(list_ly, ref_ly, shape=(512, 1024)):
    print('确定一下是不是这里的mlc.py')
    """
    Reproject a list of layouts into the passed reference using comprehension lists
    """
    '''
    这段代码的目的是计算参考布局（ref_ly）相对于世界坐标系的变换矩阵的逆矩阵，以便将其他布局（ly）投影到参考布局的位置。
    先来看一下 ref_ly.pose.SE3_scaled()，这个函数的返回值是参考布局相对于世界坐标系的变换矩阵。
    具体地说，它返回一个 SE3 对象，表示参考布局从世界坐标系的原点（或其他参考点）开始的平移和旋转变换。
    SE3 对象是一个 4x4 的矩阵，可以用来表示三维坐标系中的变换，其中前三行是旋转矩阵，最后一列是平移向量。
    然后，np.linalg.inv 函数是 numpy 库中的一个函数，用于计算一个矩阵的逆矩阵。
    所以，np.linalg.inv(ref_ly.pose.SE3_scaled()) 的作用就是求出参考布局相对于世界坐标系的变换矩阵的逆矩阵。这个逆矩阵将在后面的代码中用于将其他布局投影到参考布局的位置上。
    '''
    pose_W2ref = np.linalg.inv(ref_ly.pose.SE3_scaled())#计算参考布局相对于世界坐标系的变换矩阵的逆矩阵。

    boundaries_xyz = []#创建一个空列表，准备存储结果。
    #使用列表推导式，对 list_ly 中的每个布局 ly，将其底部和顶部边界的坐标点转换为参考布局坐标系中的点，
    # 然后将它们添加到 boundaries_xyz 列表中。注意到这行代码实际上执行的是一个循环，但是使用了列表推导式的语法。
    '''
    [(..., ...) for ly in list_ly]：这是一个列表推导式的语法，表示要对 list_ly 中的每个元素 ly 进行操作。其中省略号 <...> 表示对 ly 进行的操作。
    pose_W2ref[:3, :]：这个表达式是一个 3x4 的矩阵，表示参考布局中的点在世界坐标系中的坐标值。pose_W2ref 是在前面计算的参考布局相对于世界坐标系的变换矩阵的逆矩阵，
    pose_W2ref[:3, :] 取逆矩阵的前三行，表示旋转矩阵和平移向量，构成转换矩阵从而能够将布局坐标投影到参考布局中的位置上。
    ly.boundary_floor 和 ly.boundary_ceiling 分别表示布局底部和顶部的边界点集合，是一个 N x 3 的矩阵，其中 N 表示点的数量，每个点用三个坐标值表示。
    extend_array_to_homogeneous 函数将维数为 d 的数组扩充到 d+1 维，即增加一维并将值设为 1，用于将点集表示为齐次坐标，从而使矩阵乘法能够将点集变换到正确的位置上。
    @ 运算符表示矩阵乘法，将参考布局坐标系中的点转换为世界坐标系中的点。
    将转换后的坐标点分别添加到 boundaries_xyz 列表中。由于有两个边界点（底部和顶部），因此用了两个 append 方法。
    最终，boundaries_xyz 列表包含了所有布局的边界点的世界坐标系中的值，可以将这些点投影到二维平面上得到像素坐标。
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

    #将 boundaries_xyz 列表中的所有点合并成一个数组，然后使用 xyz2uv 函数将这些点从三维空间投影到二维图像平面上。得到的 uv 是一个二维数组，表示在 shape 形状的图像上的像素位置。
    '''
    np.hstack(boundaries_xyz)：这个代码先将之前处理得到的所有边界点的世界坐标系中的值组合成一个大数组。
    boundaries_xyz 是一个列表，其中每个元素都是一个矩阵，表示一个布局中的边界点集合。np.hstack 函数将列表中的所有矩阵按列连接（即沿着第二个维度组合），形成一个一维数组。
    xyz2uv 函数接受两个参数：组成所有点的一维数组，和图像的大小 shape（一个二元组，表示行数和列数）。它会将所有点从三维坐标系中投影到二维图像平面上，并返回投影后的像素坐标。注意，在这里没有提供 xyz2uv 函数的实现，因此不清楚具体的实现方式。
    uv = xyz2uv(np.hstack(boundaries_xyz), shape)：将组成所有点的一维数组和图像大小传递给 xyz2uv 函数进行投影，将得到的结果存储在一个名为 uv 的变量中。uv 是一个二维数组，表示所有边界点在二维图像平面上的像素坐标位置。
    '''
    uv = xyz2uv(np.hstack(boundaries_xyz), shape)
    print('在mlc里的这个uv shape',uv.shape)
    print('在mlc里的这个uv',uv)

    #uv2depth拿到每个帧对齐的depth,优化这个depth

    return uv

#函数compute_pseudo_labels接受三个参数：list_frames是一个由帧组成的列表，ref_frame是被参考的帧，shape是指定形状的元组，用于定义输出的形状。
# 该函数基于列表中的帧和参考帧ref_frame计算伪标签
def compute_pseudo_labels(list_frames, ref_frame, shape=(512, 1024)):
    """
    Computes the pseudo labels based on the passed list of frames wrt the ref_frame
    """
    cfg = ref_frame.cfg
    # 调用reproject_ly_boundaries()函数来将list_frames中的边界点投影到参考帧ref_frame的图像坐标系中，并保存在变量uv中。
    uv = reproject_ly_boundaries(list_frames, ref_frame, shape)#这个或许是我要找的0626

    # 然后将uv中的所有点坐标限制在标签形状范围内
    uv[0] = np.clip(uv[0], 0, shape[1] - 1)#np.clip(uv[0], 0, shape[1] - 1)：将uv数组的第一个维度（即uv[0]）中的值限制在0到shape[1] - 1之间。这个操作确保了uv[0]中的值不会超过给定的边界范围。
    uv[1] = np.clip(uv[1], 0, shape[0] - 1)#np.clip(uv[1], 0, shape[0] - 1)：对uv数组的第二个维度（即uv[1]）中的值限制在0到shape[0] - 1之间。同样，这个操作确保了uv[1]中的值不会超过给定的边界范围。

    #接着，将uv中相同坐标的点按照出现次数统计到一个直方图中，并返回该直方图及其坐标信息。
    '''
    这段代码实现了以下功能：
    首先，使用np.unique函数将数组uv中的唯一行向量及其出现次数分别存储在uv和counts中。
    接下来，创建一个形状为shape的全零数组prj_map，作为最终的投影映射。
    使用uv中的行和列索引，将counts中的值赋给prj_map对应位置上的元素。这样，prj_map中的非零元素表示了原始映射中每个像素值在投影平面上的出现次数。
    最后，通过copy函数创建_prj_map，它是prj_map的副本。
    这段代码主要用于生成投影映射，其中每个像素值的出现次数表示了在原始映射中对应像素值的像素点数目
    '''
    # ! Aggregating projected points into a histogram
    uv, counts = np.unique(list(uv.T), return_counts=True, axis=0)

    prj_map = np.zeros(shape)
    prj_map[uv[:, 1], uv[:, 0]] = counts

    _prj_map = prj_map.copy()

    '''
    之后，将直方图中的数据根据图像的底部和顶部进行分割，分别计算出底部和顶部的标签。
    对于底部标签，首先将直方图下半部分提取出来，然后在该部分中找到最大值的位置，该位置表示底部标签的垂直位置，同时计算该位置的方差。
    对于顶部标签，同样首先将直方图上半部分提取出来，然后找到该部分最大值的位置，计算该位置的方差。最后将所有计算所得的标签位置和方差信息以及直方图返回。
    '''

    '''
    这段代码是一个函数，执行以下操作：
    定义变量floor_map和ceiling_map，分别表示地板和天花板的映射。
    使用np.argmax函数在floor_map和ceiling_map的每一列中找到最大值的索引，表示地板和天花板的垂直位置。
    调用get_std函数计算地板和天花板映射的标准差。
    构建平均分布的水平坐标u。
    返回地板和天花板的垂直位置，地板和天花板的标准差以及原始映射_prj_map。  
    '''
    # ! Floor
    floor_map = _prj_map[512//2:, :]#代码从_prj_map中选取了一部分作为floor_map。具体来说，floor_map是从_prj_map数组的第512//2行（即从地板位置开始）开始到最后一行的所有行，即表示地板的部分
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
