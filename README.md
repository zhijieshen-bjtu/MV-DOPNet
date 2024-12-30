Pytorch implementation of ["360 Layout Estimation via Orthogonal Planes Disentanglement and Multi-View Geometric Consistency Perception"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10634822)) (TPAMI'24)  
The project is reproducted based on [PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer), [DOPNet](https://github.com/zhijieshen-bjtu/DOPNet) and [mvl_tookit](https://github.com/mvlchallenge/mvl_toolkit).  

**Prepare data**  
Please prepare the dataset by following the guidelines in [mvl_toolkit](https://github.com/mvlchallenge/mvl_toolkit).  If you have correctly prepared the dataset, the data can be found in the following file:

```

../mvl_challenge/assets/zips/ 

```
where includes 5 files: challenge_phase__training_set.zip、challenge_phase__testing_set.zip、pilot_set.zip、warm_up_training_set.zip, and warm_up_testing_set.zip 

```

../mvl_challenge/assets/mvl_data/  
```

where includes 3 files: img、labels, and geometry_info 

If you finish all the work, you can go next!

**Generate pseudo labels**  
In this step, we employ DOPNet to generate pseudo labels. You may need to modify the following file to choose the model, dataset splits, etc.

```
/tutorial/create_mlc_labels.py
/tutorial/create_mlc_labels.yaml
```
The pre-trained model weights should be placed correctly in the following folder：
```

```
