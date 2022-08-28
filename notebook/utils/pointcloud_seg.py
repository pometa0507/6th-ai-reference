######################
# Based on https://github.com/Song-Jingyu/PointPainting
# Licensed under The MIT License
# Authors Chen Gao, Jingyu Song, Youngsun Wi, Zeyu Wang
######################

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .calibration_kitti import Calibration, get_calib_from_file


def get_segmentation_score(img_path, model, device):
    """
    画像に対するセグメンテーションのスコアを出力する for KITTI

    Returns
    -------
    output_reassign_softmax : modelの出力(torch.tensor)
        a tensor H  * W * 4, for each pixel we have 4 scorer that sums to 1
    """

    input_image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # バッチ軸追加

    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_permute = output.permute(1, 2, 0)
    output_probability, output_predictions = output_permute.max(2)

    other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | \
                          (output_predictions == 7) | (output_predictions == 15))
    detect_object_mask = ~other_object_mask
    sf = torch.nn.Softmax(dim=2)

    # bicycle = 2 car = 7 person = 15 background = 0
    output_reassign = torch.zeros(
        output_permute.size(0), output_permute.size(1), 4)
    output_reassign[:, :, 0] = detect_object_mask * output_permute[:, :, 0] + \
                               other_object_mask * output_probability  # background
    output_reassign[:, :, 1] = output_permute[:, :, 2]  # bicycle
    output_reassign[:, :, 2] = output_permute[:, :, 7]  # car
    output_reassign[:, :, 3] = output_permute[:, :, 15]  # person
    output_reassign_softmax = sf(output_reassign).cpu().numpy()

    return output_reassign_softmax

def get_calib_fromfile(calib_file):
    calib = get_calib_from_file(calib_file)
    calib['P2'] = np.concatenate([calib['P2'],
                                  np.array([[0., 0., 0., 1.]])], axis=0)
    calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
    calib['R0_rect'][3, 3] = 1.
    calib['R0_rect'][:3, :3] = calib['R0']
    calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'],
                                           np.array([[0., 0., 0., 1.]])], axis=0)
    return calib

def create_cyclist(augmented_lidar): # for KITTI function
    bike_idx = np.where(augmented_lidar[:,5]>=0.2)[0] # 0, 1(bike), 2, 3(person)
    bike_points = augmented_lidar[bike_idx]
    cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
    for i in range(bike_idx.shape[0]):
        cyclist_mask = (np.linalg.norm(augmented_lidar[:,:3]-bike_points[i,:3], axis=1) < 1) & (np.argmax(augmented_lidar[:,-4:],axis=1) == 3)
        if np.sum(cyclist_mask) > 0:
            cyclist_mask_total |= cyclist_mask
        else:
            augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[bike_idx[i], 5], 0
    augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[cyclist_mask_total, 7]
    return augmented_lidar
    

def augment_lidar_class_scores(class_scores, lidar_raw, projection_mats):
    """
    Projects lidar points onto segmentation map, appends class score each point projects onto.
    """
    class_num = class_scores.shape[2]
    
    lidar_cam_coords = cam_to_lidar(lidar_raw, projection_mats)

    # Projection
    lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
    points_projected_on_mask = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
    points_projected_on_mask = points_projected_on_mask.transpose()
    points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

    true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores.shape[1]) #x in img coords is cols of img
    true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores.shape[0])
    true_where_point_on_img = true_where_x_on_img & true_where_y_on_img

    points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
    points_projected_on_mask = np.floor(points_projected_on_mask).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
    points_projected_on_mask = points_projected_on_mask[:, :2] # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

    #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
    point_scores = class_scores[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1, class_num)

    augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_num))), axis=1)

    augmented_lidar[true_where_point_on_img, -class_num:] += point_scores
    augmented_lidar = augmented_lidar[true_where_point_on_img]
    augmented_lidar = create_cyclist(augmented_lidar)  # personとbikeクラスからcyclistを作成

    augmented_lidar = augmented_lidar.astype(np.float32)
    augmented_lidar = augmented_lidar[augmented_lidar[:, 0] > 0, :]    # x>0 でマスク

    return augmented_lidar

def cam_to_lidar(pointcloud, projection_mats):
    """
    Takes in lidar in velo coords, returns lidar points in camera coords

    :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
    :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
    """

    lidar_velo_coords = copy.deepcopy(pointcloud)
    reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
    lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
    lidar_cam_coords = projection_mats['Tr_velo2cam'] @ lidar_velo_coords.T

    lidar_cam_coords = lidar_cam_coords.T
    lidar_cam_coords[:, -1] = reflectances

    return lidar_cam_coords

def overlap_seg(img,
                class_scores,
                palette=None,
                opacity=0.5):
    """
    Draw `class_scores` over `img`
    """

    if palette is None:
        palette = [[255, 255, 255], # background
                   [202, 105, 157], # bicycle
                   [253, 141, 60],  # car
                   [0, 0, 255],     # person
                  ]

    result = np.argmax(class_scores, 2) 
    seg = result
    
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(class_scores.shape[2], 3))

    palette = np.array(palette)
    assert 0 < opacity <= 1.0

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    
    for label, color in enumerate(palette):      
        color_seg[seg == label, :] = color
    
    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    return img
    