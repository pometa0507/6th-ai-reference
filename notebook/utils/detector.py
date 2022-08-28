import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from google.protobuf import text_format

sys.path.append('../second.pytorch/')
from second.protos import pipeline_pb2
from second.pytorch.train import build_network
from second.utils import config_tool

class Second3DDector(object):

    def __init__(self, config_p, model_p, calib_data=None, num_points_feature=4, device="cpu"):
        self.config_p = config_p
        self.model_p = model_p
        self.calib_data = calib_data
        self.num_points_feature = num_points_feature
        self.device = device
        self._init_model()

    def _init_model(self):
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_p, 'r') as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        
        self.input_cfg = self.config.eval_input_reader
        self.model_cfg = self.config.model.second
        config_tool.change_detection_range_v2(self.model_cfg, [-50, -50, 50, 50])
        
        # logging.info('config loaded.')

        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(self.model_p))
        
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator
        # logging.info('network done, voxel done.')

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2]//config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = self.target_assigner.generate_anchors(feature_map_size)['anchors']
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)
        # logging.info('anchors generated.')

    def load_pc_from_file(self, pc_f):
        # logging.info('loading pc from: {}'.format(pc_f))
        # return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])
        if type(pc_f) is str:
            points = np.fromfile(pc_f, dtype=np.float32, count=-1)
            points = points.reshape([-1, self.num_points_feature])
        elif type(pc_f) is np.ndarray:
            points = pc_f
        return points

    def load_an_in_example_from_points(self, points):
        res = self.voxel_generator.generate(points, max_voxels=90000)
        voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
        coords = np.pad(coords, ((0,0), (1,0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coords,
        }

    def predict_on_points(self, points_path):

        points = self.load_pc_from_file(points_path)

        example = self.load_an_in_example_from_points(points)
        pred = self.net(example)[0]
        boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        label_preds = pred["label_preds"].detach().cpu().numpy()

        return {
            'boxes_lidar': boxes_lidar,
            'scores': scores,
            'label_preds': label_preds,
        }