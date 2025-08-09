from dataclasses import dataclass

from jaxtyping import Bool
import numpy as np
from PIL import Image
from typing import Literal, TypeVar

from ..dataset import Dataset, DatasetCfg
from .dataset_image import DatasetImage, DatasetImageCfg
from .. import register_dataset
from spatialreasoners.type_extensions import ConditioningCfg, Stage
import json 
from itertools import permutations, product
import torch 
from .type_extensions import ImageExample
from pathlib import Path 

@dataclass(frozen=True, kw_only=True)
class DatasetARCAGICfg(DatasetImageCfg):
    # name: Literal["arc_agi"] = "arc_agi"
    root: str
    data_shape: list
    dataset_size: int = 9278
    num_classes: int = 10
    challenge_json_path: str = "/home/kukjin/Projects/ARC_PRIZE_2025/spatialreasoners/datasets/arc_agi/arc-agi_training_challenges.json"
    solution_json_path: str = "/home/kukjin/Projects/ARC_PRIZE_2025/spatialreasoners/datasets/arc_agi/arc-agi_training_solutions.json"

T = TypeVar("T", bound=DatasetARCAGICfg)

def preprocess_data(challenges, solutions):
    max_input_shape = (30, 30)
    max_output_shape = (30, 30)
    dict_train_XY_pairs = dict()
    dict_test_XY_pairs = dict()
    # Iterate over each task (ID) in the challenge dataset
    dict_XYXYXY_pairs = dict()
    list_XYXYXY_pairs = []
    for (task_id, task_data), (_, task_sol) in zip(challenges.items(), solutions.items()):
        # Extract training pairs
        dict_train_XY_pairs[task_id] = []
        dict_test_XY_pairs[task_id] = []
        dict_XYXYXY_pairs[task_id] = []
        for pair in task_data.get('train', []):
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            # Append input and output grids to the respective lists
            padded_input = np.pad(input_grid, [(0, max_input_shape[0] - input_grid.shape[0]), (0, max_input_shape[1] - input_grid.shape[1])], mode='constant', constant_values=0)
            padded_output = np.pad(output_grid, [(0, max_output_shape[0] - output_grid.shape[0]), (0, max_output_shape[1] - output_grid.shape[1])], mode='constant', constant_values=0)
            
            XY = np.concatenate([padded_input, padded_output], axis=-1)
            dict_train_XY_pairs[task_id].append(XY)
        for test_pair, test_sol in zip(task_data.get('test', []), task_sol):
            input_grid = np.array(test_pair['input'])
            output_grid = np.array(test_sol)
            padded_input = np.pad(input_grid, [(0, max_input_shape[0] - input_grid.shape[0]), (0, max_input_shape[1] - input_grid.shape[1])], mode='constant', constant_values=0)
            padded_output = np.pad(output_grid, [(0, max_output_shape[0] - output_grid.shape[0]), (0, max_output_shape[1] - output_grid.shape[1])], mode='constant', constant_values=0)
            XY = np.concatenate([padded_input, padded_output], axis=-1)
            dict_test_XY_pairs[task_id].append(XY)

    for task_id in dict_XYXYXY_pairs.keys():
        train_XY_pair_list = dict_train_XY_pairs[task_id]
        test_XY_pair_list = dict_test_XY_pairs[task_id]
        train_XYXY_pair_list = []
        for p in permutations(train_XY_pair_list, 2):
            # Concatenate the pair of matrices horizontally
            # p[0] is the first matrix (30x60) and p[1] is the second (30x60)
            # The result, xyxy_pair, will be a 30x120 matrix
            xyxy_pair = np.hstack((p[0], p[1]))
            train_XYXY_pair_list.append(xyxy_pair)
        train_test_XYXYXY_pair_list = []
        for train_pair, test_pair in product(train_XYXY_pair_list, test_XY_pair_list):
        # train_pair(30x120)와 test_pair(30x60)를 수평으로 연결 (axis=1)
            XYXYXY_pair = np.hstack((train_pair, test_pair))
            # 생성된 30x180 행렬을 최종 리스트에 추가
            train_test_XYXYXY_pair_list.append(XYXYXY_pair)
        dict_XYXYXY_pairs[task_id] = train_test_XYXYXY_pair_list
        list_XYXYXY_pairs.extend(train_test_XYXYXY_pair_list)
    padded_XYXYXY_ary = np.concat([list_XYXYXY_pairs])
    padded_XYXYXY_ary = (padded_XYXYXY_ary.astype(np.float32) / 4.5) - 1
    return padded_XYXYXY_ary


@register_dataset("arc_agi", DatasetARCAGICfg)
class DatasetARCAGI(Dataset[T]):
    def __init__(
        self,
        cfg: DatasetARCAGICfg,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
        ):
        super().__init__(cfg, conditioning_cfg, stage)
        self.num_classes = 10
        print("start to load json")
        
        with open(Path(cfg.challenge_json_path), 'r', encoding='utf-8') as file:
            challenges = json.load(file)
        with open(Path(cfg.solution_json_path), 'r', encoding='utf-8') as file:
            solutions = json.load(file)
        print("start to preprocess")
        self.XYXYXY = torch.tensor(preprocess_data(challenges, solutions))
        self.XYXYXY = self.XYXYXY.unsqueeze(1)
        print(f"self.XYXYXY.shape: {self.XYXYXY.shape}")
        print("preprocess is ended")

    def __getitem__(self, idx: int):
        return {"image": self.XYXYXY[idx]}

    def load(self, idx: int):
        print(self.XYXYXY[idx].shape)
        return {"image": self.XYXYXY[idx]}

    @property
    def _num_available(self) -> int:
        return self.cfg.dataset_size
