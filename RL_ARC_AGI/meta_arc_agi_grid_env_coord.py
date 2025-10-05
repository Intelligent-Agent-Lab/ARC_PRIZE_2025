from typing import Optional
import json
import random
import numpy as np
import torch 
import jax
import jax.numpy as jnp 
from jaxtyping import Array, Float, Int
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import Wrapper
from typing import Tuple, Dict, Union, List, Any
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from meta_preprocess import ActiveShapeColorOntHot, \
                            generate_meta_dataset, \
                            load_challenges_and_solutions, \

class MetaArcAgiGridEnvCoord(gym.Env):
    def __init__(self,
            meta_train_dataset: Dict[str, Any],
            meta_eval_dataset: Dict[str, Any],
            meta_test_dataset: Dict[str, Any],
            ) -> None:
        self.meta_train_dataset = meta_train_dataset
        self.meta_eval_dataset = meta_eval_dataset
        self.meta_test_dataset = meta_test_dataset
        self.meta_train_task_list = list(self.meta_train_dataset.keys())
        self.meta_eval_task_list = list(self.meta_eval_dataset.keys())
        self.meta_test_task_list = list(self.meta_test_dataset.keys())
    
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=(30,180), dtype=int)

        # action space에 대한 정의 (0~9 색상, 30x30 좌표의 product space)
        self.action_space = gym.spaces.Discrete(9000)
        
    def _select_task(self, mode: str) -> str:
        random.seed(seed)
        np.random.seed(seed)
        if mode == 'meta_train':
            task_id = random.choice(self.meta_train_task_list)
        elif mode == 'meta_eval':
            task_id = random.choice(self.meta_eval_task_list)
        elif mode == 'meta_test':
            task_id = random.choice(self.meta_test_task_list)
        else:
            raise NotImplementedError
        return task_id
    
    def _get_obs(self) -> Dict:
        return self._current_grid_img.copy()
    
    def _get_info(self) -> Dict:
        return {
            'target_grid_img': self._target_grid_img,
            'timestep': self.timestep,
            'task_id': self.task_id,
            'test_input_idx': self.test_input_idx,
            "current_grid_img": self._current_grid_img,
            "chosen_grid_img": self._chosen_grid_img,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "size_candidate": self.size_candidate,
            "color_candidate": self.color_candidate,
        }
        
    
