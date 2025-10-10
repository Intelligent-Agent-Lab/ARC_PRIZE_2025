from typing import Optional
import json
import random
import numpy as np
import torch 
import jax
import chex
from jax import jit
import jax.numpy as jnp 
from flax import struct
from jaxtyping import Array, Float, Int
from typing import Optional, Dict, Tuple, List, Any, Union
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import Wrapper
from gymnax.environments import environment
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from meta_preprocess import ActiveShapeColorOntHot, \
                            generate_meta_dataset, \
                            load_challenges_and_solutions


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
        
    def _select_task(self, mode: str, 
                     seed: int) -> str:
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
            'timestep': self.timestep,
            'mode': self.mode,
            'phase': self.phase,
            'task_id': self.task_id,
            'pair_idx': self.pair_idx,
            'rand_init': self.rand_init,
            "current_grid_img": self._current_grid_img,
            'target_grid_img': self._target_grid_img,
            "active_shape": self.active_shape,
            "active_color": self.active_color,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
        }
    
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        if options != None:
            self.mode = options['mode'] # meta_train, meta_eval, meta_test
            self.phase = options['phase'] # train(inner adaptation) or inner test(inner evaluation)
            self.task_id = options['task_id']
            self.pair_idx = options['pair_idx']
    
        self.timestep = 0
        if self.task_id == None:
            self.task_id = self._select_task(self.mode, seed)
            
        self.episode_returns = 0
        self.episode_lengths = 0
        """
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
            mode: (train, evaluation, test)
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        # super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        # task_id에 해당하는 target grid 선택 (test input이 여러 개 존재 가능하므로 한 번 더 random.choice 수행
        if self.mode == 'meta_train':
            dataset = self.meta_train_dataset
        elif self.mode == 'meta_eval' or 'meta_evaluation':
            dataset = self.meta_eval_dataset
        elif self.mode == 'meta_test':
            dataset = self.meta_test_dataset
        else:
            raise NotImplementedError
        
        if self.pair_idx == None:
            self.pair_idx = random.choice(list(range(len(self.meta_dataset[f'_data'][self.task_id]))))
        if self.mode != 'meta_test':
            self._target_grid_img = dataset[self.task_id][f'{self.phase}_data'][self.pair_idx]
        else:
            self._target_grid_img = None
        
        self.active_shape_color = dataset[self.task_id][f'{self.phase}_info'][self.pair_idx]
        self.active_shape = self.active_shape_color.shape
        self.active_color = self.active_shape_color.color
        

        # prediction의 경우 규칙 집합으로부터 예측
        #self.size_candidate, self.color_candidate = predict_candidates_from_task_id(self.task_id, self.training_challenges)

        # Get rand_init option
        self.rand_init = options.get('rand_init', False) if options else False
        
        # target grid에서 test solution에 해당하는 부분을 전부 pad_val으로 masking하고 current grid로 할당
        empty_val = 11
        self._current_grid_img = self._target_grid_img.copy()
        # Fill the solution area with different values based on size_candidate
        self._current_grid_img[:, 90:] = 10  # Fill entire solution area with 10 first
        if self.rand_init:
            # 20% chance to start with completely empty grid
            if np.random.random() < 0.2:
                # Fill only the active_shape area with empty_val (11)
                self._current_grid_img[self.active_shape == 1] = empty_val
            else:
                # Randomly initialize some cells in active_shape area with correct answers
                if self._target_grid_img is None:
                    pass
                else:
                    # Extract target values in active region
                    target_solution = self._target_grid_img[:, 90:][self.active_shape == 1]
                    
                    # Randomly fill some percentage of correct answers
                    fill_ratio = np.random.uniform(0.2, 0.7)  # 20%-70% pre-filled
                    total_cells = np.sum(self.active_shape)
                    num_filled = int(total_cells * fill_ratio)
                    
                    # Create mask for which cells to fill
                    fill_mask = np.zeros(total_cells, dtype=bool)
                    filled_indices = np.random.choice(total_cells, num_filled, replace=False)
                    fill_mask[filled_indices] = True
                    
                    # Start with empty values
                    current_solution = np.full(total_cells, empty_val)
                    
                    # Fill selected positions with correct answers
                    current_solution[fill_mask] = target_solution[fill_mask]
                    
                    # Apply to current grid
                    self._current_grid_img[:, 90:][self.active_shape == 1] = current_solution
        else:
            # Fill only the active_shape area with empty_val (11)
            print(self._current_grid_img.shape)
            self._current_grid_img[:, 90:][self.active_shape == 1] = empty_val
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        """Execute one timestep within the environment.
        Args:
            action: The action to take (0-10)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        dict_action = convert_int_to_dict(action)
        color = dict_action['color']
        coordinate = dict_action['coordinate']
        row = coordinate[0]
        col = coordinate[1]
        index = row*30 + col
        
        # 현재 칸의 값 확인 (action을 취하기 전 값)
        current_cell_value = self._current_grid_img[row, 90+col]
        
        self._current_grid_img[row, 90+col] = color
        
        # Log grid changes occasionally
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 1
            
        if self.step_counter % 1000 == 0:  # Log every 1000 steps
            solution_area = self._current_grid_img[:, 90:]
            filled_cells = int(np.sum(solution_area != 11))
            print(f"Step {self.step_counter}: Progress {filled_cells}/16 cells filled")

        target_color_img = self._target_grid_img[row, 90+col]
        
        self.timestep += 1
        
        terminated = False
        truncated = False

        # 잘못된 위치에 칠하거나, 이미 칠해진 곳에 다시 칠한 경우 (실패)
        if color != target_color_img or current_cell_value != 11:
            terminated = True
            reward = -1
        else:
            # 퍼즐을 완성한 경우
            if np.array_equal(self._current_grid_img, self._target_grid_img):
                terminated = True
                reward = 1
            # 올바른 중간 과정인 경우 (완성은 아직 아님)
            else:
                reward = 0.05 if color != 10 else 0.01

        observation = self._get_obs()
        info = self._get_info()
        self.episode_returns += reward
        self.episode_lengths += 1
        return observation, reward, terminated, truncated, info
        

def convert_int_to_dict(action: int) -> dict:
    # 0 ~ 899: color 0
    # 900 ~ 1799: color 1
    # 1800 ~ 2699: color 2
    # 2700 ~ 3599: color 3
    # 3600 ~ 4499: color 4
    # 4500 ~ 5399: color 5
    # 5400 ~ 6299: color 6
    # 6300 ~ 7199: color 7
    # 7200 ~ 8099: color 8
    # 8100 ~ 8999: color 9
    # 9000 ~ 9899: color 10
    assert (action >= 0 and action < 9900)
    color = action // 900
    row = (action - 900*color) // 30
    col = (action - 900*color) % 30
    coordinate = (row, col)
    dict_action = {'color': color, 
                   'coordinate': coordinate}
    return dict_action
    

def vectorized_convert_int_to_dict(actions: torch.Tensor) -> dict:
    """벡터화된 함수 (텐서용)"""
    # 입력 검증
    assert torch.all((actions >= 0) & (actions < 9900)), "All actions must be in range [0, 9900)"
    
    # 벡터화된 계산
    # color = action // 900
    colors = torch.div(actions, 900, rounding_mode='floor')
    
    # remainder = action - 900 * color  
    remainders = actions - 900 * colors
    
    # row = remainder // 30
    rows = torch.div(remainders, 30, rounding_mode='floor')
    
    # col = remainder % 30
    cols = remainders % 30
    
    # 결과를 딕셔너리로 반환
    result = {
        'color': colors.numpy(),
        'coordinate': torch.stack([rows, cols], dim=-1).numpy()  # (N, 2) 형태
    }
    return result


def convert_dict_to_int(dict_action: dict,
                        ) -> int:
    """
    딕셔너리 액션을 정수 액션으로 변환하는 벡터화된 함수
    Args:
        dict_actions: {
            'color': numpy array or torch.Tensor of shape (,),
            'coordinate': numpy array or torch.Tensor of shape (2,)
        }
    Returns:
        np.ndarray: 정수 액션들 (N,)
    """
    color = dict_action['color']
    coordinate = dict_action['coordinate']
    int_action = color * 900 + coordinate[0]*30 + coordinate[1]
    return int_action


def vectorized_convert_dict_to_int(dict_actions: dict) -> np.ndarray:
    """
    딕셔너리 액션들을 정수 액션들로 변환하는 벡터화된 함수
    Args:
        dict_actions: {
            'color': numpy array or torch.Tensor of shape (N,),
            'coordinate': numpy array or torch.Tensor of shape (N, 2)
        }
    Returns:
        np.ndarray: 정수 액션들 (N,)
    """
    colors = dict_actions['color']
    coordinates = dict_actions['coordinate']
    
    # 벡터화된 계산: int_action = color * 900 + row * 30 + col
    # coordinates는 (N, 2) 형태이므로 coordinates[:, 0]이 row, coordinates[:, 1]이 col
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]
    int_actions = colors * 900 + rows * 30 + cols
    return int_actions
