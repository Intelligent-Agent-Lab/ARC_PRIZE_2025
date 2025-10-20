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
from env.meta_preprocess import ActiveShapeColorOntHot, \
                            generate_meta_dataset, \
                            load_challenges_and_solutions
from env.utils import convert_int_to_dict, \
                                    convert_dict_to_int, \
                                        vectorized_convert_dict_to_int,\
                                            vectorized_convert_int_to_dict, \
                                                randomize_part_of_solution
                                            


class MetaArcAgiGridEnvCoord(gym.Env):
    """ Task 및 Pair Index를 reset 함수에서 지정가능한 버전
    """
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
    
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=(30,120), dtype=int)

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
                    randomize_part_of_solution(self._target_grid_img, self._current_grid_img, self.active_shape, 
                                                empty_val=empty_val, fill_ratio_range=(0.2, 0.7))
        else:
            # Fill only the active_shape area with empty_val (11)
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

class SelectedMetaArcAgiGridEnv(gym.Env):
    """ Task 및 Pair Index를 reset __init__에서 지정하는 버전
    """
    def __init__(self,
            meta_dataset: Dict[str, Any],
            mode: str, # (meta_train, meta_eval, meta_test)
            phase: str, # (train, test)
            task_id: str, 
            pair_idx: int,
            rand_init: bool,
            test_active_shape_color: ActiveShapeColorOntHot=None,
            ) -> None:
        
        # observation space에 대한 정의 (XYXY 30x120)
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=(30,120), dtype=int)

        # action space에 대한 정의 (0~9 색상, 30x30 좌표의 product space)
        self.action_space = gym.spaces.Discrete(9000)

        # task_id에 해당하는 target grid 선택 (test input이 여러 개 존재 가능하므로 한 번 더 random.choice 수행
        self.mode = mode
        self.phase = phase
        self.task_id = task_id
        self.pair_idx = pair_idx
        self.rand_init = rand_init
        
        if self.mode != 'meta_test':
            self._target_grid_img = meta_dataset[self.task_id][f'{self.phase}_data'][self.pair_idx]
            self.active_shape_color = meta_dataset[self.task_id][f'{self.phase}_info'][self.pair_idx]
            self.active_shape = self.active_shape_color.shape
            self.active_color = self.active_shape_color.color
        else:
            # prediction (meta-test) 의 경우 규칙 집합으로부터 예측
            self._target_grid_img = None
            self.active_shape_color = test_active_shape_color
            self.active_shape = self.active_shape_color.shape
            self.active_color = self.active_shape_color.color
        
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
                    randomize_part_of_solution(self._target_grid_img, self._current_grid_img, self.active_shape, 
                                                empty_val=empty_val, fill_ratio_range=(0.2, 0.7))
        else:
            # Fill only the active_shape area with empty_val (11)
            self._current_grid_img[:, 90:][self.active_shape == 1] = empty_val
    
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
              options: Optional[dict] = None,):
        random.seed(seed)
        np.random.seed(seed)
        self.timestep = 0
        self.episode_returns = 0
        self.episode_lengths = 0
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


def make_env(meta_dataset: Dict[str, Any],
            mode: str, # (meta_train, meta_eval, meta_test)
            phase: str, # (train, test)
            task_id: str, 
            pair_idx: int,
            rand_init,
            test_active_shape_color: ActiveShapeColorOntHot=None,
             ):
    """Create and wrap environment for vectorized training."""
    def thunk():
        env = SelectedMetaArcAgiGridEnv(meta_dataset,
                                    mode, # (meta_train, meta_eval, meta_test)
                                    phase, # (adpatation, evaluation)
                                    task_id, 
                                    pair_idx,
                                    rand_init,
                                    test_active_shape_color,)
        return env
    return thunk


def make_meta_task_env(meta_train_dataset,
                meta_eval_dataset,
                meta_test_dataset,
                mode: str, # (meta_train, meta_eval, meta_test)
                phase: str, # (train, test)
                task_id: str, 
                rand_init: bool,
                list_meta_test_shape_color_infos: List[ActiveShapeColorOntHot]=None):
    if mode == 'meta_train':
        meta_dataset = meta_train_dataset
    elif mode == 'meta_eval' or 'meta_evaluation':
        meta_dataset = meta_eval_dataset
    elif mode == 'meta_test':
        meta_dataset = meta_test_dataset
    else:
        raise NotImplementedError
    pair_indices = range(len(meta_dataset[task_id][f'{phase}_data']))
    
    if mode == 'meta_train' or mode == 'meta_evaluation' or mode == 'meta_eval':
        envs = gym.vector.SyncVectorEnv([
                make_env(
                    meta_dataset,
                    mode,
                    phase,
                    task_id,
                    pair_idx,
                    rand_init,
                    None,
                ) for pair_idx in pair_indices
            ])
    elif mode == 'meta_test':
        envs = gym.vector.SyncVectorEnv([
            make_env(
                meta_dataset,
                mode,
                phase,
                task_id,
                pair_idx,
                rand_init,
                list_meta_test_shape_color_infos[pair_idx],
            ) for pair_idx in pair_indices
        ])
    else:
        raise ValueError("mode is not (meta_train, meta_evaluation, meta_test)")
    return envs