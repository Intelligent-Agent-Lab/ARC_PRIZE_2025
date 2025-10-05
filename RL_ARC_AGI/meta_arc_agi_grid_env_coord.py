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
            self.task_id = self._select_task(seed)
            
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



# @struct.dataclass
# class EnvState(environment.EnvState):
#     current_grid_img: Int[Array, '30 30']
#     target_grid_img: Int[Array, '30 30']
#     time: int
#     task_id: str
#     pair_idx: int
#     episode_returns: float
#     episode_lengths: int
#     is_success: bool
#     max_sum_reward: float


# @struct.dataclass
# class EnvParams(environment.EnvParams):
#     max_steps_in_episode: int = 1000
#     mode: str = 'train'
#     task_id: Optional[str] = None
#     pair_idx: Optional[int] = None
#     rand_init: bool = False
#     ratio_fill_correct: float = 0.0


# def preprocess_data_jax(challenges: Dict[str, Any], solutions: Dict[str, Any]) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, List], Dict[str, List]]:
#     """
#     JAX-optimized preprocessing function for ARC AGI 2 dataset.

#     Args:
#         challenges: Dictionary containing challenge data
#         solutions: Dictionary containing solution data

#     Returns:
#         Tuple of (dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs, dict_XYXYXY_img_shape_colors, dict_XYXYXY_seq_shape_colors)
#     """
#     MAX_SHAPE = (30, 30)
#     PAD_VAL = 10

#     dict_XYXYXY_img_pairs = {}
#     dict_XYXYXY_seq_pairs = {}
#     dict_XYXYXY_img_shape_colors = {}
#     dict_XYXYXY_seq_shape_colors = {}

#     # Process each task
#     for task_id, task_data in challenges.items():
#         task_sol = solutions[task_id]

#         # Process training pairs
#         train_pairs_img, train_pairs_seq = _process_pairs_jax(
#             task_data.get('train', []),
#             MAX_SHAPE,
#             PAD_VAL
#         )

#         # Process test pairs
#         test_inputs = task_data.get('test', [])
#         test_pairs_img, test_pairs_seq, shape_color_infos = _process_test_pairs_jax(
#             test_inputs,
#             task_sol,
#             MAX_SHAPE,
#             PAD_VAL
#         )

#         # Generate XYXYXY pairs efficiently with shape color info
#         img_pairs, img_shape_colors = _generate_xyxyxy_pairs_jax(
#             train_pairs_img, test_pairs_img, shape_color_infos, is_sequence=False
#         )

#         seq_pairs, seq_shape_colors = _generate_xyxyxy_pairs_jax(
#             train_pairs_seq, test_pairs_seq, shape_color_infos, is_sequence=True
#         )

#         dict_XYXYXY_img_pairs[task_id] = img_pairs
#         dict_XYXYXY_seq_pairs[task_id] = seq_pairs
#         dict_XYXYXY_img_shape_colors[task_id] = img_shape_colors
#         dict_XYXYXY_seq_shape_colors[task_id] = seq_shape_colors

#     return dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs, dict_XYXYXY_img_shape_colors, dict_XYXYXY_seq_shape_colors


# def _pad_grid_jax(grid: chex.Array, target_shape: Tuple[int, int], pad_val: int) -> chex.Array:
#     """Efficiently pad a grid to target shape using JAX."""
#     pad_height = jnp.maximum(0, target_shape[0] - grid.shape[0])
#     pad_width = jnp.maximum(0, target_shape[1] - grid.shape[1])

#     # JAX requires fixed padding specification
#     pad_config = [(0, pad_height), (0, pad_width)]
#     return jnp.pad(grid, pad_config, mode='constant', constant_values=pad_val)


# def _process_pairs_jax(pairs: List[Dict], max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[chex.Array], List[chex.Array]]:
#     """Process input-output pairs into both image and sequence formats using JAX."""
#     img_pairs = []
#     seq_pairs = []

#     for pair in pairs:
#         input_grid = jnp.array(pair['input'])
#         output_grid = jnp.array(pair['output'])

#         # Pad grids
#         padded_input = _pad_grid_jax(input_grid, max_shape, pad_val)
#         padded_output = _pad_grid_jax(output_grid, max_shape, pad_val)

#         # Image format: concatenate along width (axis=1)
#         xy_img = jnp.concatenate([padded_input, padded_output], axis=1)
#         img_pairs.append(xy_img)

#         # Sequence format: flatten and concatenate
#         seq_input = padded_input.flatten()
#         seq_output = padded_output.flatten()
#         xy_seq = jnp.concatenate([seq_input, seq_output])
#         seq_pairs.append(xy_seq)

#     return img_pairs, seq_pairs


# def _process_test_pairs_jax(test_inputs: List[Dict], solutions: List, max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[chex.Array], List[chex.Array], List[ActiveShapeColor]]:
#     """Process test pairs with their solutions using JAX."""
#     img_pairs = []
#     seq_pairs = []
#     shape_color_infos = []

#     for test_input, solution in zip(test_inputs, solutions):
#         input_grid = jnp.array(test_input['input'])
#         output_grid = jnp.array(solution)

#         # Pad grids
#         padded_input = _pad_grid_jax(input_grid, max_shape, pad_val)
#         padded_output = _pad_grid_jax(output_grid, max_shape, pad_val)

#         # Image format
#         xy_img = jnp.concatenate([padded_input, padded_output], axis=1)
#         img_pairs.append(xy_img)

#         shape_color = ActiveShapeColor(
#             row=output_grid.shape[0],
#             col=output_grid.shape[1],
#             color=jnp.unique(output_grid)
#         )
#         shape_color_infos.append(shape_color)

#         # Sequence format
#         seq_input = padded_input.flatten()
#         seq_output = padded_output.flatten()
#         xy_seq = jnp.concatenate([seq_input, seq_output])
#         seq_pairs.append(xy_seq)

#     return img_pairs, seq_pairs, shape_color_infos


# def _generate_xyxyxy_pairs_jax(train_pairs: List[chex.Array], test_pairs: List[chex.Array], shape_color_infos: List[ActiveShapeColor], is_sequence: bool) -> Tuple[List[chex.Array], List[ActiveShapeColor]]:
#     """
#     Generate XYXYXY pairs from training and test data with corresponding shape color info using JAX.
#     """
#     if not train_pairs or not test_pairs:
#         return [], []

#     # Generate all training pair permutations (XYXY format)
#     train_xyxy_pairs = []

#     # Generate permutations manually for JAX compatibility
#     for i in range(len(train_pairs)):
#         for j in range(len(train_pairs)):
#             if i != j:
#                 if is_sequence:
#                     xyxy_pair = jnp.concatenate([train_pairs[i], train_pairs[j]])
#                 else:
#                     xyxy_pair = jnp.hstack([train_pairs[i], train_pairs[j]])
#                 train_xyxy_pairs.append(xyxy_pair)

#     # Generate XYXYXY combinations with corresponding shape color info
#     xyxyxy_pairs = []
#     corresponding_shape_colors = []

#     for train_pair in train_xyxy_pairs:
#         for test_idx, test_pair in enumerate(test_pairs):
#             # Create XYXYXY pair
#             if is_sequence:
#                 xyxyxy_pair = jnp.concatenate([train_pair, test_pair])
#             else:
#                 xyxyxy_pair = jnp.hstack([train_pair, test_pair])

#             xyxyxy_pairs.append(xyxyxy_pair)

#             # Add corresponding shape color info
#             corresponding_shape_colors.append(shape_color_infos[test_idx])

#     return xyxyxy_pairs, corresponding_shape_colors


# def preprocess_data_generator_jax(challenges: Dict[str, Any], solutions: Dict[str, Any]):
#     """
#     Generator version that yields one task at a time to reduce memory usage using JAX.
#     """
#     MAX_SHAPE = (30, 30)
#     PAD_VAL = 10

#     for task_id, task_data in challenges.items():
#         task_sol = solutions[task_id]

#         # Process training pairs
#         train_pairs_img, train_pairs_seq = _process_pairs_jax(
#             task_data.get('train', []), MAX_SHAPE, PAD_VAL
#         )

#         # Process test pairs
#         test_inputs = task_data.get('test', [])
#         test_pairs_img, test_pairs_seq, shape_color_infos = _process_test_pairs_jax(
#             test_inputs, task_sol, MAX_SHAPE, PAD_VAL
#         )

#         # Generate XYXYXY pairs
#         img_pairs, img_shape_colors = _generate_xyxyxy_pairs_jax(train_pairs_img, test_pairs_img, shape_color_infos, is_sequence=False)
#         seq_pairs, seq_shape_colors = _generate_xyxyxy_pairs_jax(train_pairs_seq, test_pairs_seq, shape_color_infos, is_sequence=True)

#         yield task_id, img_pairs, seq_pairs


# def convert_int_to_dict_jax(action: chex.Array) -> Dict:
#     """Convert integer action to dictionary using JAX operations."""
#     color = action // 900
#     remainder = action - 900 * color
#     row = remainder // 30
#     col = remainder % 30

#     return {
#         'color': color,
#         'coordinate': jnp.array([row, col])
#     }


# def convert_dict_to_int_jax(dict_action: Dict) -> chex.Array:
#     """Convert dictionary action to integer using JAX operations."""
#     color = dict_action['color']
#     coordinate = dict_action['coordinate']
#     int_action = color * 900 + coordinate[0] * 30 + coordinate[1]
#     return int_action


# def vectorized_convert_int_to_dict_jax(actions: chex.Array) -> Dict:
#     """Vectorized function to convert integer actions to dictionary using JAX."""
#     colors = actions // 900
#     remainders = actions - 900 * colors
#     rows = remainders // 30
#     cols = remainders % 30

#     return {
#         'color': colors,
#         'coordinate': jnp.stack([rows, cols], axis=-1)
#     }


# def vectorized_convert_dict_to_int_jax(dict_actions: Dict) -> chex.Array:
#     """Vectorized function to convert dictionary actions to integers using JAX."""
#     colors = dict_actions['color']
#     coordinates = dict_actions['coordinate']
#     rows = coordinates[..., 0]
#     cols = coordinates[..., 1]
#     int_actions = colors * 900 + rows * 30 + cols
#     return int_actions


# class ArcAgiGridEnvCoord(environment.Environment):
#     """
#     JAX-based ARC AGI Grid Environment for coordinate-based actions.
#     """

#     def __init__(self,
#                  training_challenges,
#                  training_solutions,
#                  evaluation_challenges,
#                  evaluation_solutions,
#                  test_challenges,
#                  train_task_img_dict,
#                  eval_task_img_dict):

#         self.training_challenges = training_challenges
#         self.training_solutions = training_solutions
#         self.evaluation_challenges = evaluation_challenges
#         self.evaluation_solutions = evaluation_solutions
#         self.test_challenges = test_challenges
#         self.train_task_img_dict = train_task_img_dict
#         self.eval_task_img_dict = eval_task_img_dict
#         self.train_task_list = list(self.train_task_img_dict.keys())
#         self.eval_task_list = list(self.eval_task_img_dict.keys())

#     @property
#     def default_params(self) -> EnvParams:
#         """Default environment parameters."""
#         return EnvParams()

#     def step_env(self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams) -> Tuple[chex.Array, EnvState, float, bool, Dict]:
#         """Execute one time within the environment."""
#         dict_action = convert_int_to_dict_jax(action)
#         color = dict_action['color']
#         coordinate = dict_action['coordinate']
#         row = coordinate[0]
#         col = coordinate[1]

#         # Check current cell value before action
#         current_cell_value = state.current_grid_img[row, 90 + col]

#         # Update current grid
#         new_current_grid = state.current_grid_img.at[row, 90 + col].set(color)

#         # Get target color
#         target_color = state.target_grid_img[row, 90 + col]

#         # Update chosen grid
#         new_chosen_grid = state.chosen_grid_img.at[row, col].add(1)

#         # Calculate reward and termination
#         # Case 1: Already selected this cell multiple times
#         reward = jax.lax.select(
#             state.chosen_grid_img[row, col] > 1,
#             -1.0,
#             # Case 2: Wrong color or cell not empty
#             jax.lax.select(
#                 (color != target_color) | (current_cell_value != 11),
#                 -1.0,
#                 # Case 3: Check if puzzle is complete
#                 jax.lax.select(
#                     jnp.array_equal(new_current_grid, state.target_grid_img),
#                     1.0,
#                     # Case 4: Correct intermediate step
#                     jax.lax.select(color != 10, 0.05, 0.01)
#                 )
#             )
#         )

#         # Determine termination
#         terminated = (reward == -1.0) | (reward == 1.0)

#         # Check success
#         is_success = reward == 1.0

#         # Update state
#         new_state = state.replace(
#             current_grid_img=new_current_grid,
#             chosen_grid_img=new_chosen_grid,
#             time=state.time + 1,
#             episode_returns=state.episode_returns + reward,
#             episode_lengths=state.episode_lengths + 1,
#             is_success=is_success
#         )

#         # Observation
#         obs = new_state.current_grid_img

#         # Info
#         info = {
#             'target_grid_img': state.target_grid_img,
#             'time': new_state.time,
#             'task_id': state.task_id,
#             'test_input_idx': state.test_input_idx,
#             'current_grid_img': new_state.current_grid_img,
#             'chosen_grid_img': new_state.chosen_grid_img,
#             'episode_returns': new_state.episode_returns,
#             'episode_lengths': new_state.episode_lengths,
#             'size_candidate': state.size_candidate,
#             'color_candidate': state.color_candidate,
#             'is_success': new_state.is_success,
#         }

#         return obs, new_state, reward, terminated, info

#     def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
#         """Reset the environment to initial state."""

#         # Select task based on mode and parameters
#         if params.task_id is None:
#             # Select random task
#             task_idx = jax.random.choice(key, len(self.train_task_list))
#             task_id = self.train_task_list[task_idx]
#         else:
#             task_id = params.task_id

#         # Select pair index
#         if params.pair_idx is None:
#             if params.mode == 'train':
#                 num_pairs = len(self.train_task_img_dict[task_id])
#                 key, subkey = jax.random.split(key)
#                 pair_idx = jax.random.choice(subkey, num_pairs)
#                 target_grid_img = jnp.array(self.train_task_img_dict[task_id][int(pair_idx)])
#             else:  # evaluation
#                 num_pairs = len(self.eval_task_img_dict[task_id])
#                 key, subkey = jax.random.split(key)
#                 pair_idx = jax.random.choice(subkey, num_pairs)
#                 target_grid_img = jnp.array(self.eval_task_img_dict[task_id][int(pair_idx)])
#         else:
#             pair_idx = params.pair_idx
#             if params.mode == 'train':
#                 target_grid_img = jnp.array(self.train_task_img_dict[task_id][pair_idx])
#             else:  # evaluation
#                 target_grid_img = jnp.array(self.eval_task_img_dict[task_id][pair_idx])

#         # Extract solution area and determine size/color candidates
#         solution_area = target_grid_img[:, 90:]
#         unique_values = jnp.unique(solution_area)
#         non_10_mask = unique_values != 10
#         non_10_values = unique_values[non_10_mask]

#         # Find actual grid size
#         non_10_positions = jnp.where(solution_area != 10)
#         height = jnp.max(non_10_positions[0]) + 1
#         width = jnp.max(non_10_positions[1]) + 1
#         size_candidate = jnp.array([height, width])

#         # Extract valid colors (0-9)
#         valid_color_mask = (unique_values >= 0) & (unique_values <= 9)
#         color_candidate = unique_values[valid_color_mask]

#         max_sum_reward = 0.05 * (height * width - 1) + 1

#         # Initialize grids
#         chosen_grid_img = jnp.zeros((30, 30))
#         current_grid_img = target_grid_img.copy()

#         # Handle random initialization
#         if params.rand_init:
#             # Fill solution area with padding value first
#             current_grid_img = current_grid_img.at[0:30, 90:].set(10)

#             if params.ratio_fill_correct > 0.0:
#                 target_solution = target_grid_img[0:height, 90:90+width]
#                 current_solution = jnp.full((height, width), 11)  # Start with empty

#                 # Random initialization of some cells
#                 total_cells = int(height * width)
#                 num_filled = int(total_cells * params.ratio_fill_correct)

#                 if num_filled > 0:
#                     key, subkey = jax.random.split(key)
#                     filled_indices = jax.random.choice(subkey, total_cells, (num_filled,), replace=False)

#                     # Convert 1D indices to 2D coordinates
#                     filled_rows = filled_indices // width
#                     filled_cols = filled_indices % width

#                     # Use JAX operations to fill chosen cells
#                     for i in range(num_filled):
#                         r, c = int(filled_rows[i]), int(filled_cols[i])
#                         current_solution = current_solution.at[r, c].set(target_solution[r, c])
#                         chosen_grid_img = chosen_grid_img.at[r, c].add(1)

#                 current_grid_img = current_grid_img.at[0:height, 90:90+width].set(current_solution)
#             else:
#                 # Fill only the size_candidate area with empty value (11)
#                 current_grid_img = current_grid_img.at[0:height, 90:90+width].set(11)
#         else:
#             # Fill only the size_candidate area with empty value (11)
#             current_grid_img = current_grid_img.at[0:height, 90:90+width].set(11)

#         # Create initial state
#         state = EnvState(
#             current_grid_img=current_grid_img,
#             target_grid_img=target_grid_img,
#             chosen_grid_img=chosen_grid_img,
#             time=0,
#             task_id=task_id,
#             test_input_idx=int(pair_idx),
#             episode_returns=0.0,
#             episode_lengths=0,
#             size_candidate=size_candidate,
#             color_candidate=color_candidate,
#             is_success=False,
#             max_sum_reward=max_sum_reward
#         )

#         obs = state.current_grid_img
#         return obs, state

#     def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
#         """Check if state is terminal."""
#         return state.time >= params.max_steps_in_episode

#     @property
#     def name(self) -> str:
#         """Environment name."""
#         return "ArcAgiGridEnvCoord-JAX"

#     @property
#     def num_actions(self) -> int:
#         """Number of actions in the environment."""
#         return 9000  # 10 colors * 900 coordinates

#     def action_space(self, params: Optional[EnvParams] = None) -> chex.Array:
#         """Action space of the environment."""
#         return jnp.arange(self.num_actions)

#     def observation_space(self, params: Optional[EnvParams] = None) -> chex.Array:
#         """Observation space of the environment."""
#         return jnp.zeros((30, 180), dtype=jnp.int32)


# def load_challenges_and_solutions_jax(
#     training_challenges_json: str,
#     training_solutions_json: str,
#     evaluation_challenges_json: str,
#     evaluation_solutions_json: str,
#     test_challenges_json: str,
# ) -> Tuple:
#     """Load challenges and solutions for JAX environment."""
#     # Load training, evaluation, test challenge and solutions
#     with open(training_challenges_json, 'r', encoding='utf-8') as file:
#         training_challenges = json.load(file)
#     with open(training_solutions_json, 'r', encoding='utf-8') as file:
#         training_solutions = json.load(file)
#     with open(evaluation_challenges_json, 'r', encoding='utf-8') as file:
#         evaluation_challenges = json.load(file)
#     with open(evaluation_solutions_json, 'r', encoding='utf-8') as file:
#         evaluation_solutions = json.load(file)
#     with open(test_challenges_json, 'r', encoding='utf-8') as file:
#         test_challenges = json.load(file)
#     return training_challenges, training_solutions, \
#            evaluation_challenges, evaluation_solutions, test_challenges