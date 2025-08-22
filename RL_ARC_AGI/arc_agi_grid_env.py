# %%
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from itertools import permutations, product
import json
from typing import Tuple, Dict, Union, List, Any
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap, Normalize

cmap = colors.ListedColormap(
    ['#000000', # 0: black
     '#0074D9', # 1: blue
     '#FF4136', # 2: red
     '#2ECC40', # 3: green
     '#FFDC00', # 4: yello
     '#AAAAAA', # 5: gray
     '#F012BE', # 6: magenta
     '#FF851B', # 7: oragne
     '#7FDBFF', # 8: sky
     '#870C25', # 9: brwon
     '#FFFFFF', # 10: mask
     ])
norm = colors.Normalize(vmin=0, vmax=10)


def preprocess_data(challenges: Dict[str, Any], solutions: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Optimized preprocessing function for ARC AGI 2 dataset.
    
    Args:
        challenges: Dictionary containing challenge data
        solutions: Dictionary containing solution data
        
    Returns:
        Tuple of (dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs)
    """
    MAX_SHAPE = (30, 30)
    PAD_VAL = 10
    
    dict_XYXYXY_img_pairs = {}
    dict_XYXYXY_seq_pairs = {}
    
    # Process each task
    for task_id, task_data in challenges.items():
        task_sol = solutions[task_id]
        
        # Process training pairs
        train_pairs_img, train_pairs_seq = _process_pairs(
            task_data.get('train', []), 
            MAX_SHAPE, 
            PAD_VAL
        )
        
        # Process test pairs
        test_inputs = task_data.get('test', [])
        test_pairs_img, test_pairs_seq = _process_test_pairs(
            test_inputs, 
            task_sol, 
            MAX_SHAPE, 
            PAD_VAL
        )
        
        # Generate XYXYXY pairs efficiently
        dict_XYXYXY_img_pairs[task_id] = _generate_xyxyxy_pairs(
            train_pairs_img, test_pairs_img, is_sequence=False
        )
        
        dict_XYXYXY_seq_pairs[task_id] = _generate_xyxyxy_pairs(
            train_pairs_seq, test_pairs_seq, is_sequence=True
        )
    
    return dict_XYXYXY_img_pairs, dict_XYXYXY_seq_pairs


def _pad_grid(grid: np.ndarray, target_shape: Tuple[int, int], pad_val: int) -> np.ndarray:
    """Efficiently pad a grid to target shape."""
    pad_height = max(0, target_shape[0] - grid.shape[0])
    pad_width = max(0, target_shape[1] - grid.shape[1])
    
    if pad_height == 0 and pad_width == 0:
        return grid
        
    return np.pad(grid, [(0, pad_height), (0, pad_width)], 
                  mode='constant', constant_values=pad_val)


def _process_pairs(pairs: List[Dict], max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process input-output pairs into both image and sequence formats."""
    img_pairs = []
    seq_pairs = []
    
    for pair in pairs:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Pad grids
        padded_input = _pad_grid(input_grid, max_shape, pad_val)
        padded_output = _pad_grid(output_grid, max_shape, pad_val)
        
        # Image format: concatenate along width (axis=1)
        xy_img = np.concatenate([padded_input, padded_output], axis=1)
        img_pairs.append(xy_img)
        
        # Sequence format: flatten and concatenate
        seq_input = padded_input.flatten()
        seq_output = padded_output.flatten()
        xy_seq = np.concatenate([seq_input, seq_output])
        seq_pairs.append(xy_seq)
    
    return img_pairs, seq_pairs


def _process_test_pairs(test_inputs: List[Dict], solutions: List, max_shape: Tuple[int, int], pad_val: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process test pairs with their solutions."""
    img_pairs = []
    seq_pairs = []
    
    for test_input, solution in zip(test_inputs, solutions):
        input_grid = np.array(test_input['input'])
        output_grid = np.array(solution)
        
        # Pad grids
        padded_input = _pad_grid(input_grid, max_shape, pad_val)
        padded_output = _pad_grid(output_grid, max_shape, pad_val)
        
        # Image format
        xy_img = np.concatenate([padded_input, padded_output], axis=1)
        img_pairs.append(xy_img)
        
        # Sequence format
        seq_input = padded_input.flatten()
        seq_output = padded_output.flatten()
        xy_seq = np.concatenate([seq_input, seq_output])
        seq_pairs.append(xy_seq)
    
    return img_pairs, seq_pairs


def _generate_xyxyxy_pairs(train_pairs: List[np.ndarray], test_pairs: List[np.ndarray], is_sequence: bool) -> List[np.ndarray]:
    """Generate XYXYXY pairs from training and test data."""
    if not train_pairs or not test_pairs:
        return []
    
    # Generate all training pair permutations (XYXY format)
    train_xyxy_pairs = []
    for p in permutations(train_pairs, 2):
        if is_sequence:
            xyxy_pair = np.concatenate(p)
        else:
            xyxy_pair = np.hstack(p)
        train_xyxy_pairs.append(xyxy_pair)
    
    # Generate XYXYXY combinations
    xyxyxy_pairs = []
    for train_pair, test_pair in product(train_xyxy_pairs, test_pairs):
        if is_sequence:
            xyxyxy_pair = np.concatenate([train_pair, test_pair])
        else:
            xyxyxy_pair = np.hstack([train_pair, test_pair])
        xyxyxy_pairs.append(xyxyxy_pair)
    
    return xyxyxy_pairs


# Memory-efficient version for large datasets
def preprocess_data_generator(challenges: Dict[str, Any], solutions: Dict[str, Any]):
    """
    Generator version that yields one task at a time to reduce memory usage.
    
    Yields:
        Tuple of (task_id, img_pairs, seq_pairs)
    """
    MAX_SHAPE = (30, 30)
    PAD_VAL = 10
    
    for task_id, task_data in challenges.items():
        task_sol = solutions[task_id]
        
        # Process training pairs
        train_pairs_img, train_pairs_seq = _process_pairs(
            task_data.get('train', []), MAX_SHAPE, PAD_VAL
        )
        
        # Process test pairs
        test_inputs = task_data.get('test', [])
        test_pairs_img, test_pairs_seq = _process_test_pairs(
            test_inputs, task_sol, MAX_SHAPE, PAD_VAL
        )
        
        # Generate XYXYXY pairs
        img_pairs = _generate_xyxyxy_pairs(train_pairs_img, test_pairs_img, is_sequence=False)
        seq_pairs = _generate_xyxyxy_pairs(train_pairs_seq, test_pairs_seq, is_sequence=True)
        
        yield task_id, img_pairs, seq_pairs


class ArcAgiGridEnv(gym.Env):
    def __init__(self,
                 training_challenges_json: str,
                 training_solutions_json: str,
                 evaluation_challenges_json: str,
                 evaluation_solutions_json: str,
                 test_challenges_json: str,
                 ):
        # training, evaluation, test challenge 및 solution들을 불러오기
        with open(training_challenges_json, 'r', encoding='utf-8') as file:
            self.training_challenges = json.load(file)
        with open(training_solutions_json, 'r', encoding='utf-8') as file:
            self.training_solutions = json.load(file)
        with open(evaluation_challenges_json, 'r', encoding='utf-8') as file:
            self.evaluation_challenges = json.load(file)
        with open(evaluation_solutions_json, 'r', encoding='utf-8') as file:
            self.evaluation_solutions = json.load(file)
        if test_challenges_json is not None:
            with open(test_challenges_json, 'r', encoding='utf-8') as file:
                self.test_challenges = json.load(file)
        self.train_task_img_dict, self.train_task_seq_dict = preprocess_data(self.training_challenges, self.training_solutions)
        self.eval_task_img_dict, self.eval_task_seq_dict = preprocess_data(self.evaluation_challenges, self.evaluation_solutions)
        self.train_task_list = list(self.train_task_img_dict.keys())
        self.eval_task_list = list(self.eval_task_img_dict.keys())

        # observation space에 대한 정의
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "current_grid_img": gym.spaces.Box(low=0, high=10, shape=(30,180), dtype=int),
                "current_grid_seq": gym.spaces.Box(low=0, high=10, shape=(5400,), dtype=int),
            }
        )
        # action space에 대한 정의 (0~9 색상, 10: 마스크)
        self.action_space = gym.spaces.Discrete(11)

    def _select_task(self, seed) -> str:
        random.seed(seed)
        np.random.seed(seed)
        task_id = random.choice(self.train_task_list)
        return task_id

    def _get_obs(self) -> Dict:
        return {"current_grid_img": self._current_grid_img,
                "current_grid_seq": self._current_grid_seq}

    def _get_info(self) -> Dict:
        return {
            'target_grid_img': self._target_grid_img,
            'target_grid_seq': self._target_grid_seq,
            'timestep': self.timestep,
            'task_id': self.task_id,
            'test_input_idx': self.test_input_idx,
        }

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        
        mode = options['mode']
        task_id = options['task_id']
        reset_sol_grid = options['reset_sol_grid']
        
        self.timestep = 0
        if task_id == None:
            task_id = self._select_task(seed)
        self.task_id = task_id
        """
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
            mode: (train, evaluation, test)
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        # task_id에 해당하는 target grid 선택 (test input이 여러 개 존재 가능하므로 한 번 더 random.choice 수행
        if mode == 'train':
            test_input_idx = random.choice(list(range(len(self.train_task_img_dict[task_id]))))
            self._target_grid_img = self.train_task_img_dict[task_id][test_input_idx]
            self._target_grid_seq = self.train_task_seq_dict[task_id][test_input_idx]
        elif mode == 'evaluation' or mode == 'eval':
            test_input_idx = random.choice(list(range(len(self.eval_task_img_dict[task_id]))))
            self._target_grid_img = self.eval_task_img_dict[task_id][test_input_idx]
            self._target_grid_seq = self.eval_task_seq_dict[task_id][test_input_idx]
        self.test_input_idx = test_input_idx
        # target grid에서 test solution에 해당하는 부분을 전부 pad_val으로 masking하고 current grid로 할당
        if reset_sol_grid == 'padding':
            pad_val= 10
            self._current_grid_img = self._target_grid_img.copy()
            self._current_grid_img[0:30, 150:] = pad_val # H: 0:30 W: 630:660
            self._current_grid_seq = self._target_grid_seq.copy()
            self._current_grid_seq[4500:] = pad_val
        # target grid에서 test solution에 해당하는 부분을 전부 random value로 채우고 current grid로 할당
        elif reset_sol_grid == 'random':
            rand_grid = np.random.randint(low=0,
                                            high=10,
                                            size=(30, 30))
            self._current_grid_img = self._target_grid_img.copy()
            self._current_grid_img[0:30, 150:] = rand_grid
            self._current_grid_seq = self._target_grid_seq.copy()
            self._current_grid_seq[4500:] = rand_grid.flatten()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Union[int]):
        """Execute one timestep within the environment.
        Args:
            action: The action to take (0-10)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        row = self.timestep // 30
        col = self.timestep % 30
        self._current_grid_img[row, 150+col] = action # H: 0:30 W: 630:660
        self._current_grid_seq[4500+self.timestep] = action
        
        # target_action이 취한 action과 같은지 검사
        target_action_img = self._target_grid_img[row, 150+col]
        target_action_seq = self._target_grid_seq[4500+self.timestep]
        assert target_action_img == target_action_seq
        if action != target_action_seq:
            terminated = True
            reward = -1
        else:
            reward = 0.01
            terminated = False
        truncated = (self.timestep == (900 - 1))
        if truncated and not terminated:
            reward = 1 + 0.01
        # TODO: 매 step 마다 +1/900 -1/900 못 맞추는 즉시, 에피소드 종료, 다 맞추면 +1
        observation = self._get_obs()
        info = self._get_info()
        self.timestep += 1
        return observation, reward, terminated, truncated, info

    def plot_current_task_and_sol(self, mode='train'):
        self.plot_one_task(mode, self.task_id)
        
    def plot_one_task(self, mode, task_id, size=2.5, w1=0.9):
        if mode == 'train':
            task = self.training_challenges[task_id]
            task_solutions = self.training_solutions[task_id]
        elif mode == 'evaluation' or self.mode == 'eval':
            task = self.evaluation_challenges[task_id]
            task_solutions = self.evaluation_solutions[task_id]
        else:
            raise NotImplementedError
        titleSize=16
        num_train = len(task['train'])
        num_test  = len(task['test'])
        wn=num_train+num_test
        fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
        plt.suptitle(f'Task #{task_id}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')
        '''train:'''
        for j in range(num_train):
            self.plot_one(axs[0, j], j,task, 'train', 'input',  w=w1)
            self.plot_one(axs[1, j], j,task, 'train', 'output', w=w1)
        '''test:'''
        for k in range(num_test):
            self.plot_one(axs[0, j+k+1], k, task, 'test', 'input', w=w1)
            task['test'][k]['output'] = task_solutions[k]
            self.plot_one(axs[1, j+k+1], k, task, 'test', 'output', w=w1)
        axs[1, j+1].set_xticklabels([])
        axs[1, j+1].set_yticklabels([])
        axs[1, j+1] = plt.figure(1).add_subplot(111)
        axs[1, j+1].set_xlim([0, wn])
        '''Separators:'''
        colorSeparator = 'white'
        for m in range(1, wn):
            axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
        axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)
        axs[1, j+1].axis("off")
        '''Frame and background:'''
        fig.patch.set_linewidth(5) #widthframe
        fig.patch.set_edgecolor('black') #colorframe
        fig.patch.set_facecolor('#444444') #background
        plt.tight_layout()
        print(f'#{task_id}') # for fast and convinience search
        plt.show()

    def plot_one(self, ax, i, task, train_or_test, input_or_output, solution=None, w=0.8):
        fs=12
        input_matrix = task[train_or_test][i][input_or_output]
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        '''Grid:'''
        ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        ax.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        ax.set_title(train_or_test + ' ' + input_or_output, fontsize=fs, color = '#dddddd')

    def plot_original_task(self, task_id, train_or_test, i, input_or_output, mode='train', w=0.8):
        fs=12
        if mode == 'train':
            task = self.training_challenges[task_id]
        elif mode == 'evaluation' or self.mode == 'eval':
            task = self.evaluation_challenges[task_id]
        input_matrix = task[train_or_test][i][input_or_output]
        plt.imshow(input_matrix, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {task_id}' + '  ' + train_or_test + ' ' + input_or_output + f'  #{i}', fontsize=fs, color = '#000000')

    def plot_padded_task(self, task_id, i, w=0.5):
        fs=12
        task = self.train_task_img_dict[task_id]
        input_matrix = task[i]
        plt.figure(figsize=(100, 200)) #
        plt.imshow(input_matrix, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(input_matrix))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {task_id}' + '   ' + f'#{i}', fontsize=fs, color = '#000000')

    def plot_current_grid(self, w=0.5):
        fs=12
        test_sol_current_mat = self._current_grid_img[:, 150:]
        plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {self.task_id}' + '   ' + f'#{self.test_input_idx}', fontsize=fs, color = '#000000')

    def plot_target_grid(self, w=0.5):
        fs=12
        test_sol_target_mat = self._target_grid_img[:, 150:]
        plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
        plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        # '''Grid:'''
        plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
        plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
        plt.tick_params(axis='both', color='none', length=0)
        '''sub title:'''
        plt.title(f'task: {self.task_id}' + '   ' + f'#{self.test_input_idx}', fontsize=fs, color = '#000000')

    def print_train_task_info(self, task_id):
        print(f"training_challenges: num_train_pairs: {len(self.training_challenges[task_id]['train'])}")
        print(f"training_challenges: num_test_pairs: {len(self.training_challenges[task_id]['test'])}")
        
class ArcAgiWrapper(Wrapper):
    """
    Custom wrapper that preserves access to all custom methods
    while maintaining compatibility with Gymnasium's interface.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def __getattr__(self, name):
        """
        Forward any attribute access to the wrapped environment.
        This allows access to custom methods like print_train_task_info.
        """
        return getattr(self.env, name)
    
    # 필요한 경우 특정 메서드들을 명시적으로 forwarding
    def print_train_task_info(self, task_id):
        return self.env.print_train_task_info(task_id)
    
    def plot_current_grid(self, w=0.5):
        return self.env.plot_current_grid(w)
    
    def plot_target_grid(self, w=0.5):
        return self.env.plot_target_grid(w)
    
    def plot_current_task_and_sol(self, mode='train'):
        return self.env.plot_current_task_and_sol(mode)
    
    def plot_one_task(self, mode, task_id, size=2.5, w1=0.9):
        return self.env.plot_one_task(mode, task_id, size, w1)
    
    def plot_original_task(self, task_id, train_or_test, i, input_or_output, mode='train', w=0.8):
        return self.env.plot_original_task(task_id, train_or_test, i, input_or_output, mode, w)
    
    def plot_padded_task(self, task_id, i, w=0.5):
        return self.env.plot_padded_task(task_id, i, w)

# 사용 예시
def create_arc_env(*args, **kwargs):
    """Factory function to create ARC environment with custom wrapper"""
    base_env = ArcAgiGridEnv(*args, **kwargs)
    wrapped_env = ArcAgiWrapper(base_env)
    return wrapped_env