# %%
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import Autoreset
from itertools import permutations, product
import json
from typing import Tuple, Dict, Union, List, Any
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap, Normalize
import torch 

cmap = colors.ListedColormap(
    [
    '#000000', # 0: black
     '#0074D9', # 1: blue
     '#FF4136', # 2: red
     '#2ECC40', # 3: green
     '#FFDC00', # 4: yello
     '#8B00FF', # 5: gray
     '#F012BE', # 6: magenta
     '#FF851B', # 7: oragne
     '#7FDBFF', # 8: sky
     '#870C25', # 9: brwon
     '#AAAAAA', # 10: mask
     '#FFFFFF', # 11: empty
     ])
norm = colors.Normalize(vmin=0, vmax=11)


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


from arc_agi_grid_env import ArcAgiGridEnv

def load_challenges_and_solutions(
                                    training_challenges_json: str,
                                    training_solutions_json: str,
                                    evaluation_challenges_json: str,
                                    evaluation_solutions_json: str,
                                    test_challenges_json: str,
                                    ) -> Tuple:
    # training, evaluation, test challenge 및 solution들을 불러오기
    with open(training_challenges_json, 'r', encoding='utf-8') as file:
        training_challenges = json.load(file)
    with open(training_solutions_json, 'r', encoding='utf-8') as file:
        training_solutions = json.load(file)
    with open(evaluation_challenges_json, 'r', encoding='utf-8') as file:
        evaluation_challenges = json.load(file)
    with open(evaluation_solutions_json, 'r', encoding='utf-8') as file:
        evaluation_solutions = json.load(file)
    with open(test_challenges_json, 'r', encoding='utf-8') as file:
        test_challenges = json.load(file)
    return training_challenges, training_solutions, \
            evaluation_challenges, evaluation_solutions, test_challenges
                
class ArcAgiGridEnvCoord(ArcAgiGridEnv):
    def __init__(self,
                training_challenges,
                training_solutions,
                evaluation_challenges,
                evaluation_solutions,
                test_challenges,
                train_task_img_dict,
                eval_task_img_dict,
                ):
        self.training_challenges = training_challenges
        self.training_solutions = training_solutions
        self.evaluation_challenges = evaluation_challenges
        self.evaluation_solutions = evaluation_solutions
        self.test_challenges = test_challenges
        self.train_task_img_dict = train_task_img_dict
        self.eval_task_img_dict = eval_task_img_dict
        self.train_task_list = list(self.train_task_img_dict.keys())
        self.eval_task_list = list(self.eval_task_img_dict.keys())
        
        # Initialize candidates as empty lists
        self.size_candidate = []
        self.color_candidate = []

        # observation space에 대한 정의
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Box(low=0, high=11, shape=(30,180), dtype=int)

        # action space에 대한 정의 (0~9 색상만)
        self.action_space = gym.spaces.Dict({
            'color': gym.spaces.Discrete(10),
            'coordinate': gym.spaces.MultiDiscrete([30, 30]),
            }
        )

    def _select_task(self, seed) -> str:
        random.seed(seed)
        np.random.seed(seed)
        task_id = random.choice(self.train_task_list)
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

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        if options != None:
            mode = options['mode']
            self.task_id = options['task_id']
            self.pair_idx = options['pair_idx']
            reset_sol_grid = options['reset_sol_grid']
    
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
        if self.pair_idx == None:
            if mode == 'train':
                self.pair_idx = random.choice(list(range(len(self.train_task_img_dict[self.task_id]))))
                self._target_grid_img = self.train_task_img_dict[self.task_id][self.pair_idx]
            elif mode == 'evaluation' or mode == 'eval':
                self.pair_idx = random.choice(list(range(len(self.eval_task_img_dict[self.task_id]))))
                self._target_grid_img = self.eval_task_img_dict[self.task_id][self.pair_idx]
        else:
            if mode == 'train':
                self._target_grid_img = self.train_task_img_dict[self.task_id][self.pair_idx]
            elif mode == 'evaluation' or mode == 'eval':
                self._target_grid_img = self.eval_task_img_dict[self.task_id][self.pair_idx]
        self.test_input_idx = self.pair_idx
        
        # Set size_candidate and color_candidate from options or use defaults FIRST
        if options and 'size_candidate' in options:
            self.size_candidate = options['size_candidate']
        else:
            self.size_candidate = [4, 4]  # Default for task 3cd86f4f
            
        if options and 'color_candidate' in options:
            self.color_candidate = options['color_candidate']
        else:
            self.color_candidate = [0, 1, 4, 5, 9]  # Default for task 3cd86f4f
        
        # target grid에서 test solution에 해당하는 부분을 전부 pad_val으로 masking하고 current grid로 할당
        if reset_sol_grid == 'padding':
            empty_val = 11
            self._current_grid_img = self._target_grid_img.copy()
            # Fill the solution area with different values based on size_candidate
            self._current_grid_img[0:30, 150:] = 10  # Fill entire solution area with 10 first
            # Then fill only the size_candidate area with empty_val (11)
            height, width = self.size_candidate[0], self.size_candidate[1]
            self._current_grid_img[0:height, 150:150+width] = empty_val 
        elif reset_sol_grid == 'random':
            # ! 여기서 solution에 해당하는 부분을 랜덤으로 초기화해도 좋을듯?
            rand_grid = np.random.randint(low=0,
                                            high=11,
                                            size=(30, 30))
            self._current_grid_img = self._target_grid_img.copy()
            self._current_grid_img[0:30, 150:] = rand_grid

        self._chosen_grid_img = np.zeros([30, 30]).astype(int)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict):
        """Execute one timestep within the environment.
        Args:
            action: The action to take (0-10)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        color = action['color']
        coordinate = action['coordinate']
        row = coordinate[0]
        col = coordinate[1]
        index = row*30 + col
        
        # 현재 칸의 값 확인 (action을 취하기 전 값)
        current_cell_value = self._current_grid_img[row, 150+col]
        
        self._current_grid_img[row, 150+col] = color
        self._chosen_grid_img[row, col] += 1
        
        # Log grid changes occasionally
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 1
            
        if self.step_counter % 1000 == 0:  # Log every 1000 steps
            solution_area = self._current_grid_img[:, 150:]
            filled_cells = int(np.sum(solution_area != 11))
            print(f"Step {self.step_counter}: Progress {filled_cells}/16 cells filled")

        target_color_img = self._target_grid_img[row, 150+col]
        
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

    def plot_chosen_grid(self,):
        plt.imshow(self._chosen_grid_img, cmap='viridis')  # 배열을 히트맵으로 표시
        plt.colorbar()  # 색상 바 추가
        # 각 위치에 값 표시
        for i in range(30):
            for j in range(30):
                plt.text(j, i, self._chosen_grid_img[i, j], ha='center', va='center', color='white')
    
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
    
    def __init__(self, env, 
                 fixed_task: bool, 
                 fixed_pair_idx: bool, 
                task_id: str,
                pair_idx: int,):
        super().__init__(env)
        self.fixed_task = fixed_task
        self.fixed_pair_idx = fixed_pair_idx
        self.task_id = task_id
        self.pair_idx = pair_idx
    
    def reset(self, *args, **kwargs,):
        if self.fixed_task and not self.fixed_pair_idx:
            options = {'mode': 'train',
                    'task_id': self.task_id, # 794b24be, 3cd86f4f
                    'pair_idx': None, 
                    'reset_sol_grid': 'padding',}
        elif self.fixed_task and self.fixed_pair_idx:
            options = {'mode': 'train',
                    'task_id': self.task_id, # 794b24be, 3cd86f4f
                    'pair_idx': self.pair_idx, 
                    'reset_sol_grid': 'padding',}
        elif not self.fixed_task and self.fixed_pair_idx:
            options = {'mode': 'train',
                    'task_id': None, # 794b24be, 3cd86f4f
                    'pair_idx': self.pair_idx, 
                    'reset_sol_grid': 'padding',}
        else:
            options = {'mode': 'train',
                    'task_id': None, # 794b24be, 3cd86f4f
                    'pair_idx': None, 
                    'reset_sol_grid': 'padding',}
        # task_id를 항상 포함시켜서 reset 호출
        return self.env.reset(options=options,)
    
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
def create_arc_env_coord(
                        fixed_task: bool, 
                        fixed_pair_idx: bool,
                        task_id: str,
                        pair_idx: int, *args, **kwargs):
    """Factory function to create ARC environment with custom wrapper"""
    base_env = ArcAgiGridEnvCoord(*args, **kwargs)
    wrapped_env = ArcAgiWrapper(base_env, 
                                fixed_task, 
                                fixed_pair_idx,
                                task_id,
                                pair_idx,)
    return wrapped_env


def action_converter(action: int) -> dict:
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
    

def vectorized_action_converter(actions: torch.Tensor) -> dict:
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