# %%
import json
import random
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Optional
from itertools import permutations, product
from typing import Tuple, Dict, Union, List, Any
from dataclasses import dataclass

@dataclass
class ActiveShapeColorOntHot:
    '''
    shape: 30 x 30 모양에 0 또는 1값들이 채워짐
            active shape만 1로 채움 (masking value가 아닌 영역)
    color: 길이가 10인 one_hot vector
            사용하는 색만 1로 결정
    '''
    shape: Int[Array, '30 30'] 
    color: Int[Array, '10']


def generate_meta_dataset(challenges: Dict[str, Any],
                          solutions: Dict[str, Any],
                          ):
    MAX_SHAPE = (30, 30)
    PAD_VAL = 10
    meta_dataset = dict()
    
    # 1. train tasks (for adaptation)
    # train_challenges 또는 eval_challenges 내의 train_input_ouput_pair에 대해서
    # 각 task 별로 XYXY를 순열로 구성하여 nP2 개 만큼의 XYXY pair들을 생성
    
    # 2. test tasks (for evaluation)
    # train_input_output, test_input_output으로 XYXY 생성
    # 각 task 별로 XYXY를 product로 구성하여
    # num_train_input_output_pairs X num_test_input_output_pairs
    # 개수만큼 pair들을 생성
    
    # generate meta-train or meta-evaluation dataset
    if solutions is not None:
        for task_id, task_data in challenges.items():
            meta_dataset[task_id] = dict()
            
            task_sol = solutions[task_id]
            
            # train_XY pair padding
            train_pairs_img = _pad_train_xy_pairs(
                task_data.get('train', []), 
                MAX_SHAPE, 
                PAD_VAL
            )
            
            # test_XY pair padding
            test_pairs_img = _pad_test_xy_pairs(
                task_data.get('test', []),
                task_sol, 
                MAX_SHAPE, 
                PAD_VAL
            )
            
            # generate meta-train XYXY dataset
            train_xyxy_pairs, train_shape_colors, dict_train_input_idx_to_pair_idx = _generate_train_xyxy_dataset(
                        train_pairs_img
                    )
            
            meta_dataset[task_id]['train_data'] = train_xyxy_pairs
            meta_dataset[task_id]['train_info'] = train_shape_colors
            meta_dataset[task_id]['train_index_map'] = dict_train_input_idx_to_pair_idx
            
            
            # generate meta-test XYXY dataset
            test_xyxy_pairs, test_shape_colors, dict_test_input_idx_to_pair_idx = _generate_test_xyxy_dataset(
                        train_pairs_img, test_pairs_img
                    )
            meta_dataset[task_id]['test_data'] = test_xyxy_pairs
            meta_dataset[task_id]['test_info'] = test_shape_colors
            meta_dataset[task_id]['test_index_map'] = dict_test_input_idx_to_pair_idx
            
    # generate meta-test dataset
    else: 
        for task_id, task_data in challenges.items():
            meta_dataset[task_id] = dict()
            task_sol = None
            # train_XY pair padding
            train_pairs_img = _pad_train_xy_pairs(
                task_data.get('train', []), 
                MAX_SHAPE, 
                PAD_VAL
            )
            
            # test_XY pair padding
            test_inputs = task_data.get('test', [])
            test_pairs_img = _pad_test_xy_pairs(
                test_inputs, 
                task_sol, 
                MAX_SHAPE, 
                PAD_VAL
            )
            
            # generate meta-train XYXY dataset
            train_xyxy_pairs, train_shape_colors, dict_train_input_idx_to_pair_idx = _generate_train_xyxy_dataset(
                        train_pairs_img
                    )
            
            meta_dataset[task_id]['train_data'] = train_xyxy_pairs
            meta_dataset[task_id]['train_info'] = train_shape_colors
            meta_dataset[task_id]['train_index_map'] = dict_train_input_idx_to_pair_idx
            
            
            # generate meta-test XYX dataset
            test_xyx_pairs, dict_test_input_idx_to_pair_idx = _generate_test_xyx_dataset(
                        train_pairs_img, test_pairs_img
                    )
            meta_dataset[task_id]['test_data'] = test_xyx_pairs
            meta_dataset[task_id]['test_index_map'] = dict_test_input_idx_to_pair_idx

    return meta_dataset
    
        
def _pad_train_xy_pairs(train_pairs: List[Dict],
                       max_shape: Tuple[int, int],
                       pad_val: int,
                       ) -> List[np.ndarray]:
    img_train_pairs = []
    for pair in train_pairs:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Pad grids
        padded_input = _pad_grid(input_grid, max_shape, pad_val)
        padded_output = _pad_grid(output_grid, max_shape, pad_val)

        # Image format: concatenate along width (axis=1)
        xy_img = np.concatenate([padded_input, padded_output], axis=1)
        img_train_pairs.append(xy_img)

    return img_train_pairs


def _pad_test_xy_pairs(test_inputs: List[Dict], 
                       solutions: List, 
                       max_shape: Tuple[int, int], 
                       pad_val: int,
                       ) -> List[np.ndarray]:
    """Process test pairs with their solutions."""
    if solutions is not None:
        img_test_pairs = []
        for test_input, solution in zip(test_inputs, solutions):
            input_grid = np.array(test_input['input'])
            output_grid = np.array(solution)
            
            # Pad grids
            padded_input = _pad_grid(input_grid, max_shape, pad_val)
            padded_output = _pad_grid(output_grid, max_shape, pad_val)
            
            # Image format
            xy_img = np.concatenate([padded_input, padded_output], axis=1)
            img_test_pairs.append(xy_img)
    else:
        img_test_pairs = []
        for test_input in test_inputs:
            input_grid = np.array(test_input['input'])
            
            # Pad grids
            padded_input = _pad_grid(input_grid, max_shape, pad_val)
            padded_output = np.ones_like(padded_input) * 11
            # Image format
            xy_img = np.concatenate([padded_input, padded_output], axis=1)
            img_test_pairs.append(xy_img)
        
    return img_test_pairs

def permutation_index_np(i, j, n):
    return i * (n - 1) + j - np.bool(j > i).astype(int).item()

def _generate_train_xyxy_dataset(train_pairs: List[np.ndarray]) -> Tuple:
    if not train_pairs:
        return [], []
    
    n = len(train_pairs)
    train_pairs_array = np.array(train_pairs)  # (n, 30, 60)
    
    train_xyxy_pairs = []
    train_shape_colors = []
    dict_train_input_idx_to_pair_idx = dict()
    for j in range(n):
        dict_train_input_idx_to_pair_idx[j] = []
        
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # 직접 concatenate
            xyxy_pair = np.concatenate([train_pairs_array[i], train_pairs_array[j]], axis=1)
            train_xyxy_pairs.append(xyxy_pair)
            
            # target_y: 마지막 Y (90: 열)
            target_y = xyxy_pair[:, 90:]
            
            # shape 마스크 (10이 아닌 영역 = 1)
            active_shape = (target_y < 10).astype(np.int8)  # int8로 메모리 절약
            
            # 
            pair_idx = permutation_index_np(i, j, n)
            dict_train_input_idx_to_pair_idx[j].append(pair_idx)
            
            # color one-hot vector
            unique_colors = np.unique(target_y)
            # 11 제외하고 0-9 범위만 선택
            valid_colors = unique_colors[unique_colors < 10]
            ont_hot_color = np.zeros(10, dtype=np.int8)
            ont_hot_color[valid_colors] = 1
            
            shape_color_info = ActiveShapeColorOntHot(
                shape=active_shape,
                color=ont_hot_color
            )
            train_shape_colors.append(shape_color_info)
    
    return train_xyxy_pairs, train_shape_colors, dict_train_input_idx_to_pair_idx


def _generate_test_xyxy_dataset(train_pairs: List[np.ndarray],
                                               test_pairs: List[np.ndarray]) -> Tuple:
    """
    Highly optimized version using numpy arrays.
    """
    if not train_pairs or not test_pairs:
        return [], []
    
    # numpy 배열로 변환 (반복 접근 시 성능 향상)
    train_array = np.array(train_pairs)  # (n_train, 30, 60)
    test_array = np.array(test_pairs)    # (n_test, 30, 60)
    
    n_train, n_test = len(train_pairs), len(test_pairs)
    
    test_xyxy_pairs = []
    corresponding_shape_colors = []
    dict_test_input_idx_to_pair_idx = dict()
    for j in range(n_test):
        dict_test_input_idx_to_pair_idx[j] = []
    
    # 각 test pair의 Y 부분만 미리 추출 (중복 계산 방지)
    test_y_parts = test_array[:, :, 30:]  # (n_test, 30, 30)
    
    for i in range(n_train):
        for j in range(n_test):
            # XYXY 생성
            xyxy_pair = np.concatenate([train_array[i], test_array[j]], axis=1)
            test_xyxy_pairs.append(xyxy_pair)
            pair_idx = i*n_test + j
            dict_test_input_idx_to_pair_idx[j].append(pair_idx)
            # Target Y (미리 추출한 것 사용)
            target_y = test_y_parts[j]
            
            # Shape mask
            active_shape = (target_y < 10).astype(np.int8)
            
            # Color one-hot (안전하게 처리)
            unique_colors = np.unique(target_y)
            valid_colors = unique_colors[unique_colors < 10]
            ont_hot_color = np.zeros(10, dtype=np.int8)
            if valid_colors.size > 0:  # 빈 배열 체크
                ont_hot_color[valid_colors] = 1
            
            shape_color_info = ActiveShapeColorOntHot(
                shape=active_shape,
                color=ont_hot_color
            )
            corresponding_shape_colors.append(shape_color_info)
    
    return test_xyxy_pairs, corresponding_shape_colors, dict_test_input_idx_to_pair_idx

def _generate_test_xyx_dataset(train_pairs: List[np.ndarray],
                                               test_pairs: List[np.ndarray]) -> Tuple:
    """
    Highly optimized version using numpy arrays.
    """
    if not train_pairs or not test_pairs:
        return [], []
    
    # numpy 배열로 변환 (반복 접근 시 성능 향상)
    train_array = np.array(train_pairs)  # (n_train, 30, 60)
    test_array = np.array(test_pairs)    # (n_test, 30, 60)
    
    n_train, n_test = len(train_pairs), len(test_pairs)
    
    test_xyx_pairs = []
    
    dict_test_input_idx_to_pair_idx = dict()
    for j in range(n_test):
        dict_test_input_idx_to_pair_idx[j] = []
    
    for i in range(n_train):
        for j in range(n_test):
            # XYX 생성
            xyx_pair = np.concatenate([train_array[i], test_array[j]], axis=1)
            pair_idx = i*n_test + j
            dict_test_input_idx_to_pair_idx[j].append(pair_idx)
            test_xyx_pairs.append(xyx_pair)
            
    return test_xyx_pairs, dict_test_input_idx_to_pair_idx


def _pad_grid(grid: np.ndarray, target_shape: Tuple[int, int], pad_val: int) -> np.ndarray:
    """Efficiently pad a grid to target shape."""
    pad_height = max(0, target_shape[0] - grid.shape[0])
    pad_width = max(0, target_shape[1] - grid.shape[1])
    
    if pad_height == 0 and pad_width == 0:
        return grid
        
    return np.pad(grid, [(0, pad_height), (0, pad_width)], 
                  mode='constant', constant_values=pad_val)


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


# test code in-file
    # %%
if __name__ == "__main__":
    training_challenges_json = "../datasets/arc-agi_training_challenges.json"
    training_solutions_json = "../datasets/arc-agi_training_solutions.json" 
    evaluation_challenges_json = "../datasets/arc-agi_evaluation_challenges.json"
    evaluation_solutions_json = "../datasets/arc-agi_evaluation_solutions.json"
    test_challenges_json = "../datasets/arc-agi_test_challenges.json"

    training_challenges, training_solutions, evaluation_challenges, \
    evaluation_solutions, test_challenges = load_challenges_and_solutions(
                                            training_challenges_json,
                                            training_solutions_json,
                                            evaluation_challenges_json,
                                            evaluation_solutions_json,
                                            test_challenges_json,
                                        )

    # %%
    task_id_list = list(training_challenges.keys())
    max_train_input_output_pairs = -1
    max_test_input_output_pairs = -1
    min_train_input_output_pairs = 100
    min_test_input_output_pairs = 100
    max_train_pair_id = None
    max_test_pair_id = None
    min_train_pair_id = None
    min_test_pair_id = None
    for task_id, train_test_dict in training_challenges.items():
        train_input_ouput_arrays = train_test_dict['train']
        test_input_ouput_arrays = train_test_dict['test']
        if len(train_input_ouput_arrays) > max_train_input_output_pairs:
            max_train_input_output_pairs = len(train_input_ouput_arrays)
            max_train_pair_id = task_id
        if len(test_input_ouput_arrays) > max_test_input_output_pairs:
            max_test_input_output_pairs = len(test_input_ouput_arrays)
            max_test_pair_id = task_id
        if len(train_input_ouput_arrays) < min_train_input_output_pairs:
            min_train_input_output_pairs = len(train_input_ouput_arrays)
            min_train_pair_id = task_id
        if len(test_input_ouput_arrays) < min_test_input_output_pairs:
            min_test_input_output_pairs = len(test_input_ouput_arrays)
            min_test_pair_id = task_id

    # %%
    print(max_train_input_output_pairs)
    print(max_train_pair_id)
    print(min_train_input_output_pairs)
    print(min_train_pair_id)
    print(max_test_input_output_pairs)
    print(max_test_pair_id)
    print(min_test_input_output_pairs)
    print(min_test_pair_id)

    # %%
    print(len(training_challenges['794b24be']['train']))
    # %%
    print(len(training_challenges['794b24be']['test']))
    
    # %%
    meta_train_dataset = generate_meta_dataset(training_challenges, training_solutions)
    meta_eval_dataset = generate_meta_dataset(evaluation_challenges, evaluation_solutions)
    meta_test_dataset = generate_meta_dataset(test_challenges, None)
    # %%
    meta_train_dataset['794b24be']['train_data']
    print(len(meta_train_dataset['794b24be']['train_data']))
    # %%
    print(len(meta_train_dataset['794b24be']['train_info']))
    # %%
    print(len(meta_train_dataset['794b24be']['train_index_map']))
    # %%
    print((meta_train_dataset['794b24be']['train_index_map']))
    
    
    # %%
    meta_train_dataset['794b24be']['test_data']
    print(len(meta_train_dataset['794b24be']['test_data']))
    # %%
    meta_train_dataset['794b24be']['test_info']
    print(len(meta_train_dataset['794b24be']['test_info']))
    # %%
    print((meta_train_dataset['794b24be']['test_index_map']))
    
    # %%
    from RL_ARC_AGI.env.visualize_arc_agi import ArcAgiVisualizer
    visualizer = ArcAgiVisualizer()
    # %%
    task_id = '794b24be'
    pair_idx = 1
    meta_train_xyxy_ex = meta_train_dataset[task_id]['train_data'][pair_idx]
    meta_train_xyxy_ex
    visualizer.plot_target_grid(meta_train_xyxy_ex, task_id, pair_idx)
    visualizer.plot_one_task(training_challenges,
                             training_solutions,
                             task_id, 
                             mode='train',
                            )
    visualizer.plot_padded_task(meta_train_dataset,
                                task_id,
                                'train',
                                pair_idx,
                                )

    # %%
    meta_test_task_id = list(meta_test_dataset.keys())[0]
    print(meta_test_task_id)
    # %%
    visualizer.plot_padded_task(meta_test_dataset,
                                meta_test_task_id,
                                'test',
                                pair_idx,
                                )
    # %%
    print(len(test_challenges[meta_test_task_id]['train']))
    print(len(test_challenges[meta_test_task_id]['test']))
    print(len(meta_test_dataset[meta_test_task_id]['train_data']))
    print(len(meta_test_dataset[meta_test_task_id]['test_data']))

    # %%
    print(meta_train_dataset['794b24be']['train_info'][0])
    # %%
    print(meta_train_dataset['794b24be']['test_info'][0])
    # %%
    print(meta_test_dataset[meta_test_task_id]['train_index_map'])
    # %%
    print(meta_test_dataset[meta_test_task_id]['test_index_map'])
    # %%
    

