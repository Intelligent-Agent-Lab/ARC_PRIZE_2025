# 이 스크립트는 ARC 과제의 훈련 예제를 분석하여, 테스트 문제의 출력 형태(shape)와 색상 팔레트를 예측하는 규칙 기반 엔진입니다.
# 훈련 예제들에서 일관된 변환 규칙을 찾아내고, 이를 테스트 입력에 적용하여 결과를 추론합니다.

import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

# --- 내부 객체 탐지 헬퍼 함수 ---

def _get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """마스크 바운딩 박스."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return (0, 0, -1, -1)  # 마스크가 비어있으면 유효하지 않은 박스 반환
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

def _find_components_by_color(grid: np.ndarray, color: int) -> List[np.ndarray]:
    """단일 색상에 대한 모든 연결된 구성 요소를 찾기"""
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    masks = []
    # 4방향 연결성을 사용하여 이웃을 정의 (상,하,좌,우)
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row, start_col] or grid[start_row, start_col] != color:
                continue
            
            # 새로운 구성 요소를 찾기 위해 BFS 시작
            queue = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component_points = [(start_row, start_col)]
            while queue:
                current_row, current_col = queue.pop(0)
                for delta_row, delta_col in neighbor_offsets:
                    next_row, next_col = current_row + delta_row, current_col + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width and not visited[next_row, next_col] and grid[next_row, next_col] == color:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
                        component_points.append((next_row, next_col))
            
            mask = np.zeros((height, width), dtype=bool)
            rows = [p[0] for p in component_points]
            cols = [p[1] for p in component_points]
            mask[rows, cols] = True
            masks.append(mask)
    return masks


@dataclass(frozen=True)
class Shape:
    height: int
    width: int

    def __repr__(self):
        return f"({self.height}, {self.width})"

def find_objects(grid):
    """그리드 내의 모든 연결된 객체를 찾아 크기 순으로 정렬하여 반환"""
    grid = np.array(grid)
    objects = []
    unique_colors = [int(c) for c in np.unique(grid) if c != 0] # 배경색(0) 제외

    for color in unique_colors:
        masks = _find_components_by_color(grid, color)
        for mask in masks:
            rows, cols = np.where(mask)
            if rows.size > 0:
                objects.append({
                    "color": color,
                    "points": [],
                    "size": len(rows),
                    "bbox": _get_bounding_box(mask)
                })
    return sorted(objects, key=lambda x: x["size"], reverse=True)

# --- 변환 규칙 ---

def _get_inner_objects(train_example):
    """그리드 경계에 닿지 않는 객체들의 리스트를 반환"""
    objects = find_objects(train_example['input'])
    if not objects: return []
    grid_h, grid_w = train_example['input_shape'].height, train_example['input_shape'].width
    return [o for o in objects if o['bbox'][0] > 0 and o['bbox'][1] > 0 and o['bbox'][2] < grid_h - 1 and o['bbox'][3] < grid_w - 1]

# --- 모양(Shape) 변환 ---
# 각 함수는 예시를 받아 예측된 모양을 반환하는 규칙들

def id_transform(train_example): return Shape(height=train_example['input_shape'].height, width=train_example['input_shape'].width)
def transpose_transform(train_example): return Shape(height=train_example['input_shape'].width, width=train_example['input_shape'].height)

def crop_to_all_inner_objects_bbox(train_example):
    inner_objects = _get_inner_objects(train_example)
    if not inner_objects: return None
    min_r = min(o['bbox'][0] for o in inner_objects)
    min_c = min(o['bbox'][1] for o in inner_objects)
    max_r = max(o['bbox'][2] for o in inner_objects)
    max_c = max(o['bbox'][3] for o in inner_objects)
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_to_largest_non_boundary_object_bbox(train_example):
    objects = find_objects(train_example['input'])
    if not objects: return None
    grid_h, grid_w = train_example['input_shape'].height, train_example['input_shape'].width
    non_boundary_objects = [o for o in objects if not (o['bbox'][0] == 0 or o['bbox'][1] == 0 or o['bbox'][2] == grid_h - 1 or o['bbox'][3] == grid_w - 1)]
    if not non_boundary_objects: return None
    lrg_obj = max(non_boundary_objects, key=lambda x: x['size'])
    min_r, min_c, max_r, max_c = lrg_obj['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_to_object_bbox(train_example):
    objects = find_objects(train_example['input'])
    if not objects: return None
    min_r, min_c, max_r, max_c = objects[0]['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def _create_shape_from_object_count_rule(shape_template):
    def rule(train_example):
        n = len(find_objects(train_example['input']))
        if n == 0:
            return None
        height = n if shape_template[0] == 'n' else shape_template[0]
        width = n if shape_template[1] == 'n' else shape_template[1]
        return Shape(height, width)
    return rule

shape_from_object_count_nxn = _create_shape_from_object_count_rule(('n', 'n'))
shape_from_object_count_1xn = _create_shape_from_object_count_rule((1, 'n'))
shape_from_object_count_nx1 = _create_shape_from_object_count_rule(('n', 1))

# --- 색상 변환 규칙 ---

def color_id_transform(train_example): return set(np.array(train_example['input']).flatten())

def color_of_inner_objects(train_example):
    inner_objects = _get_inner_objects(train_example)
    if not inner_objects: return None
    inner_colors = {o['color'] for o in inner_objects}
    inner_colors.add(0) # 배경색은 항상 포함
    return inner_colors

def color_of_largest_object(train_example):
    objects = find_objects(train_example['input'])
    return {objects[0]['color'], 0} if objects else {0}

def color_palette_subtraction(train_example):
    input_colors = Counter(np.array(train_example['input']).flatten())
    if 0 in input_colors: del input_colors[0]
    if not input_colors: return set(np.array(train_example['input']).flatten())
    most_common_color = input_colors.most_common(1)[0][0] # 배경색 (진짜 가장 흔한것) 제외
    return {c for c in np.array(train_example['input']).flatten() if c != most_common_color}

# --- 예측 로직 ---

def _prepare_train_tasks(task):
    train_tasks = []
    for example in task.get('train', []):
        if 'input' not in example or 'output' not in example or not example['input'] or not example['input'][0] or not example['output'] or not example['output'][0]:
            continue
        train_tasks.append({
            "input": example['input'], "output": example['output'],
            "input_shape": Shape(len(example['input']), len(example['input'][0])),
            "output_shape": Shape(len(example['output']), len(example['output'][0]))
        })
    return train_tasks

def predict_shape(train_tasks):
    """모양 예측: 미리 정의된 규칙들을 우선순위에 따라 검사하고, 모든 훈련 예제를 통과하는 첫 번째 규칙을 채택"""
    if not train_tasks: return "fallback_id", id_transform

    simple_rules = [
        ("id_transform", id_transform, False),
        ("transpose_transform", transpose_transform, False),
        ("crop_to_all_inner_objects_bbox", crop_to_all_inner_objects_bbox, True),
        ("crop_to_largest_non_boundary_object_bbox", crop_to_largest_non_boundary_object_bbox, True),
        ("crop_to_object_bbox", crop_to_object_bbox, True),
        ("shape_from_object_count_nxn", shape_from_object_count_nxn, True),
        ("shape_from_object_count_1xn", shape_from_object_count_1xn, True),
        ("shape_from_object_count_nx1", shape_from_object_count_nx1, True),
    ]
    for name, rule_func, check_none in simple_rules:
        try:
            if check_none:
                if all(rule_func(train_example) is not None and rule_func(train_example) == train_example['output_shape'] for train_example in train_tasks):
                    return name, rule_func
            else:
                if all(rule_func(train_example) == train_example['output_shape'] for train_example in train_tasks):
                    return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    # 입력/출력 모양의 높이/너비 차이가 일정한지 확인
    deltas = {(train_example['output_shape'].height - train_example['input_shape'].height, train_example['output_shape'].width - train_example['input_shape'].width) for train_example in train_tasks}
    if len(deltas) == 1:
        delta_height, delta_width = deltas.pop()
        predictor = lambda t, h=delta_height, w=delta_width: Shape(t['input_shape'].height + h, t['input_shape'].width + w)
        return ("crop_transform" if delta_height < 0 or delta_width < 0 else "pad_transform"), predictor

    # 입력 모양을 일정하게 타일링하는지 확인
    factors = set()
    is_tileable = True
    for train_example in train_tasks:
        if train_example['input_shape'].height > 0 and train_example['input_shape'].width > 0 and \
           train_example['output_shape'].height % train_example['input_shape'].height == 0 and \
           train_example['output_shape'].width % train_example['input_shape'].width == 0:
            factors.add((train_example['output_shape'].height // train_example['input_shape'].height, train_example['output_shape'].width // train_example['input_shape'].width))
        else:
            is_tileable = False
            break
    if is_tileable and len(factors) == 1:
        factor_height, factor_width = factors.pop()
        if factor_height > 0 and factor_width > 0:
            return "tile_transform", lambda t, h=factor_height, w=factor_width: Shape(t['input_shape'].height * h, t['input_shape'].width * w)

    # 실패 1: 모든 출력 모양이 같다면 그 모양을 예측값으로 사용
    output_shapes = [train_example['output_shape'] for train_example in train_tasks]
    most_common_shape = Counter(output_shapes).most_common(1)[0][0]
    if all(train_example['output_shape'] == most_common_shape for train_example in train_tasks):
        return "constant_shape", lambda t, shape=most_common_shape: shape

    # 실패 2: 위 모든 규칙 실패 시, 첫 번째 훈련 예제의 출력 모양을 사용
    first_output_shape = train_tasks[0]['output_shape']
    return "fallback_first_train_shape", lambda t, shape=first_output_shape: shape

def predict_color_palette(train_tasks):
    """색상 예측: 모양 예측과 유사하게, 우선순위에 따라 색상 변환 규칙을 검사하고 가장 먼저 일치하는 규칙을 사용"""
    if not train_tasks: return "color_union_transform", lambda t: set()

    # 간단한 규칙부터 우선적으로 확인
    rules = [
        ("color_id_transform", color_id_transform, False),
        ("color_of_inner_objects", color_of_inner_objects, True),
        ("color_of_largest_object", color_of_largest_object, False),
        ("color_palette_subtraction", color_palette_subtraction, False),
    ]
    for name, rule_func, check_none in rules:
        try:
            if all((result := rule_func(train_example)) is not None and result == set(np.array(train_example['output']).flatten()) for train_example in train_tasks) if check_none \
            else all(rule_func(train_example) == set(np.array(train_example['output']).flatten()) for train_example in train_tasks):
                return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    # 실패 규칙 1: 모든 출력 색상이 같다면 그 색상 팔레트를 예측값으로 사용
    first_output_colors = set(np.array(train_tasks[0]['output']).flatten())
    if all(set(np.array(train_example['output']).flatten()) == first_output_colors for train_example in train_tasks[1:]):
        return "color_constant_transform", lambda t, colors=first_output_colors: colors

    # 실패 규칙 2: 위 모든 규칙 실패 시, 모든 훈련 예제의 출력 색상을 합쳐서 사용
    all_output_colors = set.union(*[set(np.array(train_example['output']).flatten()) for train_example in train_tasks])
    return "color_union_transform", lambda t, colors=all_output_colors: colors

def predict_candidates_from_task_id(task_id, training_challenges):
    task = training_challenges.get(task_id)
    if not task: return ([10, 10], list(range(10)))

    train_tasks = _prepare_train_tasks(task)
    if not train_tasks: return ([10, 10], list(range(10)))

    # 모양과 색상에 대한 최적의 규칙을 각각 찾음
    _, shape_rule_func = predict_shape(train_tasks)
    _, color_rule_func = predict_color_palette(train_tasks)

    # 찾은 규칙을 테스트 입력에 적용하여 최종 후보를 예측
    test_inputs = task.get('test', [])
    test_input_grid = test_inputs[0]['input'] if test_inputs and test_inputs[0].get('input') else train_tasks[0]['input']
    
    if not test_input_grid or not test_input_grid[0]: return ([10, 10], list(range(10)))

    leak_free_test_instance = {"input": test_input_grid, "input_shape": Shape(len(test_input_grid), len(test_input_grid[0]))}
    
    predicted_shape = shape_rule_func(leak_free_test_instance)
    predicted_palette = color_rule_func(leak_free_test_instance)

    # 결과를 포맷팅하고, 예측 실패 시 대체(fallback) 값을 사용
    if predicted_shape is None:
        predicted_shape = train_tasks[0]['output_shape']
    size_candidate = [predicted_shape.height, predicted_shape.width]

    color_candidate = []
    if predicted_palette is not None:
        color_candidate = sorted([c for c in predicted_palette if 0 <= c <= 9])
    
    if not color_candidate:
        color_candidate = sorted({int(c) for c in np.array(train_tasks[0]['output']).flatten() if 0 <= c <= 9})

    # 최종 후보값이 비어있지 않도록 보장
    if not size_candidate or len(size_candidate) != 2: size_candidate = [10, 10]
    if not color_candidate: color_candidate = list(range(10))

    return size_candidate, color_candidate

def _print_task_analysis(task_id, train_tasks, results):
    print(f"--- 평가 중: 태스크 ID: {task_id} ---")
    print("[Train Data]")
    for i, train_example in enumerate(train_tasks):
        in_colors = sorted(list({int(c) for c in np.array(train_example['input']).flatten()}))
        out_colors = sorted(list({int(c) for c in np.array(train_example['output']).flatten()}))
        print(f"- 예제 {i+1}: Shape: {train_example['input_shape']} -> {train_example['output_shape']} | Colors: {in_colors} -> {out_colors}")
    
    print("\n[Prediction Result]")
    shape_status = "성공!" if results['shape_correct'] else "실패!"
    color_status = "성공!" if results['color_correct'] else "실패!"
    
    print(f"Shape: {shape_status} | 규칙: {results['shape_rule_name']} | 예측: {results['predicted_shape']} (정답: {results['actual_shape']})")
    print(f"Color: {color_status} | 규칙: {results['color_rule_name']} | 예측: {sorted(list(results['predicted_palette']))} (정답: {sorted(list(results['actual_palette']))})")
    print("-" * 40)
def analyze_dataset(challenge_file, solution_file):
    with open(challenge_file, 'r') as f: challenges = json.load(f)
    with open(solution_file, 'r') as f: solutions = json.load(f)
    stats = {'total': 0, 'shape_correct': 0, 'color_correct': 0, 'total_correct': 0}
    for task_id, task in challenges.items():
        if task_id not in solutions: continue
        
        train_tasks = _prepare_train_tasks(task)
        if not train_tasks: continue
        stats['total'] += 1
        shape_rule_name, shape_rule_func = predict_shape(train_tasks)
        color_rule_name, color_rule_func = predict_color_palette(train_tasks)
        test_input = task['test'][0]['input']
        leak_free_test_instance = {"input": test_input, "input_shape": Shape(len(test_input), len(test_input[0]))}
        predicted_shape = shape_rule_func(leak_free_test_instance)
        predicted_palette = color_rule_func(leak_free_test_instance)
      
        actual_output = solutions[task_id][0]
        actual_shape = Shape(len(actual_output), len(actual_output[0]))
        actual_palette = set(np.array(actual_output).flatten())
        shape_correct = (predicted_shape == actual_shape)
        color_correct = (predicted_palette == actual_palette)
        if shape_correct: stats['shape_correct'] += 1
        if color_correct: stats['color_correct'] += 1
        if shape_correct and color_correct: stats['total_correct'] += 1

        _print_task_analysis(task_id, train_tasks, {
            'shape_correct': shape_correct, 'color_correct': color_correct,
            'shape_rule_name': shape_rule_name, 'color_rule_name': color_rule_name,
            'predicted_shape': predicted_shape, 'actual_shape': actual_shape,
            'predicted_palette': predicted_palette, 'actual_palette': actual_palette
        })

    if stats['total'] > 0:
        print(f"\n--- 최종 결과: {Path(challenge_file).stem} ---")
        print(f"Shape 예측 성공률: {stats['shape_correct']/stats['total']:.2%} ({stats['shape_correct']}/{stats['total']})")
        print(f"Color 예측 성공률: {stats['color_correct']/stats['total']:.2%} ({stats['color_correct']}/{stats['total']})")
        print(f"Total 예측 성공률: {stats['total_correct']/stats['total']:.2%} ({stats['total_correct']}/{stats['total']})")

if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / 'datasets'
    
    print("="*20 + " 훈련 데이터셋 분석 " + "="*20)
    analyze_dataset(
        dataset_path / 'arc-agi_training_challenges.json',
        dataset_path / 'arc-agi_training_solutions.json'
    )
