import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import math

# --- 내부 객체 탐지 헬퍼 함수 ---

def get_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """마스크로부터 바운딩 박스를 계산합니다."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return (0, 0, -1, -1)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

def find_components(grid: np.ndarray, color: int) -> List[np.ndarray]:
    """주어진 색상에 대한 모든 연결된 컴포넌트(마스크)를 찾습니다."""
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    masks = []
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row, start_col] or grid[start_row, start_col] != color:
                continue
            
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
    unique_colors = [int(c) for c in np.unique(grid) if c != 0]

    for color in unique_colors:
        masks = find_components(grid, color)
        for mask in masks:
            rows, cols = np.where(mask)
            if rows.size > 0:
                objects.append({
                    "color": color,
                    "points": [],
                    "size": len(rows),
                    "bbox": get_bbox(mask)
                })
    return sorted(objects, key=lambda x: x["size"], reverse=True)

# --- 변환 규칙 ---

def get_inner_objects(train_example):
    """그리드 경계에 닿지 않는 객체들의 리스트를 반환"""
    objects = find_objects(train_example['input'])
    if not objects: return []
    grid_h, grid_w = train_example['input_shape'].height, train_example['input_shape'].width
    return [o for o in objects if o['bbox'][0] > 0 and o['bbox'][1] > 0 and o['bbox'][2] < grid_h - 1 and o['bbox'][3] < grid_w - 1]

# --- 모양(Shape) 변환 규칙 ---

def identity_shape(train_example):
    """입력과 동일한 모양을 반환합니다."""
    return Shape(height=train_example['input_shape'].height, width=train_example['input_shape'].width)

def transpose_shape(train_example):
    """입력 모양의 높이와 너비를 바꾼 모양을 반환합니다."""
    return Shape(height=train_example['input_shape'].width, width=train_example['input_shape'].height)

def crop_inner_bbox(train_example):
    """모든 내부 객체들을 포함하는 최소 바운딩 박스 모양을 반환합니다."""
    inner_objects = get_inner_objects(train_example)
    if not inner_objects: return None
    min_r = min(o['bbox'][0] for o in inner_objects)
    min_c = min(o['bbox'][1] for o in inner_objects)
    max_r = max(o['bbox'][2] for o in inner_objects)
    max_c = max(o['bbox'][3] for o in inner_objects)
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_main_inner_obj(train_example):
    """경계에 닿지 않는 가장 큰 객체의 바운딩 박스 모양을 반환합니다."""
    objects = find_objects(train_example['input'])
    if not objects: return None
    grid_h, grid_w = train_example['input_shape'].height, train_example['input_shape'].width
    non_boundary_objects = [o for o in objects if not (o['bbox'][0] == 0 or o['bbox'][1] == 0 or o['bbox'][2] == grid_h - 1 or o['bbox'][3] == grid_w - 1)]
    if not non_boundary_objects: return None
    lrg_obj = max(non_boundary_objects, key=lambda x: x['size'])
    min_r, min_c, max_r, max_c = lrg_obj['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def crop_main_obj(train_example):
    """가장 큰 객체의 바운딩 박스 모양을 반환합니다."""
    objects = find_objects(train_example['input'])
    if not objects: return None
    min_r, min_c, max_r, max_c = objects[0]['bbox']
    return Shape(height=max_r - min_r + 1, width=max_c - min_c + 1)

def create_count_shaper(shape_template):
    """객체 수에 따라 모양을 결정하는 규칙 함수를 생성합니다."""
    def rule(train_example):
        n = len(find_objects(train_example['input']))
        if n == 0: return None
        height = n if shape_template[0] == 'n' else shape_template[0]
        width = n if shape_template[1] == 'n' else shape_template[1]
        return Shape(height, width)
    return rule

shape_by_count_nxn = create_count_shaper(('n', 'n'))
shape_by_count_1xn = create_count_shaper((1, 'n'))
shape_by_count_nx1 = create_count_shaper(('n', 1))

def create_inner_count_shaper(shape_template):
    """내부 객체 수에 따라 모양을 결정하는 규칙 함수를 생성합니다."""
    def rule(train_example):
        n = len(get_inner_objects(train_example))
        if n == 0: return None
        height = n if shape_template[0] == 'n' else shape_template[0]
        width = n if shape_template[1] == 'n' else shape_template[1]
        return Shape(height, width)
    return rule

shape_by_inner_count_nxn = create_inner_count_shaper(('n', 'n'))
shape_by_inner_count_1xn = create_inner_count_shaper((1, 'n'))
shape_by_inner_count_nx1 = create_inner_count_shaper(('n', 1))

def assemble_shape_wide(train_example):
    """모든 객체의 총 면적을 기반으로 가로가 긴 모양을 조립합니다."""
    objects = find_objects(train_example['input'])
    if not objects: return None
    total_area = sum(o['size'] for o in objects)
    if total_area == 0: return None
    h = 1
    for i in range(1, int(math.sqrt(total_area)) + 1):
        if total_area % i == 0: h = i
    w = total_area // h
    return Shape(height=h, width=w)

def assemble_shape_tall(train_example):
    """모든 객체의 총 면적을 기반으로 세로가 긴 모양을 조립합니다."""
    objects = find_objects(train_example['input'])
    if not objects: return None
    total_area = sum(o['size'] for o in objects)
    if total_area == 0: return None
    h = 1
    for i in range(1, int(math.sqrt(total_area)) + 1):
        if total_area % i == 0: h = i
    w = total_area // h
    return Shape(height=w, width=h)

# --- 색상(Palette) 변환 규칙 ---

def identity_palette(train_example):
    """입력과 동일한 색상 팔레트를 반환합니다."""
    return set(np.array(train_example['input']).flatten())

def get_inner_palette(train_example):
    """내부 객체들의 색상으로 구성된 팔레트를 반환합니다."""
    inner_objects = get_inner_objects(train_example)
    if not inner_objects: return None
    inner_colors = {o['color'] for o in inner_objects}
    inner_colors.add(0)
    return inner_colors

def get_largest_palette(train_example):
    """가장 큰 객체의 색상으로 구성된 팔레트를 반환합니다."""
    objects = find_objects(train_example['input'])
    return {objects[0]['color'], 0} if objects else {0}

def subtract_common_color(train_example):
    """배경을 제외한 가장 흔한 색상을 제거한 팔레트를 반환합니다."""
    input_colors = Counter(np.array(train_example['input']).flatten())
    if 0 in input_colors: del input_colors[0]
    if not input_colors: return set(np.array(train_example['input']).flatten())
    most_common_color = input_colors.most_common(1)[0][0]
    return {c for c in np.array(train_example['input']).flatten() if c != most_common_color}

def get_unique_palette(train_example):
    """유일하게 한 번만 나타나는 색상들로 구성된 팔레트를 반환합니다."""
    input_colors = Counter(np.array(train_example['input']).flatten())
    if 0 in input_colors: del input_colors[0]
    unique_colors = {color for color, count in input_colors.items() if count == 1}
    if not unique_colors: return None
    unique_colors.add(0)
    return unique_colors

def fill_canvas_palette(train_example):
    """가장 흔한 색(캔버스)을 가장 작은 객체(패턴)의 색으로 교체한 팔레트를 예측합니다."""
    input_grid = np.array(train_example['input'])
    input_colors = Counter(input_grid.flatten())
    if 0 in input_colors: del input_colors[0]
    if not input_colors: return None
    canvas_color = input_colors.most_common(1)[0][0]
    objects = find_objects(input_grid)
    if not objects or len(objects) < 2: return None
    smallest_object = objects[-1]
    pattern_colors = {smallest_object['color']}
    input_palette = set(input_grid.flatten())
    predicted_palette = (input_palette - {canvas_color}) | pattern_colors
    return predicted_palette

# --- 예측 로직 ---

def prepare_tasks(task):
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
    if not train_tasks: return "fallback_id", identity_shape

    rules = [
        ("identity_shape", identity_shape, False),
        ("transpose_shape", transpose_shape, False),
        ("crop_inner_bbox", crop_inner_bbox, True),
        ("crop_main_inner_obj", crop_main_inner_obj, True),
        ("crop_main_obj", crop_main_obj, True),
        ("shape_by_count_nxn", shape_by_count_nxn, True),
        ("shape_by_count_1xn", shape_by_count_1xn, True),
        ("shape_by_count_nx1", shape_by_count_nx1, True),
        ("shape_by_inner_count_nxn", shape_by_inner_count_nxn, True),
        ("shape_by_inner_count_1xn", shape_by_inner_count_1xn, True),
        ("shape_by_inner_count_nx1", shape_by_inner_count_nx1, True),
        ("assemble_shape_wide", assemble_shape_wide, True),
        ("assemble_shape_tall", assemble_shape_tall, True),
    ]
    for name, rule_func, check_none in rules:
        try:
            if check_none:
                if all(rule_func(train_example) is not None and rule_func(train_example) == train_example['output_shape'] for train_example in train_tasks):
                    return name, rule_func
            else:
                if all(rule_func(train_example) == train_example['output_shape'] for train_example in train_tasks):
                    return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    deltas = {(t['output_shape'].height - t['input_shape'].height, t['output_shape'].width - t['input_shape'].width) for t in train_tasks}
    if len(deltas) == 1:
        delta_height, delta_width = deltas.pop()
        predictor = lambda t, h=delta_height, w=delta_width: Shape(t['input_shape'].height + h, t['input_shape'].width + w)
        return ("crop_transform" if delta_height < 0 or delta_width < 0 else "pad_transform"), predictor

    factors = set()
    is_tileable = True
    for t in train_tasks:
        if t['input_shape'].height > 0 and t['input_shape'].width > 0 and t['output_shape'].height % t['input_shape'].height == 0 and t['output_shape'].width % t['input_shape'].width == 0:
            factors.add((t['output_shape'].height // t['input_shape'].height, t['output_shape'].width // t['input_shape'].width))
        else:
            is_tileable = False
            break
    if is_tileable and len(factors) == 1:
        factor_height, factor_width = factors.pop()
        if factor_height > 0 and factor_width > 0:
            return "tile_transform", lambda t, h=factor_height, w=factor_width: Shape(t['input_shape'].height * h, t['input_shape'].width * w)

    output_shapes = [t['output_shape'] for t in train_tasks]
    most_common_shape = Counter(output_shapes).most_common(1)[0][0]
    if all(t['output_shape'] == most_common_shape for t in train_tasks):
        return "constant_shape", lambda t, shape=most_common_shape: shape

    first_output_shape = train_tasks[0]['output_shape']
    return "fallback_first_train_shape", lambda t, shape=first_output_shape: shape

def predict_palette(train_tasks):
    """색상 예측: 모양 예측과 유사하게, 우선순위에 따라 색상 변환 규칙을 검사하고 가장 먼저 일치하는 규칙을 사용"""
    if not train_tasks: return "color_union_transform", lambda t: set()

    try:
        input_colors_union = set.union(*[set(np.array(t['input']).flatten()) for t in train_tasks])
        output_colors_union = set.union(*[set(np.array(t['output']).flatten()) for t in train_tasks])
        instructional_colors = input_colors_union - output_colors_union
        if instructional_colors:
            def instructional_rule_predictor(train_example):
                input_palette = set(np.array(train_example['input']).flatten())
                return input_palette - instructional_colors
            if all(instructional_rule_predictor(t) == set(np.array(t['output']).flatten()) for t in train_tasks):
                return "color_instructional_subtraction", instructional_rule_predictor
    except (AttributeError, TypeError, IndexError):
        pass

    rules = [
        ("identity_palette", identity_palette, False),
        ("get_inner_palette", get_inner_palette, True),
        ("get_largest_palette", get_largest_palette, False),
        ("subtract_common_color", subtract_common_color, False),
        ("get_unique_palette", get_unique_palette, True),
        ("fill_canvas_palette", fill_canvas_palette, True),
    ]
    for name, rule_func, check_none in rules:
        try:
            if check_none:
                if all((result := rule_func(t)) is not None and result == set(np.array(t['output']).flatten()) for t in train_tasks):
                    return name, rule_func
            else:
                if all(rule_func(t) == set(np.array(t['output']).flatten()) for t in train_tasks):
                    return name, rule_func
        except (AttributeError, TypeError, IndexError):
            continue

    first_output_colors = set(np.array(train_tasks[0]['output']).flatten())
    if all(set(np.array(t['output']).flatten()) == first_output_colors for t in train_tasks[1:]):
        return "color_constant_transform", lambda t, colors=first_output_colors: colors

    all_output_colors = set.union(*[set(np.array(t['output']).flatten()) for t in train_tasks])
    return "color_union_transform", lambda t, colors=all_output_colors: colors

def get_candidates_from_target(target_grid_img):
    solution_area = target_grid_img[:, -30:]
    unique_values = np.unique(solution_area)
    non_10_values = unique_values[unique_values != 10]
    if len(non_10_values) > 0:
        rows, cols = np.where(solution_area != 10)
        if len(rows) > 0 and len(cols) > 0:
            height = max(rows) + 1
            width = max(cols) + 1
            size_candidate = [height, width]
        else:
            raise ValueError("Cannot determine grid size: non-10 values has negative height or width")
    else:
        raise ValueError("Cannot determine grid size: no non-10 values found in solution area")

    unique_colors = np.unique(solution_area)
    valid_colors = unique_colors[(unique_colors >= 0) & (unique_colors <= 9)]
    if len(valid_colors) == 0:
        raise ValueError("Cannot determine color candidates: no valid colors (0-9) found in solution area")
    color_candidate = valid_colors.tolist()
    return size_candidate, color_candidate

def predict_candidates_from_task_id(task_id, training_challenges):
    task = training_challenges.get(task_id)
    if not task: return ([10, 10], list(range(10)))

    train_tasks = prepare_tasks(task)
    if not train_tasks: return ([10, 10], list(range(10)))

    _, shape_rule_func = predict_shape(train_tasks)
    _, color_rule_func = predict_palette(train_tasks)

    test_inputs = task.get('test', [])
    test_input_grid = test_inputs[0]['input'] if test_inputs and test_inputs[0].get('input') else train_tasks[0]['input']
    
    if not test_input_grid or not test_input_grid[0]: return ([10, 10], list(range(10)))

    leak_free_test_instance = {"input": test_input_grid, "input_shape": Shape(len(test_input_grid), len(test_input_grid[0]))}
    
    predicted_shape = shape_rule_func(leak_free_test_instance)
    predicted_palette = color_rule_func(leak_free_test_instance)

    if predicted_shape is None:
        predicted_shape = train_tasks[0]['output_shape']
    size_candidate = [predicted_shape.height, predicted_shape.width]

    color_candidate = []
    if predicted_palette is not None:
        color_candidate = sorted([c for c in predicted_palette if 0 <= c <= 9])
    
    if not color_candidate:
        color_candidate = sorted({int(c) for c in np.array(train_tasks[0]['output']).flatten() if 0 <= c <= 9})

    if not size_candidate or len(size_candidate) != 2: size_candidate = [10, 10]
    if not color_candidate: color_candidate = list(range(10))

    return size_candidate, color_candidate

def print_task_analysis(task_id, train_tasks, results):
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
    predicted_palette_str = sorted(list(results['predicted_palette'])) if results['predicted_palette'] is not None else []
    actual_palette_str = sorted(list(results['actual_palette']))
    print(f"Color: {color_status} | 규칙: {results['color_rule_name']} | 예측: {predicted_palette_str} (정답: {actual_palette_str})")
    print("-" * 40)

def analyze_dataset(challenge_file, solution_file):
    with open(challenge_file, 'r') as f: challenges = json.load(f)
    with open(solution_file, 'r') as f: solutions = json.load(f)
    
    stats = {'total': 0, 'shape_correct': 0, 'color_correct': 0, 'total_correct': 0}
    shape_rule_success_counts = Counter()
    color_rule_success_counts = Counter()
    non_fallback_stats = {
        'shape_total': 0, 'shape_correct': 0,
        'color_total': 0, 'color_correct': 0
    }
    
    fallback_shape_rules = {"crop_transform", "pad_transform", "tile_transform", "constant_shape", "fallback_first_train_shape", "fallback_id"}
    fallback_color_rules = {"color_constant_transform", "color_union_transform"}

    for task_id, task in challenges.items():
        if task_id not in solutions: continue
        
        train_tasks = prepare_tasks(task)
        if not train_tasks: continue
        stats['total'] += 1
        
        shape_rule_name, shape_rule_func = predict_shape(train_tasks)
        color_rule_name, color_rule_func = predict_palette(train_tasks)
        
        test_input = task['test'][0]['input']
        leak_free_test_instance = {"input": test_input, "input_shape": Shape(len(test_input), len(test_input[0]))}
        
        predicted_shape = shape_rule_func(leak_free_test_instance)
        predicted_palette = color_rule_func(leak_free_test_instance)

        if predicted_palette is None:
            predicted_palette = set()
      
        actual_output = solutions[task_id][0]
        actual_shape = Shape(len(actual_output), len(actual_output[0]))
        actual_palette = set(np.array(actual_output).flatten())
        
        shape_correct = (predicted_shape == actual_shape)
        color_correct = (predicted_palette == actual_palette)

        if shape_rule_name not in fallback_shape_rules:
            non_fallback_stats['shape_total'] += 1
            if shape_correct:
                non_fallback_stats['shape_correct'] += 1

        if color_rule_name not in fallback_color_rules:
            non_fallback_stats['color_total'] += 1
            if color_correct:
                non_fallback_stats['color_correct'] += 1

        if shape_correct:
            stats['shape_correct'] += 1
            shape_rule_success_counts[shape_rule_name] += 1
        
        if color_correct:
            stats['color_correct'] += 1
            color_rule_success_counts[color_rule_name] += 1
            
        if shape_correct and color_correct:
            stats['total_correct'] += 1

        print_task_analysis(task_id, train_tasks, {
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

        print("\n--- 패턴 매칭 규칙 분석 ---")
        if non_fallback_stats['shape_total'] > 0:
            shape_acc = non_fallback_stats['shape_correct'] / non_fallback_stats['shape_total']
            print(f"Shape 패턴 매칭 정확도: {shape_acc:.2%} ({non_fallback_stats['shape_correct']}/{non_fallback_stats['shape_total']})")
        else:
            print("Shape 패턴 매칭이 발생하지 않았습니다.")

        if non_fallback_stats['color_total'] > 0:
            color_acc = non_fallback_stats['color_correct'] / non_fallback_stats['color_total']
            print(f"Color 패턴 매칭 정확도: {color_acc:.2%} ({non_fallback_stats['color_correct']}/{non_fallback_stats['color_total']})")
        else:
            print("Color 패턴 매칭이 발생하지 않았습니다.")

        print("\n--- 성공한 규칙 분포 ---")
        print("[Shape Rules]")
        if not shape_rule_success_counts:
            print("  (성공한 Shape 예측 없음)")
        else:
            for rule, count in shape_rule_success_counts.most_common():
                print(f"  - {rule}: {count}회")

        print("\n[Color Rules]")
        if not color_rule_success_counts:
            print("  (성공한 Color 예측 없음)")
        else:
            for rule, count in color_rule_success_counts.most_common():
                print(f"  - {rule}: {count}회")

if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent / 'datasets'
    
    print("="*20 + " 훈련 데이터셋 분석 " + "="*20)
    analyze_dataset(
        dataset_path / 'arc-agi_evaluation_challenges.json',
        dataset_path / 'arc-agi_evaluation_solutions.json'
    )