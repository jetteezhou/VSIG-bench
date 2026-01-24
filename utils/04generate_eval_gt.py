
import json
import string
import os
import sys

# Change this if your data structure is different relative to this script
# Assuming script is in src/utils/
# and data is in data_new/
RELATIVE_DATA_PATH = "../data_new"


def get_upper_label(n):
    if n < 0:
        return ""
    if n < 26:
        return string.ascii_uppercase[n]
    else:
        return get_upper_label(n // 26 - 1) + string.ascii_uppercase[n % 26]


def get_lower_label(n):
    if n < 0:
        return ""
    if n < 26:
        return string.ascii_lowercase[n]
    else:
        return get_lower_label(n // 26 - 1) + string.ascii_lowercase[n % 26]


def get_time(item, key):
    val = item.get(key)
    if val is None:
        return 0
    return val


def process_instruction_1_2(data, instruction_name):
    """
    Logic for Instruction 1 and 2:
    Simple mapping of Object Name -> Upper Case Label.
    """
    counter = 0
    names_with_labels = []
    video_eval_data = {}

    # Instructions 1 and 2 don't deduplicate in the original script.
    # We first collect all labels and names to form object_choices.
    all_objects = []
    for entry in data:
        object_space = entry.get('object_space', [])
        for obj in object_space:
            name = obj.get('name', 'Unknown Name').strip()
            if name:
                label = get_upper_label(counter)
                all_objects.append((name, label))
                counter += 1

    object_choices = [f"{label}. {name}" for name, label in all_objects]
    space_choices = []

    # Reset counter to match labels for each video
    counter = 0
    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])
        
        answers = []
        for obj in object_space:
            name = obj.get('name', 'Unknown Name').strip()
            if not name:
                continue
            
            label = get_upper_label(counter)
            answers.append({
                "choice": label,
                "asr_begin_time": get_time(obj, 'asr_begin_time'),
                "asr_end_time": get_time(obj, 'asr_end_time'),
                "mask": obj.get('mask')
            })
            counter += 1
        
        if answers:
            video_eval_data[video_name] = {
                "object_choices": object_choices,
                "space_choices": space_choices,
                "answer": answers
            }

    return video_eval_data, "simple"


def process_instruction_3(data):
    """
    Logic for Instruction 3:
    Object (Upper) + Space (Lower).
    Deduplicates names.
    """
    unique_objects = {}
    unique_spaces = {}

    obj_counter = 0
    space_counter = 0

    # First pass to get all unique objects and spaces for choices
    for entry in data:
        object_space = entry.get('object_space', [])
        for item in object_space:
            name = item.get('name', '').strip()
            if not name:
                continue
            if item.get('type') == 'object':
                if name not in unique_objects:
                    unique_objects[name] = get_upper_label(obj_counter)
                    obj_counter += 1
            elif item.get('type') == 'space':
                if name not in unique_spaces:
                    unique_spaces[name] = get_lower_label(space_counter)
                    space_counter += 1

    sorted_objects = sorted(unique_objects.items(), key=lambda x: (len(x[1]), x[1]))
    sorted_spaces = sorted(unique_spaces.items(), key=lambda x: (len(x[1]), x[1]))
    
    object_choices = [f"{label}. {name}" for name, label in sorted_objects]
    space_choices = [f"{label}. {name}" for name, label in sorted_spaces]

    video_eval_data = {}
    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        obj_item = next((item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()), None)
        space_item = next((item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()), None)

        if not obj_item or not space_item:
            continue

        obj_name = obj_item.get('name', '').strip()
        space_name = space_item.get('name', '').strip()
        
        obj_label = unique_objects[obj_name]
        space_label = unique_spaces[space_name]

        answers = [
            {
                "choice": obj_label,
                "asr_begin_time": get_time(obj_item, 'asr_begin_time'),
                "asr_end_time": get_time(obj_item, 'asr_end_time'),
                "mask": obj_item.get('mask')
            },
            {
                "choice": space_label,
                "asr_begin_time": get_time(space_item, 'asr_begin_time'),
                "asr_end_time": get_time(space_item, 'asr_end_time'),
                "points": space_item.get('points')
            }
        ]

        video_eval_data[video_name] = {
            "object_choices": object_choices,
            "space_choices": space_choices,
            "answer": answers
        }

    return video_eval_data, "complex_3"


def process_instruction_4(data):
    """
    Logic for Instruction 4:
    Object 1 (Upper), Object 2 + Space (Lower, Combined).
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0

    for entry in data:
        object_space = entry.get('object_space', [])
        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 2 or len(spaces) < 1:
            continue

        name1 = objects[0].get('name', '').strip()
        if name1 not in unique_objects:
            unique_objects[name1] = get_upper_label(obj_counter)
            obj_counter += 1

        name2 = objects[1].get('name', '').strip()
        name_space = spaces[0].get('name', '').strip()
        combined_name = f"{name2}{name_space}"
        if combined_name not in unique_combined:
            unique_combined[combined_name] = get_lower_label(combined_counter)
            combined_counter += 1

    sorted_objects = sorted(unique_objects.items(), key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(), key=lambda x: (len(x[1]), x[1]))
    
    object_choices = [f"{label}. {name}" for name, label in sorted_objects]
    space_choices = [f"{label}. {name}" for name, label in sorted_combined]

    video_eval_data = {}
    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 2 or len(spaces) < 1:
            continue

        name1 = objects[0].get('name', '').strip()
        name2 = objects[1].get('name', '').strip()
        name_space = spaces[0].get('name', '').strip()
        combined_name = f"{name2}{name_space}"

        label1 = unique_objects[name1]
        label2 = unique_combined[combined_name]

        # Merge ASR times for combined label
        combined_start = min(get_time(objects[1], 'asr_begin_time'), get_time(spaces[0], 'asr_begin_time'))
        combined_end = max(get_time(objects[1], 'asr_end_time'), get_time(spaces[0], 'asr_end_time'))

        answers = [
            {
                "choice": label1,
                "asr_begin_time": get_time(objects[0], 'asr_begin_time'),
                "asr_end_time": get_time(objects[0], 'asr_end_time'),
                "mask": objects[0].get('mask')
            },
            {
                "choice": label2,
                "asr_begin_time": combined_start,
                "asr_end_time": combined_end,
                "points": spaces[0].get('points')
            }
        ]

        video_eval_data[video_name] = {
            "object_choices": object_choices,
            "space_choices": space_choices,
            "answer": answers
        }

    return video_eval_data, "complex_4"


def process_instruction_5(data):
    """
    Instruction 5:
    Obj 1 (Upper), [Obj 2 + Space] (Lower), Obj 3 (Upper).
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0

    for entry in data:
        object_space = entry.get('object_space', [])
        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 3 or len(spaces) < 1:
            continue

        name1 = objects[0].get('name', '').strip()
        if name1 not in unique_objects:
            unique_objects[name1] = get_upper_label(obj_counter)
            obj_counter += 1

        name2 = objects[1].get('name', '').strip()
        name_space = spaces[0].get('name', '').strip()
        combined_name = f"{name2}{name_space}"
        if combined_name not in unique_combined:
            unique_combined[combined_name] = get_lower_label(combined_counter)
            combined_counter += 1

        name3 = objects[2].get('name', '').strip()
        if name3 not in unique_objects:
            unique_objects[name3] = get_upper_label(obj_counter)
            obj_counter += 1

    sorted_objects = sorted(unique_objects.items(), key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(), key=lambda x: (len(x[1]), x[1]))
    
    object_choices = [f"{label}. {name}" for name, label in sorted_objects]
    space_choices = [f"{label}. {name}" for name, label in sorted_combined]

    video_eval_data = {}
    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 3 or len(spaces) < 1:
            continue

        name1 = objects[0].get('name', '').strip()
        name2 = objects[1].get('name', '').strip()
        name_space = spaces[0].get('name', '').strip()
        combined_name = f"{name2}{name_space}"
        name3 = objects[2].get('name', '').strip()

        label1 = unique_objects[name1]
        label2 = unique_combined[combined_name]
        label3 = unique_objects[name3]

        combined_start = min(get_time(objects[1], 'asr_begin_time'), get_time(spaces[0], 'asr_begin_time'))
        combined_end = max(get_time(objects[1], 'asr_end_time'), get_time(spaces[0], 'asr_end_time'))

        answers = [
            {"choice": label1, "asr_begin_time": get_time(objects[0], 'asr_begin_time'), "asr_end_time": get_time(objects[0], 'asr_end_time'), "mask": objects[0].get('mask')},
            {"choice": label2, "asr_begin_time": combined_start, "asr_end_time": combined_end, "points": spaces[0].get('points')},
            {"choice": label3, "asr_begin_time": get_time(objects[2], 'asr_begin_time'), "asr_end_time": get_time(objects[2], 'asr_end_time'), "mask": objects[2].get('mask')}
        ]

        video_eval_data[video_name] = {
            "object_choices": object_choices,
            "space_choices": space_choices,
            "answer": answers
        }

    return video_eval_data, "complex_5"


def process_instruction_6(data):
    """
    Instruction 6:
    Obj1(U), [Obj2+Space1](L), Obj3(U), [Obj4+Space2](L)
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0

    for entry in data:
        object_space = entry.get('object_space', [])
        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 4 or len(spaces) < 2:
            continue

        for i in [0, 2]:
            name = objects[i].get('name', '').strip()
            if name not in unique_objects:
                unique_objects[name] = get_upper_label(obj_counter)
                obj_counter += 1

        for i, j in [(1, 0), (3, 1)]:
            name = objects[i].get('name', '').strip()
            space = spaces[j].get('name', '').strip()
            combined_name = f"{name}{space}"
            if combined_name not in unique_combined:
                unique_combined[combined_name] = get_lower_label(combined_counter)
                combined_counter += 1

    sorted_objects = sorted(unique_objects.items(), key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(), key=lambda x: (len(x[1]), x[1]))
    
    object_choices = [f"{label}. {name}" for name, label in sorted_objects]
    space_choices = [f"{label}. {name}" for name, label in sorted_combined]

    video_eval_data = {}
    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get('type') == 'object' and item.get('name', '').strip()]
        spaces = [item for item in object_space if item.get('type') == 'space' and item.get('name', '').strip()]

        if len(objects) < 4 or len(spaces) < 2:
            continue

        labels = []
        for i in [0, 2]:
            labels.append(unique_objects[objects[i].get('name', '').strip()])
        
        combined_labels = []
        combined_asr = []
        for i, j in [(1, 0), (3, 1)]:
            combined_name = f"{objects[i].get('name', '').strip()}{spaces[j].get('name', '').strip()}"
            combined_labels.append(unique_combined[combined_name])
            start = min(get_time(objects[i], 'asr_begin_time'), get_time(spaces[j], 'asr_begin_time'))
            end = max(get_time(objects[i], 'asr_end_time'), get_time(spaces[j], 'asr_end_time'))
            combined_asr.append((start, end))

        answers = [
            {"choice": labels[0], "asr_begin_time": get_time(objects[0], 'asr_begin_time'), "asr_end_time": get_time(objects[0], 'asr_end_time'), "mask": objects[0].get('mask')},
            {"choice": combined_labels[0], "asr_begin_time": combined_asr[0][0], "asr_end_time": combined_asr[0][1], "points": spaces[0].get('points')},
            {"choice": labels[1], "asr_begin_time": get_time(objects[2], 'asr_begin_time'), "asr_end_time": get_time(objects[2], 'asr_end_time'), "mask": objects[2].get('mask')},
            {"choice": combined_labels[1], "asr_begin_time": combined_asr[1][0], "asr_end_time": combined_asr[1][1], "points": spaces[1].get('points')}
        ]

        video_eval_data[video_name] = {
            "object_choices": object_choices,
            "space_choices": space_choices,
            "answer": answers
        }

    return video_eval_data, "complex_6"


def write_eval_gt(output_path, video_eval_data):
    # 删除已存在的文件
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(video_eval_data, out_f, ensure_ascii=False, indent=2)


def main():
    # Resolve absolute path for RELATIVE_DATA_PATH based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, RELATIVE_DATA_PATH))

    print(f"Searching for instruction directories in: {data_root}")

    if not os.path.exists(data_root):
        print(f"Error: Data root directory not found: {data_root}")
        return

    # Iterate through all subdirectories in data_new
    for root, dirs, files in os.walk(data_root):
        for dirname in dirs:
            if dirname in ["指令1", "指令2", "指令3", "指令4", "指令5", "指令6"]:
                full_dir_path = os.path.join(root, dirname)
                json_path = os.path.join(full_dir_path, "annotations.json")
                out_path = os.path.join(full_dir_path, "eval_gt.json")

                if not os.path.exists(json_path):
                    continue

                print(f"Processing {dirname} in {full_dir_path}...")

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 数据去重：基于 id 字段（保留第一次出现）
                    seen_ids = set()
                    deduplicated_data = []
                    for item in data:
                        item_id = item.get("id")
                        if item_id is None:
                            deduplicated_data.append(item)
                        elif item_id not in seen_ids:
                            seen_ids.add(item_id)
                            deduplicated_data.append(item)
                    data = deduplicated_data

                    if dirname in ["指令1", "指令2"]:
                        video_eval_data, mode = process_instruction_1_2(data, dirname)
                        write_eval_gt(out_path, video_eval_data)
                    elif dirname == "指令3":
                        video_eval_data, mode = process_instruction_3(data)
                        write_eval_gt(out_path, video_eval_data)
                    elif dirname == "指令4":
                        video_eval_data, mode = process_instruction_4(data)
                        write_eval_gt(out_path, video_eval_data)
                    elif dirname == "指令5":
                        video_eval_data, mode = process_instruction_5(data)
                        write_eval_gt(out_path, video_eval_data)
                    elif dirname == "指令6":
                        video_eval_data, mode = process_instruction_6(data)
                        write_eval_gt(out_path, video_eval_data)

                    print(f"  -> Generated {out_path}")
                except Exception as e:
                    print(f"  -> Error processing {json_path}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
