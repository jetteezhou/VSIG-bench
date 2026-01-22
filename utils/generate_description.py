
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


def process_instruction_1_2(data, instruction_name):
    """
    Logic for Instruction 1 and 2:
    Simple mapping of Object Name -> Upper Case Label.
    """
    counter = 0
    unique_names = {}  # Name -> Label
    # List of (Label, Name) for sorted output order if needed, or just use dict items
    names_with_labels = []
    video_map = []         # (Video Name, Label)

    # Use a dictionary to ensure unique names get the same label
    # The original script for 1/2 didn't explicitly deduplicate names in a dict first,
    # but just appended. However, looking at the logic:
    # "label = get_label(counter); names_with_labels.append..."
    # It assigned a NEW label to every occurrence even if name was same?
    # Let's re-read script 1.py.
    # Script 1.py:
    # for obj in object_space:
    #    label = get_label(counter)
    #    names_with_labels.append((label, name))
    #    counter += 1
    # It does NOT deduplicate. It assigns a unique label to EVERY object instance in EVERY video.
    # Wait, "names_with_labels.append((label, name))".
    # If "Cup" appears in Video 1 and Video 2, it gets two different labels?
    # Yes, based on 1.py logic.
    # Let's stick to the original logic for compatibility unless it looks like a bug.
    # Actually, usually in these tasks, we want global consistency if it's the SAME object instance?
    # But 1.py clearly just increments counter for every object found.
    # "first step" list will be very long if it's not deduplicated.
    # But let's look at 1.py output format:
    # "A. Cup", "B. Cup"...
    # If the user wants to merge logic, I should replicate exactly what provided scripts do.

    # Correction: Script 1.py and 2.py do NOT deduplicate.

    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        for obj in object_space:
            name = obj.get('name', 'Unknown Name')
            label = get_upper_label(counter)

            names_with_labels.append((label, name))
            video_map.append((video_name, label))

            counter += 1

    return names_with_labels, video_map, "simple"


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

    video_map = []  # (Video Name, ObjLabel, SpaceLabel)

    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        current_obj_label = "?"
        current_space_label = "?"

        # Find object
        obj_name = ""
        for item in object_space:
            if item.get('type') == 'object':
                obj_name = item.get('name', '').strip()
                if obj_name:  # 只有当名称不为空时才处理
                    if obj_name in unique_objects:
                        current_obj_label = unique_objects[obj_name]
                    else:
                        label = get_upper_label(obj_counter)
                        unique_objects[obj_name] = label
                        current_obj_label = label
                        obj_counter += 1
                break

        # Find space
        space_name = ""
        for item in object_space:
            if item.get('type') == 'space':
                space_name = item.get('name', '').strip()
                if space_name:  # 只有当名称不为空时才处理
                    if space_name in unique_spaces:
                        current_space_label = unique_spaces[space_name]
                    else:
                        label = get_lower_label(space_counter)
                        unique_spaces[space_name] = label
                        current_space_label = label
                        space_counter += 1
                break

        # 验证：必须同时有object和space标签，且名称都不为空，否则跳过该视频
        if current_obj_label == "?" or current_space_label == "?" or not obj_name or not space_name:
            continue

        video_map.append((video_name, current_obj_label, current_space_label))

    # Prepare definitions
    sorted_objects = sorted(unique_objects.items(),
                            key=lambda x: (len(x[1]), x[1]))
    sorted_spaces = sorted(unique_spaces.items(),
                           key=lambda x: (len(x[1]), x[1]))

    definitions = {
        "objects": sorted_objects,
        "spaces": sorted_spaces
    }
    return definitions, video_map, "complex_3"


def process_instruction_4(data):
    """
    Logic for Instruction 4:
    Object 1 (Upper), Object 2 + Space (Lower, Combined).
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0

    video_map = []  # (Video Name, [ObjLabels], [CombinedLabels])

    for entry in data:
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get(
            'type') == 'object']
        spaces = [item for item in object_space if item.get('type') == 'space']

        current_obj_labels = []
        current_combined_labels = []

        # 1st Object -> Upper
        name1 = ""
        if len(objects) >= 1:
            name1 = objects[0].get('name', '').strip()
            if name1:  # 只有当名称不为空时才处理
                if name1 in unique_objects:
                    label1 = unique_objects[name1]
                else:
                    label1 = get_upper_label(obj_counter)
                    unique_objects[name1] = label1
                    obj_counter += 1
                current_obj_labels.append(label1)

        # 2nd Object + Space -> Lower (Combined)
        name2 = ""
        name_space = ""
        if len(objects) >= 2 and len(spaces) >= 1:
            name2 = objects[1].get('name', '').strip()
            name_space = spaces[0].get('name', '').strip()
            if name2 and name_space:  # 只有当名称都不为空时才处理
                combined_name = f"{name2}{name_space}"

                if combined_name in unique_combined:
                    label2 = unique_combined[combined_name]
                else:
                    label2 = get_lower_label(combined_counter)
                    unique_combined[combined_name] = label2
                    combined_counter += 1
                current_combined_labels.append(label2)

        # 验证：必须同时有object label和combined label，且名称都不为空，否则跳过该视频
        if len(current_obj_labels) == 0 or len(current_combined_labels) == 0 or not name1 or not name2 or not name_space:
            continue

        video_map.append(
            (video_name, current_obj_labels, current_combined_labels))

    sorted_objects = sorted(unique_objects.items(),
                            key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(),
                             key=lambda x: (len(x[1]), x[1]))

    definitions = {
        "objects": sorted_objects,
        "combined": sorted_combined
    }
    return definitions, video_map, "complex_4"


def process_instruction_5(data):
    """
    Instruction 5:
    Obj 1 (Upper), [Obj 2 + Space] (Lower), Obj 3 (Upper).
    Shared Upper pool for Obj 1 and 3.
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0
    video_map = []

    for entry in data:
        if entry.get('is_invalid', False):
            continue

        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get(
            'type') == 'object']
        spaces = [item for item in object_space if item.get('type') == 'space']

        if len(objects) < 3 or len(spaces) < 1:
            continue

        # 1. First Obj
        name1 = objects[0].get('name', '').strip()
        if not name1:
            continue  # 如果名称为空，跳过该视频
        if name1 in unique_objects:
            label1 = unique_objects[name1]
        else:
            label1 = get_upper_label(obj_counter)
            unique_objects[name1] = label1
            obj_counter += 1

        # 2. Obj 2 + Space
        name2 = objects[1].get('name', '').strip()
        name_space = spaces[0].get('name', '').strip()
        if not name2 or not name_space:
            continue  # 如果名称为空，跳过该视频
        combined_name = f"{name2}{name_space}"
        if combined_name in unique_combined:
            label2 = unique_combined[combined_name]
        else:
            label2 = get_lower_label(combined_counter)
            unique_combined[combined_name] = label2
            combined_counter += 1

        # 3. Third Obj
        name3 = objects[2].get('name', '').strip()
        if not name3:
            continue  # 如果名称为空，跳过该视频
        if name3 in unique_objects:
            label3 = unique_objects[name3]
        else:
            label3 = get_upper_label(obj_counter)
            unique_objects[name3] = label3
            obj_counter += 1

        # 验证：必须提取到3个标签（label1, label2, label3），否则跳过该视频
        if not label1 or not label2 or not label3:
            continue

        video_map.append((video_name, label1, label2, label3))

    sorted_objects = sorted(unique_objects.items(),
                            key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(),
                             key=lambda x: (len(x[1]), x[1]))

    definitions = {
        "objects": sorted_objects,
        "combined": sorted_combined
    }
    return definitions, video_map, "complex_5"


def process_instruction_6(data):
    """
    Instruction 6:
    Obj1(U), [Obj2+Space1](L), Obj3(U), [Obj4+Space2](L)
    """
    unique_objects = {}
    unique_combined = {}

    obj_counter = 0
    combined_counter = 0
    video_map = []

    for entry in data:
        if entry.get('is_invalid', False):
            continue
        video_name = entry.get('video_name', 'Unknown Video')
        object_space = entry.get('object_space', [])

        objects = [item for item in object_space if item.get(
            'type') == 'object']
        spaces = [item for item in object_space if item.get('type') == 'space']

        if len(objects) < 4 or len(spaces) < 2:
            continue

        # Component 1: Uppercase
        name1 = objects[0].get('name', '').strip()
        if not name1:
            continue  # 如果名称为空，跳过该视频
        if name1 in unique_objects:
            label1 = unique_objects[name1]
        else:
            label1 = get_upper_label(obj_counter)
            unique_objects[name1] = label1
            obj_counter += 1

        # Component 2: Lowercase (Obj 2 + Space 1)
        name2 = objects[1].get('name', '').strip()
        space1 = spaces[0].get('name', '').strip()
        if not name2 or not space1:
            continue  # 如果名称为空，跳过该视频
        combined_name1 = f"{name2}{space1}"
        if combined_name1 in unique_combined:
            label2 = unique_combined[combined_name1]
        else:
            label2 = get_lower_label(combined_counter)
            unique_combined[combined_name1] = label2
            combined_counter += 1

        # Component 3: Uppercase
        name3 = objects[2].get('name', '').strip()
        if not name3:
            continue  # 如果名称为空，跳过该视频
        if name3 in unique_objects:
            label3 = unique_objects[name3]
        else:
            label3 = get_upper_label(obj_counter)
            unique_objects[name3] = label3
            obj_counter += 1

        # Component 4: Lowercase (Obj 4 + Space 2)
        name4 = objects[3].get('name', '').strip()
        space2 = spaces[1].get('name', '').strip()
        if not name4 or not space2:
            continue  # 如果名称为空，跳过该视频
        combined_name2 = f"{name4}{space2}"
        if combined_name2 in unique_combined:
            label4 = unique_combined[combined_name2]
        else:
            label4 = get_lower_label(combined_counter)
            unique_combined[combined_name2] = label4
            combined_counter += 1

        # 验证：必须提取到4个标签（label1, label2, label3, label4），否则跳过该视频
        if not label1 or not label2 or not label3 or not label4:
            continue

        video_map.append((video_name, label1, label2, label3, label4))

    sorted_objects = sorted(unique_objects.items(),
                            key=lambda x: (len(x[1]), x[1]))
    sorted_combined = sorted(unique_combined.items(),
                             key=lambda x: (len(x[1]), x[1]))

    definitions = {
        "objects": sorted_objects,
        "combined": sorted_combined
    }
    return definitions, video_map, "complex_6"


def write_output(output_path, definitions, video_map, mode):
    # 删除已存在的文件，确保完全重新生成而不是叠加
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        if mode == "simple":
            # Instruction 1 & 2
            out_f.write("第一步：名称与标号\n")
            out_f.write("-" * 50 + "\n")
            # definitions is list of (label, name)
            for label, name in definitions:
                out_f.write(f"{label}. {name}\n")

            out_f.write("\n")
            out_f.write("第二步：视频与标号对应\n")
            out_f.write("-" * 50 + "\n")
            for video, label in video_map:
                out_f.write(f"{video}: {label}\n")

        elif mode == "complex_3":
            # Instruction 3
            out_f.write("第一步：名称与标号 (去重后)\n")
            out_f.write("=" * 50 + "\n")

            out_f.write("物体 (Object) - 大写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["objects"]:
                out_f.write(f"{label}. {name}\n")

            out_f.write("\n")
            out_f.write("空间 (Space) - 小写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["spaces"]:
                out_f.write(f"{label}. {name}\n")

            out_f.write("\n\n")
            out_f.write("第二步：视频与标号对应\n")
            out_f.write("=" * 50 + "\n")
            for video, l1, l2 in video_map:
                out_f.write(f"{video}: {l1}, {l2}\n")

        elif mode == "complex_4":
            # Instruction 4
            out_f.write("第一步：名称与标号 (去重后)\n")
            out_f.write("=" * 50 + "\n")

            out_f.write("物体 (Object) - 大写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["objects"]:
                out_f.write(f"{label}. {name}\n")

            out_f.write("\n")
            out_f.write("物体2 + 空间 (Object 2 + Space) - 小写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["combined"]:
                out_f.write(f"{label}. {name}\n")

            out_f.write("\n\n")
            out_f.write("第二步：视频与标号对应\n")
            out_f.write("=" * 50 + "\n")
            for video, obj_lbls, combined_lbls in video_map:
                objs_str = ", ".join(obj_lbls)
                combined_str = ", ".join(combined_lbls)
                parts = []
                if objs_str:
                    parts.append(objs_str)
                if combined_str:
                    parts.append(combined_str)
                value_str = ", ".join(parts)
                out_f.write(f"{video}: {value_str}\n")

        elif mode == "complex_5":
            # Instruction 5
            out_f.write("第一步：名称与标号 (去重后)\n")
            out_f.write("=" * 50 + "\n")
            out_f.write("物体 (Object 1 & 3) - 大写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["objects"]:
                out_f.write(f"{label}. {name}\n")
            out_f.write("\n")
            out_f.write("物体2 + 空间 (Object 2 + Space) - 小写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["combined"]:
                out_f.write(f"{label}. {name}\n")
            out_f.write("\n\n")
            out_f.write("第二步：视频与标号对应\n")
            out_f.write("=" * 50 + "\n")
            for video, l1, l2, l3 in video_map:
                out_f.write(f"{video}: {l1}, {l2}, {l3}\n")

        elif mode == "complex_6":
            # Instruction 6
            out_f.write("第一步：名称与标号 (去重后)\n")
            out_f.write("=" * 50 + "\n")
            out_f.write("物体 (Object) - 大写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["objects"]:
                out_f.write(f"{label}. {name}\n")
            out_f.write("\n")
            out_f.write("物体 + 空间 (Object + Space) - 小写字母:\n")
            out_f.write("-" * 20 + "\n")
            for name, label in definitions["combined"]:
                out_f.write(f"{label}. {name}\n")
            out_f.write("\n\n")
            out_f.write("第二步：视频与标号对应\n")
            out_f.write("=" * 50 + "\n")
            for video, l1, l2, l3, l4 in video_map:
                out_f.write(f"{video}: {l1}, {l2}, {l3}, {l4}\n")


def main():
    # Resolve absolute path for RELATIVE_DATA_PATH based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, RELATIVE_DATA_PATH))

    print(f"Searching for instruction directories in: {data_root}")

    if not os.path.exists(data_root):
        print(f"Error: Data root directory not found: {data_root}")
        return

    # Iterate through all subdirectories in data_new
    # We look for "指令1" to "指令6" in any subdirectory depth?
    # Based on file paths provided: data_new/四楼实训区.../指令X
    # So we should recursively find any directory named "指令X"

    for root, dirs, files in os.walk(data_root):
        for dirname in dirs:
            if dirname in ["指令1", "指令2", "指令3", "指令4", "指令5", "指令6"]:
                full_dir_path = os.path.join(root, dirname)
                json_path = os.path.join(full_dir_path, "annotations.json")
                out_path = os.path.join(full_dir_path, "description.txt")

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
                        defs, vmap, mode = process_instruction_1_2(
                            data, dirname)
                        write_output(out_path, defs, vmap, mode)
                    elif dirname == "指令3":
                        defs, vmap, mode = process_instruction_3(data)
                        write_output(out_path, defs, vmap, mode)
                    elif dirname == "指令4":
                        defs, vmap, mode = process_instruction_4(data)
                        write_output(out_path, defs, vmap, mode)
                    elif dirname == "指令5":
                        defs, vmap, mode = process_instruction_5(data)
                        write_output(out_path, defs, vmap, mode)
                    elif dirname == "指令6":
                        defs, vmap, mode = process_instruction_6(data)
                        write_output(out_path, defs, vmap, mode)

                    print(f"  -> Generated {out_path}")
                except Exception as e:
                    print(f"  -> Error processing {json_path}: {e}")


if __name__ == "__main__":
    main()
