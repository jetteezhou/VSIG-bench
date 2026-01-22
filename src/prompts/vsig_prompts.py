# src/prompts/vsig_prompts.py

import json


class VSIGPrompts:
    """
    管理 Visual-Speech Intent Grounding (VSIG) 任务的 Prompt 模板。
    """

    @staticmethod
    def get_system_prompt(task_template=None, coord_order="xy", options_text=""):
        """
        返回 VSIG 任务的系统级 Prompt。根据不同的任务模板（指令1-6）返回定制化的引导。

        Args:
            task_template: 任务模板（指令1-6）
            coord_order: 坐标顺序，"xy" 表示 [x, y]（默认），"yx" 表示 [y, x]
            options_text: 选项定义文本（从description.txt解析）
        """
        # 根据坐标顺序动态生成坐标说明
        if coord_order == "yx":
            coord_desc = "使用 [y, x] 格式（y=纵坐标/高, x=横坐标/宽），范围 [0-1000]。(0,0) 为左上角。"
            coord_field_desc = "- `point`: [y, x] 整数坐标，0-1000。"
        else:  # coord_order == "xy" (默认)
            coord_desc = "使用 [x, y] 格式（x=横坐标/宽, y=纵坐标/高），范围 [0-1000]。(0,0) 为左上角。"
            coord_field_desc = "- `point`: [x, y] 整数坐标，0-1000。"

        base_prompt = f"""你是一个具备具身智能（Embodied AI）能力的视觉-语言助手。你的任务是执行"视觉-语音意图定位"（Visual-Speech Intent Grounding, VSIG）。

**任务背景：**
你需要根据第一人称视角（Ego-centric）的视频帧和用户的语音指令，理解用户的潜在意图，并将其转化为具体的行动指令、选项选择和视觉坐标。

**选项列表（用于选择题）：**
{options_text}

**核心要求：**
1. **纯净 JSON 输出**: 必须直接输出 JSON 字符串，**严禁**使用 Markdown 代码块（如 ```json ... ```），**严禁**包含任何解释性文字或注释。
2. **坐标系**: {coord_desc}
3. **point参考图像**: point必须基于最后一帧的物体或放置区域位置给出。
4. **内容准确**:
   - **Target Object**: 用户意图操作的物体。
   - **Spatial Affordance**: 物体放置的目标位置或空间关系。

**JSON 输出模板 (请严格遵守此结构):**
{{
    "explicit_command": "在此处填写明确的中文指令",
    "selected_options": ["A", "b"],
    "point_list": [
        {{
            "type": "target_object",
            "description": "物体的外观描述",
            "point": [500, 500],
            "timestamp": [1000, 2000]
        }},
        {{
            "type": "spatial_affordance",
            "description": "位置的描述",
            "point": [600, 600],
            "timestamp": [2500, 3000]
        }}
    ],
    "reasoning": "一步步的简短推理过程"
}}

**字段说明：**
- `explicit_command`: 将用户的意图翻译为完整的执行指令（如"把红色螺丝刀放到桌子上"）。
- `selected_options`: 一个字符串列表，包含你认为正确的选项标号（如 "A", "b" 等）。顺序应对应指令执行的逻辑顺序（如果涉及多步）。
- `point_list`: 一个列表，包含和selected_options对应的多个点。每个点包含以下字段：
    - `type`: 点的类型，"target_object" 或 "spatial_affordance"。
    - `description`: 点的描述，如物体的描述或放置位置的描述。
    - `point`: {coord_field_desc}。最后一帧图像中的物体或放置区域位置的像素坐标。
    - `timestamp`: [start_ms, end_ms] 整数毫秒。如果指令中未提及该物体（如静默指向），设为 null。
"""
        task_specific_guidance = ""
        if task_template == "指令1":
            task_specific_guidance = "\n**任务场景 (指令1)**：用户无语音，仅用手指向某物体。`timestamp` 设为 null。示例：selected_options: [\"A\"], point_list 包含一个 target_object。"
        elif task_template == "指令2":
            task_specific_guidance = "\n**任务场景 (指令2)**：示例：\"把这个拿起来\"，selected_options: [\"A\"], point_list 包含一个 target_object（带 timestamp）。"
        elif task_template == "指令3":
            task_specific_guidance = "\n**任务场景 (指令3)**：示例：\"把这个放到这里\"，selected_options: [\"A\", \"b\"], point_list 包含一个target_object和一个spatial_affordance。"
        elif task_template == "指令4":
            task_specific_guidance = "\n**任务场景 (指令4)**：示例：\"把这个放到它的右边\"，selected_options: [\"A\", \"b\"], point_list 包含一个target_object和一个spatial_affordance。"
        elif task_template == "指令5":
            task_specific_guidance = "\n**任务场景 (指令5)**：示例：\"把这个放到它的右边，然后把它拿起来\"，selected_options: [\"A\", \"b\", \"C\"], point_list 包含一个target_object、一个spatial_affordance和一个target_object。"
        elif task_template == "指令6":
            task_specific_guidance = "\n**任务场景 (指令6)**：示例：\"把这个放到它的前面，然后再把这个放到它的后面\"，selected_options: [\"A\", \"b\", \"C\", \"d\"], point_list 包含一个target_object、一个spatial_affordance、一个target_object和一个spatial_affordance。"

        full_prompt = base_prompt + task_specific_guidance
        return full_prompt

    @staticmethod
    def get_user_prompt(user_transcript, asr_result=None):
        """
        构建用户输入的 Prompt。
        """
        asr_info = ""
        if asr_result:
            asr_info = f"\n**语音转录详细信息 (ASR)**：{json.dumps(asr_result, ensure_ascii=False)}"

        return "请分析视频内容和指令，输出符合上述 JSON 格式的结果。"
