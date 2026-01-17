# src/prompts/vsig_prompts.py

import json

class VSIGPrompts:
    """
    管理 Visual-Speech Intent Grounding (VSIG) 任务的 Prompt 模板。
    """

    @staticmethod
    def get_system_prompt(task_template=None):
        """
        返回 VSIG 任务的系统级 Prompt。根据不同的任务模板（指令1-6）返回定制化的引导。
        """
        base_prompt = """你是一个具备具身智能（Embodied AI）能力的视觉-语言助手。你的任务是执行“视觉-语音意图定位”（Visual-Speech Intent Grounding, VSIG）。

**任务背景：**
你需要根据第一人称视角（Ego-centric）的视频帧和用户的语音指令，理解用户的潜在意图，并将其转化为具体的行动指令和视觉坐标。

**核心要求：**
1. **纯净 JSON 输出**: 必须直接输出 JSON 字符串，**严禁**使用 Markdown 代码块（如 ```json ... ```），**严禁**包含任何解释性文字或注释。
2. **坐标系**: 使用 [x, y] 格式（x=横坐标/宽, y=纵坐标/高），范围 [0-1000]。(0,0) 为左上角。
3. **内容准确**: 
   - **Target Object**: 用户意图操作的物体。
   - **Spatial Affordance**: 物体放置的目标位置或空间关系。

**JSON 输出模板 (请严格遵守此结构):**
{
    "explicit_command": "在此处填写明确的中文指令",
    "point_list": [
        {
            "type": "target_object",
            "description": "物体的外观描述",
            "point": [500, 500],
            "timestamp": [1000, 2000]
        },
        {
            "type": "spatial_affordance",
            "description": "位置的描述",
            "point": [600, 600],
            "timestamp": [2500, 3000]
        }
    ],
    "reasoning": "一步步的简短推理过程"
}

**字段说明：**
- `explicit_command`: 将用户的意图翻译为完整的执行指令（如“把红色螺丝刀放到桌子上”）。
- `point`: [x, y] 整数坐标，0-1000。
- `timestamp`: [start_ms, end_ms] 整数毫秒。如果指令中未提及该物体（如静默指向），设为 null。
"""
        task_specific_guidance = ""
        if task_template == "指令1":
            task_specific_guidance = "\n**任务场景 (指令1)**：用户无语音，仅用手指向某物体。请识别该物体，输出其描述和坐标。`timestamp` 设为 null。"
        elif task_template == "指令2":
            task_specific_guidance = "\n**任务场景 (指令2)**：用户指向某物体并说类似“把这个拿起来”。你需要：1. 识别被指向的物体（target_object）。2. 结合语音时间戳定位指代词“这个”。"
        elif task_template == "指令3":
            task_specific_guidance = "\n**任务场景 (指令3)**：用户先指物体（这个），再指区域（这里），说“把这个放到这里”。需要输出两个点：1. 物体 (target_object)；2. 放置区域 (spatial_affordance)。"
        elif task_template == "指令4":
            task_specific_guidance = "\n**任务场景 (指令4)**：用户指物体A，再指物体B，说“把这个放到它的右边”。需要输出：1. 物体A (target_object)；2. 物体B旁的放置区域 (spatial_affordance)。"
        elif task_template == "指令5":
            task_specific_guidance = "\n**任务场景 (指令5)**：复合指令。例如“把这个放到它的右边，然后把它拿起来”。需解析出多个步骤涉及的物体和位置。"
        elif task_template == "指令6":
            task_specific_guidance = "\n**任务场景 (指令6)**：多步复合指令。例如“把这个放到它的前面，然后再把这个放到它的后面”。需解析所有涉及的物体和位置坐标。"

        return base_prompt + task_specific_guidance

    @staticmethod
    def get_user_prompt(user_transcript, asr_result=None):
        """
        构建用户输入的 Prompt。
        """
        asr_info = ""
        if asr_result:
            asr_info = f"\n**语音转录详细信息 (ASR)**：{json.dumps(asr_result, ensure_ascii=False)}"
            
        return f"""
**当前视频内容**：[VIDEO/IMAGES]
**用户语音指令**："{user_transcript}"{asr_info}

请分析视频内容和指令，输出符合上述 JSON 格式的结果。
"""
