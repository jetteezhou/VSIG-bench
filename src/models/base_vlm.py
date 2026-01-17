import json
import base64
import os
import time
import logging
from abc import ABC, abstractmethod
import PIL.Image

logger = logging.getLogger("VSIG_Logger")


class BaseVLM(ABC):
    """
    视觉语言模型 (VLM) 的抽象基类。
    """

    @abstractmethod
    def generate(self, image_paths, prompt, system_prompt=None):
        """
        根据图像和提示生成响应。

        Args:
            image_paths (str or list): 图像文件的路径或路径列表。
            prompt (str): 用户输入的提示词。
            system_prompt (str, optional): 系统预设提示词。

        Returns:
            dict: 解析后的 JSON 响应。
        """
        pass

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        """
        根据视频文件和提示生成响应。
        默认实现为抛出错误，子类需覆盖此方法以支持直接视频输入。
        """
        raise NotImplementedError("该模型不支持直接视频文件输入，请使用 generate 方法传入帧列表。")


class OpenAIVLM(BaseVLM):
    """
    使用 OpenAI 兼容 API (如 GPT-4o, vLLM 等) 的模型封装。
    """

    def __init__(self, api_key, base_url=None, model_name="gpt-4o", accepts_video_files=False):
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("缺少 openai 库，请执行 pip install openai")
            raise ImportError("请安装 openai 库: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.accepts_video_files = accepts_video_files  # 是否支持视频文件路径输入
        logger.info(
            f"OpenAIVLM 初始化完成。模型: {model_name}, Base URL: {base_url or 'Default'}, Video Files: {accepts_video_files}")

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _parse_json_response(self, content):
        """
        解析模型返回的 JSON 内容，处理可能的 Markdown 代码块。
        """
        content = content.strip()
        if content.startswith("```"):
            # 去除开头的 ```json 或 ```
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"无法解析模型返回的 JSON: {content}")
            return {}

    def generate(self, image_paths, prompt, system_prompt=None):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(f"正在调用模型处理 {len(image_paths)} 帧图像")

        base64_images = []
        for img_path in image_paths:
            try:
                base64_images.append(self._encode_image(img_path))
            except Exception as e:
                logger.error(f"读取图像失败: {img_path}, 错误: {e}")
                return {}

        messages = []

        # 1. System Prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 2. User Prompt (Text + Images)
        user_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        for b64_img in base64_images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # 强制 JSON 模式，如果是支持该参数的模型
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.2
            )
            result_content = response.choices[0].message.content
            logger.info(f"模型原始响应: {result_content}")
            parsed_result = self._parse_json_response(result_content)
            logger.info("模型推理成功")
            return parsed_result

        except Exception as e:
            logger.error(f"API 调用出错: {e}")
            raise e

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        """
        根据视频文件路径和提示生成响应。
        通过 video_url 字段传递视频文件路径给服务端。
        适用于支持视频文件路径输入的服务（如 HumanOmni）。
        """
        # 转换为绝对路径，确保服务端能找到文件
        abs_video_path = os.path.abspath(video_path)
        logger.info(f"正在调用模型处理视频文件: {abs_video_path}")

        messages = []

        # 1. System Prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 2. User Prompt (Text + Video URL)
        user_content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": f"file://{abs_video_path}"
                }
            }
        ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.2
            )
            result_content = response.choices[0].message.content
            logger.info(f"模型原始响应: {result_content}")
            parsed_result = self._parse_json_response(result_content)
            logger.info("模型推理成功")
            return parsed_result

        except Exception as e:
            logger.error(f"API 调用出错: {e}")
            raise e


class GeminiVLM(BaseVLM):
    """
    使用 Google Gemini API 的模型封装。
    """

    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error(
                "缺少 google-generativeai 库，请执行 pip install google-generativeai")
            raise ImportError(
                "请安装 google-generativeai 库: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.accepts_video_files = True
        logger.info(f"GeminiVLM 初始化完成。模型: {model_name}")

    def _wait_for_file_active(self, file_obj):
        """等待文件处理完成"""
        import google.generativeai as genai
        logger.info(f"等待视频文件处理: {file_obj.name}")
        while file_obj.state.name == "PROCESSING":
            time.sleep(2)
            file_obj = genai.get_file(file_obj.name)

        if file_obj.state.name != "ACTIVE":
            raise ValueError(f"文件处理失败: {file_obj.state.name}")
        logger.info(f"视频文件处理完成: {file_obj.name}")
        return file_obj

    def generate_from_video(self, video_path, prompt, system_prompt=None):
        import google.generativeai as genai
        logger.info(f"正在上传视频至 Gemini: {video_path}")

        try:
            # 1. 上传视频
            video_file = genai.upload_file(video_path)

            # 2. 等待处理
            video_file = self._wait_for_file_active(video_file)

            # 3. 组合 Prompt
            full_prompt = []
            if system_prompt:
                full_prompt.append(system_prompt)

            full_prompt.append(prompt)
            full_prompt.append(video_file)

            # 4. 生成内容
            # Gemini 1.5 支持 json_mode，通过 generation_config 配置
            generation_config = {"response_mime_type": "application/json"}

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            result_content = response.text
            logger.info(f"模型原始响应: {result_content}")
            parsed_result = self._parse_json_response(result_content)

            # 5. 清理（可选，这里为了节省云端空间）
            try:
                genai.delete_file(video_file.name)
                logger.debug(f"已删除云端临时文件: {video_file.name}")
            except Exception as e:
                logger.warning(f"删除云端文件失败: {e}")

            return parsed_result

        except Exception as e:
            logger.error(f"Gemini 视频推理出错: {e}")
            raise e

    def _parse_json_response(self, content):
        """
        解析模型返回的 JSON 内容，处理可能的 Markdown 代码块。
        """
        content = content.strip()
        if content.startswith("```"):
            # 去除开头的 ```json 或 ```
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"无法解析模型返回的 JSON: {content}")
            return {}

    def generate(self, image_paths, prompt, system_prompt=None):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(f"正在调用 Gemini 模型处理 {len(image_paths)} 帧图像")

        images = []
        for img_path in image_paths:
            try:
                images.append(PIL.Image.open(img_path))
            except Exception as e:
                logger.error(f"读取图像失败: {img_path}, 错误: {e}")
                return {}

        # 组合 Prompt
        full_prompt = []
        if system_prompt:
            full_prompt.append(system_prompt)

        full_prompt.append(prompt)
        # 添加所有图像
        full_prompt.extend(images)

        try:
            # Gemini 1.5 支持 json_mode，通过 generation_config 配置
            generation_config = {"response_mime_type": "application/json"}

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            result_content = response.text
            logger.info(f"模型原始响应: {result_content}")
            parsed_result = self._parse_json_response(result_content)
            logger.info("模型推理成功")
            return parsed_result

        except Exception as e:
            logger.error(f"Gemini API 调用出错: {e}")
            raise e
