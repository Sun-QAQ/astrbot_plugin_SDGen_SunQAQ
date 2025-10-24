"""图像处理模块，负责图像生成和处理相关功能"""

import asyncio
import base64
import logging
import re

from .api_client import SDWebUIClient
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器"""

    def __init__(self, api_client: SDWebUIClient, config_manager: ConfigManager):
        self.api_client = api_client
        self.config_manager = config_manager
        self.active_tasks = 0
        self.max_concurrent_tasks = 10  # 默认最大并发数
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

    def set_max_concurrent_tasks(self, max_tasks: int):
        """设置最大并发任务数"""
        self.max_concurrent_tasks = max_tasks
        self.task_semaphore = asyncio.Semaphore(max_tasks)

    async def generate_image_with_semaphore(self, event, prompt: str):
        """使用信号量控制并发的图像生成"""
        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                async for result in self._generate_image(event, prompt):
                    yield result
            finally:
                self.active_tasks -= 1

    async def _generate_image(self, event, prompt: str):
        """核心图像生成逻辑"""
        try:
            # 检查服务可用性
            available, status = await self.api_client.check_availability()
            if not available:
                yield event.plain_result("⚠️ 同webui无连接，目前无法生成图片！")
                return

            verbose = self.config_manager.get_verbose_mode()
            if verbose:
                yield event.plain_result("🖌️ 生成图像阶段，这可能需要一段时间...")

            # 处理提示词
            final_prompt = await self._process_prompt(prompt)

            # 输出正向提示词（如果启用）
            if self.config_manager.get_show_positive_prompt():
                yield event.plain_result(f"正向提示词：{final_prompt}")

            # 生成图像
            response = await self.api_client.generate_text_to_image(final_prompt)
            if not response.get("images"):
                raise ValueError("API返回数据异常：生成图像失败")

            # 处理图像结果
            async for result in self._process_generated_images(event, response["images"], verbose):
                yield result

            if verbose:
                yield event.plain_result("✅ 图像生成成功")

        except ValueError as e:
            logger.error(f"API返回数据异常: {e}")
            yield event.plain_result(f"❌ 图像生成失败: 参数异常，API调用失败")

        except ConnectionError as e:
            logger.error(f"网络连接失败: {e}")
            yield event.plain_result("⚠️ 生成失败! 请检查网络连接和WebUI服务是否运行正常")

        except TimeoutError as e:
            logger.error(f"请求超时: {e}")
            yield event.plain_result("⚠️ 请求超时，请稍后再试")

        except Exception as e:
            logger.error(f"生成图像时发生其他错误: {e}")
            yield event.plain_result(f"❌ 图像生成失败: 发生其他错误，请检查日志")

    async def _process_prompt(self, prompt: str) -> str:
        """处理提示词，包括生成和格式化"""
        if self.config_manager.get_generate_prompt_enabled():
            generated_prompt = await self._generate_prompt_with_llm(prompt)
            logger.debug(f"LLM generated prompt: {generated_prompt}")

            # 添加全局正面提示词
            positive_prompt = self._combine_with_global_positive_prompt(generated_prompt)
            return positive_prompt
        else:
            # 使用用户提供的提示词
            user_prompt = self._trans_prompt(prompt)
            positive_prompt = self._combine_with_global_positive_prompt(user_prompt)
            return positive_prompt

    def _trans_prompt(self, prompt: str) -> str:
        """将提示词中的空格替换字符替换为空格"""
        replace_space = self.config_manager.get_replace_space_char()
        return prompt.replace(replace_space, " ")

    def _combine_with_global_positive_prompt(self, prompt: str) -> str:
        """将提示词与全局正面提示词组合"""
        global_positive = self.config_manager.get_positive_prompt_global()
        add_to_head = self.config_manager.get_positive_prompt_add_position()

        if add_to_head:
            return global_positive + prompt
        else:
            return prompt + global_positive

    async def _generate_prompt_with_llm(self, prompt: str) -> str:
        """使用LLM生成提示词"""
        # 这个方法将在主类中被LLM工具的实际方法替换
        return ""

    async def _process_generated_images(self, event, images: list, verbose: bool):
        """处理生成的图像"""
        upscale_enabled = self.config_manager.get_upscale_enabled()

        if upscale_enabled and verbose:
            yield event.plain_result("🖼️ 处理图像阶段，即将结束...")

        if len(images) == 1:
            # 单张图像处理
            image = await self._process_single_image(images[0], upscale_enabled)
            yield event.chain_result([image])
        else:
            # 多张图像处理
            chain = []
            for image_data in images:
                image = await self._process_single_image(image_data, upscale_enabled)
                chain.append(image)
            yield event.chain_result(chain)

    async def _process_single_image(self, image_data: str, apply_upscale: bool) -> object:
        """处理单张图像"""
        # 解码base64图像数据
        image_bytes = base64.b64decode(image_data)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 应用图像增强（如果启用）
        if apply_upscale:
            image_base64 = await self.api_client.process_image_upscale(image_base64)

        # 返回图像对象（根据AstrBot的API）
        from astrbot.api.all import Image
        return Image.fromBase64(image_base64)

    def get_task_status(self) -> dict:
        """获取当前任务状态"""
        return {
            "active_tasks": self.active_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks
        }