"""命令处理模块，处理各种sd命令"""

import logging

from .config_manager import ConfigManager
from .image_processor import ImageProcessor
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class CommandHandlers:
    """命令处理器"""

    def __init__(self, config_manager: ConfigManager, image_processor: ImageProcessor, resource_manager: ResourceManager):
        self.config_manager = config_manager
        self.image_processor = image_processor
        self.resource_manager = resource_manager

    # 基础命令处理
    async def handle_check(self, event):
        """处理检查命令"""
        try:
            available, status = await self.image_processor.api_client.check_availability()
            if available:
                yield event.plain_result("✅ 同Webui连接正常")
            else:
                yield event.plain_result("❌ 同Webui无连接，请检查配置和Webui工作状态")
        except Exception as e:
            logger.error(f"❌ 检查可用性错误，报错{e}")
            yield event.plain_result("❌ 检查可用性错误，请检查日志")

    async def handle_gen(self, event, prompt: str):
        """处理图像生成命令"""
        async for result in self.image_processor.generate_image_with_semaphore(event, prompt):
            yield result

    async def handle_verbose(self, event):
        """处理详细模式切换命令"""
        try:
            current_verbose = self.config_manager.get_verbose_mode()
            new_verbose = not current_verbose
            self.config_manager.update_config("verbose", new_verbose)
            status = "开启" if new_verbose else "关闭"
            yield event.plain_result(f"📢 详细输出模式已{status}")
        except Exception as e:
            logger.error(f"切换详细输出模式失败: {e}")
            yield event.plain_result("❌ 切换详细模式失败，请检查日志")

    async def handle_upscale(self, event):
        """处理图像增强模式切换命令"""
        try:
            current_upscale = self.config_manager.get_upscale_enabled()
            new_upscale = not current_upscale
            self.config_manager.update_config("enable_upscale", new_upscale)
            status = "开启" if new_upscale else "关闭"
            yield event.plain_result(f"📢 图像增强模式已{status}")
        except Exception as e:
            logger.error(f"切换图像增强模式失败: {e}")
            yield event.plain_result("❌ 切换图像增强模式失败，请检查日志")

    async def handle_llm_mode(self, event):
        """处理LLM模式切换命令"""
        try:
            current_setting = self.config_manager.get_generate_prompt_enabled()
            new_setting = not current_setting
            self.config_manager.update_config("enable_generate_prompt", new_setting)
            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 提示词生成功能已{status}")
        except Exception as e:
            logger.error(f"切换生成提示词功能失败: {e}")
            yield event.plain_result("❌ 切换生成提示词功能失败，请检查日志")

    async def handle_show_prompt(self, event):
        """处理显示提示词切换命令"""
        try:
            current_setting = self.config_manager.get_show_positive_prompt()
            new_setting = not current_setting
            self.config_manager.update_config("enable_show_positive_prompt", new_setting)
            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 显示正向提示词功能已{status}")
        except Exception as e:
            logger.error(f"切换显示正向提示词功能失败: {e}")
            yield event.plain_result("❌ 切换显示正向提示词功能失败，请检查日志")

    async def handle_timeout(self, event, time: int):
        """处理超时设置命令"""
        try:
            if time < 10 or time > 300:
                yield event.plain_result("⚠️ 超时时间需设置在 10 到 300 秒范围内")
                return
            self.config_manager.update_config("session_timeout_time", time)
            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查日志")

    async def handle_conf(self, event):
        """处理配置显示命令"""
        try:
            gen_params = self.config_manager.get_generation_params()
            scale_params = self.config_manager.get_upscale_params()
            prompt_guidelines = self.config_manager.get_prompt_guidelines().strip() or "未设置"

            verbose = self.config_manager.get_verbose_mode()
            upscale = self.config_manager.get_upscale_enabled()
            show_positive_prompt = self.config_manager.get_show_positive_prompt()
            generate_prompt = self.config_manager.get_generate_prompt_enabled()

            conf_message = (
                f"⚙️  图像生成参数:\n{gen_params}\n\n"
                f"🔍  图像增强参数:\n{scale_params}\n\n"
                f"🛠️  提示词附加要求: {prompt_guidelines}\n\n"
                f"📢  详细输出模式: {'开启' if verbose else '关闭'}\n\n"
                f"🔧  图像增强模式: {'开启' if upscale else '关闭'}\n\n"
                f"📝  正向提示词显示: {'开启' if show_positive_prompt else '关闭'}\n\n"
                f"🤖  提示词生成模式: {'开启' if generate_prompt else '关闭'}"
            )

            yield event.plain_result(conf_message)
        except Exception as e:
            logger.error(f"获取生成参数失败: {e}")
            yield event.plain_result("❌ 获取图像生成参数失败，请检查配置是否正确")

    async def handle_help(self, event):
        """处理帮助命令"""
        help_msg = [
            "🖼️ **Stable Diffusion 插件帮助指南**",
            "该插件用于调用 Stable Diffusion WebUI 的 API 生成图像并管理相关模型资源。",
            "",
            "📜 **主要功能指令**:",
            "- `/sd gen [提示词]`：生成图片，例如 `/sd gen 星空下的城堡`。",
            "- `/sd check`：检查 WebUI 的连接状态。",
            "- `/sd conf`：显示当前使用配置，包括模型、参数和提示词设置。",
            "- `/sd help`：显示本帮助信息。",
            "",
            "🔧 **高级功能指令**:",
            "- `/sd verbose`：切换详细输出模式，用于实时告知目前AI生图进行到了哪个阶段。",
            "- `/sd upscale`：切换图像增强模式（用于超分辨率放大或高分修复）。",
            "- `/sd LLM`：在使用/sd gen指令时，将内容先发送给LLM，再由LLM来生成正向提示词",
            "- `/sd prompt`：开启时，用户发起AI生图请求后，将发送一条消息，内容为送入到Stable diffusion的正向提示词",
            "- `/sd timeout [秒数]`：设置连接超时时间（建议范围：10 到 300 秒）。",
            "- `/sd res  [宽度] [高度]`：设置图像生成的分辨率（高度和宽度均支持:1-2048之间的任意整数）。",
            "- `/sd step [步数]`：设置图像生成的步数（范围：10 到 50 步）。",
            "- `/sd batch [数量]`：设置发出AI生图请求后，每轮生成的图片数量（范围： 1 到 10 张）。"
            "- `/sd iter [次数]`：设置迭代次数（范围： 1 到 5 次）。",
            "",
            "🖼️ **基本模型与微调模型指令**:",
            "- `/sd model list`：列出 WebUI 当前可用的模型。",
            "- `/sd model set [索引]`：利用索引设置模型，索引可通过 `model list` 查询。",
            "- `/sd lora`：列出所有可用的 LoRA 模型。",
            "- `/sd embedding`：显示所有已加载的 Embedding 模型。",
            "",
            "🎨 **采样器与上采样算法指令**:",
            "- `/sd sampler list`：列出支持的采样器。",
            "- `/sd sampler set [索引]`：根据索引配置采样器，用于调整生成效果。",
            "- `/sd upscaler list`：列出支持的上采样算法。",
            "- `/sd upscaler set [索引]`：根据索引设置上采样算法。",
            "",
            "ℹ️ **注意事项**:",
            "- 如启用自动生成提示词功能，则会使用 LLM 利用提供的内容来生成提示词。",
            "- 如未启用自动生成提示词功能，若提供的自定义提示词中包含空格，则应使用 "~"（英文波浪号） 替代所有提示词中的空格，否则输入的自定义提示词组将在空格处中断。你可以在配置中修改想使用的字符。",
            "- 模型、采样器和其他资源的索引需要使用对应 `list` 命令获取后设置！",
        ]
        yield event.plain_result("\n".join(help_msg))

    # 参数设置命令
    async def handle_resolution(self, event, width: int, height: int):
        """处理分辨率设置命令"""
        try:
            if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1 or height > 2048 or width > 2048:
                yield event.plain_result("⚠️ 分辨率仅支持:1-2048之间的任意整数")
                return

            self.config_manager.update_default_param("height", height)
            self.config_manager.update_default_param("width", width)
            yield event.plain_result(f"✅ 图像生成的分辨率已设置为: 宽度——{width}，高度——{height}")
        except Exception as e:
            logger.error(f"设置分辨率失败: {e}")
            yield event.plain_result("❌ 设置分辨率失败，请检查日志")

    async def handle_step(self, event, step: int):
        """处理步数设置命令"""
        try:
            if step < 10 or step > 50:
                yield event.plain_result("⚠️ 步数需设置在 10 到 50 之间")
                return
            self.config_manager.update_default_param("steps", step)
            yield event.plain_result(f"✅ 步数已设置为: {step}")
        except Exception as e:
            logger.error(f"设置步数失败: {e}")
            yield event.plain_result("❌ 设置步数失败，请检查日志")

    async def handle_batch_size(self, event, batch_size: int):
        """处理批量大小设置命令"""
        try:
            if batch_size < 1 or batch_size > 10:
                yield event.plain_result("⚠️ 图片生成的批数量需设置在 1 到 10 之间")
                return
            self.config_manager.update_default_param("batch_size", batch_size)
            yield event.plain_result(f"✅ 图片生成批数量已设置为: {batch_size}")
        except Exception as e:
            logger.error(f"设置批量生成数量失败: {e}")
            yield event.plain_result("❌ 设置图片生成批数量失败，请检查日志")

    async def handle_n_iter(self, event, n_iter: int):
        """处理迭代次数设置命令"""
        try:
            if n_iter < 1 or n_iter > 5:
                yield event.plain_result("⚠️ 图片生成的迭代次数需设置在 1 到 5 之间")
                return
            self.config_manager.update_default_param("n_iter", n_iter)
            yield event.plain_result(f"✅ 图片生成的迭代次数已设置为: {n_iter}")
        except Exception as e:
            logger.error(f"设置生成迭代次数失败: {e}")
            yield event.plain_result("❌ 设置图片生成的迭代次数失败，请检查日志")

    # 资源管理命令
    async def handle_model_list(self, event):
        """处理模型列表命令"""
        try:
            models = await self.resource_manager.get_model_list()
            yield event.plain_result(self.resource_manager.format_resource_list(models, "模型"))
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            yield event.plain_result("❌ 获取模型列表失败，请检查 WebUI 是否运行")

    async def handle_model_set(self, event, model_index: int):
        """处理模型设置命令"""
        try:
            valid, model_name, error_msg = await self.resource_manager.validate_model_index(model_index)
            if not valid:
                yield event.plain_result(error_msg)
                return

            if await self.resource_manager.set_model(model_name):
                yield event.plain_result(f"✅ 模型已切换为: {model_name}")
            else:
                yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")
        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请检查日志")

    async def handle_lora_list(self, event):
        """处理LoRA列表命令"""
        try:
            lora_models = await self.resource_manager.get_lora_list()
            if not lora_models:
                yield event.plain_result("没有可用的 LoRA 模型。")
            else:
                yield event.plain_result(self.resource_manager.format_resource_list(lora_models, "LoRA 模型"))
        except Exception as e:
            yield event.plain_result(f"获取 LoRA 模型列表失败: {str(e)}")

    async def handle_embedding_list(self, event):
        """处理Embedding列表命令"""
        try:
            embedding_models = await self.resource_manager.get_embedding_list()
            if not embedding_models:
                yield event.plain_result("没有可用的 Embedding 模型。")
            else:
                yield event.plain_result(self.resource_manager.format_resource_list(embedding_models, "Embedding 模型"))
        except Exception as e:
            yield event.plain_result(f"获取 Embedding 模型列表失败: {str(e)}")

    # 采样器命令
    async def handle_sampler_list(self, event):
        """处理采样器列表命令"""
        try:
            samplers = await self.resource_manager.get_sampler_list()
            yield event.plain_result(self.resource_manager.format_resource_list(samplers, "采样器"))
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {str(e)}")

    async def handle_sampler_set(self, event, sampler_index: int):
        """处理采样器设置命令"""
        try:
            valid, sampler_name, error_msg = await self.resource_manager.validate_sampler_index(sampler_index)
            if not valid:
                yield event.plain_result(error_msg)
                return

            self.config_manager.update_default_param("sampler", sampler_name)
            yield event.plain_result(f"✅ 已设置采样器为: {sampler_name}")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    # 上采样算法命令
    async def handle_upscaler_list(self, event):
        """处理上采样算法列表命令"""
        try:
            upscalers = await self.resource_manager.get_upscaler_list()
            yield event.plain_result(self.resource_manager.format_resource_list(upscalers, "上采样算法"))
        except Exception as e:
            yield event.plain_result(f"获取上采样算法列表失败: {str(e)}")

    async def handle_upscaler_set(self, event, upscaler_index: int):
        """处理上采样算法设置命令"""
        try:
            valid, upscaler_name, error_msg = await self.resource_manager.validate_upscaler_index(upscaler_index)
            if not valid:
                yield event.plain_result(error_msg)
                return

            self.config_manager.update_default_param("upscaler", upscaler_name)
            yield event.plain_result(f"✅ 已设置上采样算法为: {upscaler_name}")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")