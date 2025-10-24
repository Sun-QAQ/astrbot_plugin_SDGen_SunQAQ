"""资源管理模块，负责管理模型、采样器等资源"""

import logging

from .api_client import SDWebUIClient
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ResourceManager:
    """资源管理器"""

    def __init__(self, api_client: SDWebUIClient, config_manager: ConfigManager):
        self.api_client = api_client
        self.config_manager = config_manager

    async def get_model_list(self):
        """获取可用的模型列表"""
        return await self.api_client.fetch_resources("model")

    async def get_lora_list(self):
        """获取可用的LoRA模型列表"""
        return await self.api_client.fetch_resources("lora")

    async def get_embedding_list(self):
        """获取已加载的Embedding模型列表"""
        return await self.api_client.fetch_resources("embedding")

    async def get_sampler_list(self):
        """获取可用的采样器列表"""
        return await self.api_client.fetch_resources("sampler")

    async def get_upscaler_list(self):
        """获取可用的上采样算法列表"""
        return await self.api_client.fetch_resources("upscaler")

    async def set_model(self, model_name: str) -> bool:
        """设置当前使用的模型"""
        success = await self.api_client.set_model(model_name)
        if success:
            # 更新配置中的模型
            self.config_manager.config["base_model"] = model_name
            self.config_manager.config.save_config()
        return success

    def format_resource_list(self, resources: list, resource_name: str) -> str:
        """格式化资源列表为用户友好的字符串"""
        if not resources:
            return f"⚠️ 没有可用的{resource_name}"

        formatted_list = "\n".join(f"{i + 1}. {resource}" for i, resource in enumerate(resources))
        return f"📋 可用{resource_name}列表:\n{formatted_list}"

    async def validate_model_index(self, model_index: int) -> tuple[bool, str, str]:
        """验证模型索引是否有效"""
        try:
            models = await self.get_model_list()
            if not models:
                return False, "", "⚠️ 没有可用的模型"

            index = int(model_index) - 1  # 转换为 0-based 索引
            if index < 0 or index >= len(models):
                return False, "", "❌ 无效的模型索引，请使用 list 命令获取正确的索引"

            selected_model = models[index]
            return True, selected_model, ""
        except ValueError:
            return False, "", "❌ 请输入有效的数字索引"
        except Exception as e:
            logger.error(f"验证模型索引时出错: {e}")
            return False, "", f"❌ 验证模型索引时出错: {str(e)}"

    async def validate_sampler_index(self, sampler_index: int) -> tuple[bool, str, str]:
        """验证采样器索引是否有效"""
        try:
            samplers = await self.get_sampler_list()
            if not samplers:
                return False, "", "⚠️ 没有可用的采样器"

            index = int(sampler_index) - 1
            if index < 0 or index >= len(samplers):
                return False, "", "❌ 无效的采样器索引，请使用 list 命令获取正确的索引"

            selected_sampler = samplers[index]
            return True, selected_sampler, ""
        except ValueError:
            return False, "", "❌ 请输入有效的数字索引"
        except Exception as e:
            logger.error(f"验证采样器索引时出错: {e}")
            return False, "", f"❌ 验证采样器索引时出错: {str(e)}"

    async def validate_upscaler_index(self, upscaler_index: int) -> tuple[bool, str, str]:
        """验证上采样算法索引是否有效"""
        try:
            upscalers = await self.get_upscaler_list()
            if not upscalers:
                return False, "", "⚠️ 没有可用的上采样算法"

            index = int(upscaler_index) - 1
            if index < 0 or index >= len(upscalers):
                return False, "", "❌ 无效的上采样算法索引，请使用 list 命令获取正确的索引"

            selected_upscaler = upscalers[index]
            return True, selected_upscaler, ""
        except ValueError:
            return False, "", "❌ 请输入有效的数字索引"
        except Exception as e:
            logger.error(f"验证上采样算法索引时出错: {e}")
            return False, "", f"❌ 验证上采样算法索引时出错: {str(e)}"