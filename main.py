import os
import logging

from astrbot.api.all import *

from . import ConfigManager, SDWebUIClient, ResourceManager, ImageProcessor, CommandHandlers, LLMTools

logger = logging.getLogger(__name__)
TEMP_PATH = os.path.abspath("data/temp")


@register("SDGen", "buding(AstrBot)", "Stable Diffusion图像生成器", "1.1.2")
class SDGenerator(Star):
    """Stable Diffusion图像生成插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        os.makedirs(TEMP_PATH, exist_ok=True)

        # 初始化各个模块
        self.config_manager = ConfigManager(config)
        self.api_client = SDWebUIClient(self.config_manager)
        self.resource_manager = ResourceManager(self.api_client, self.config_manager)
        self.image_processor = ImageProcessor(self.api_client, self.config_manager)
        self.command_handlers = CommandHandlers(self.config_manager, self.image_processor, self.resource_manager)
        self.llm_tools = LLMTools(context, self.config_manager)

        # 设置最大并发任务数
        max_tasks = config.get("max_concurrent_tasks", 10)
        self.image_processor.set_max_concurrent_tasks(max_tasks)

        # 配置验证
        self.config_manager.validate_config()

        # 为图像处理器注入LLM提示词生成功能
        self.image_processor._generate_prompt_with_llm = self.llm_tools.generate_prompt

        # 为LLM工具注入图像生成功能
        self.llm_tools.llm_tool_generate_image = self._llm_tool_generate_image

    async def terminate(self):
        """插件终止时清理资源"""
        if self.api_client:
            await self.api_client.close_session()

    # 基础命令组
    @command_group("sd")
    def sd(self):
        pass

    # 基础功能命令
    @sd.command("check")
    async def check(self, event: AstrMessageEvent):
        """服务状态检查"""
        async for result in self.command_handlers.handle_check(event):
            yield result

    @sd.command("gen")
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令"""
        async for result in self.command_handlers.handle_gen(event, prompt):
            yield result

    @sd.command("verbose")
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式"""
        async for result in self.command_handlers.handle_verbose(event):
            yield result

    @sd.command("upscale")
    async def set_upscale(self, event: AstrMessageEvent):
        """设置图像增强模式"""
        async for result in self.command_handlers.handle_upscale(event):
            yield result

    @sd.command("LLM")
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        async for result in self.command_handlers.handle_llm_mode(event):
            yield result

    @sd.command("prompt")
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正向提示词功能"""
        async for result in self.command_handlers.handle_show_prompt(event):
            yield result

    @sd.command("timeout")
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        async for result in self.command_handlers.handle_timeout(event, time):
            yield result

    @sd.command("conf")
    async def show_conf(self, event: AstrMessageEvent):
        """显示当前配置"""
        async for result in self.command_handlers.handle_conf(event):
            yield result

    @sd.command("help")
    async def show_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        async for result in self.command_handlers.handle_help(event):
            yield result

    # 参数设置命令
    @sd.command("res")
    async def set_resolution(self, event: AstrMessageEvent, width: int, height: int):
        """设置分辨率"""
        async for result in self.command_handlers.handle_resolution(event, width, height):
            yield result

    @sd.command("step")
    async def set_step(self, event: AstrMessageEvent, step: int):
        """设置步数"""
        async for result in self.command_handlers.handle_step(event, step):
            yield result

    @sd.command("batch")
    async def set_batch_size(self, event: AstrMessageEvent, batch_size: int):
        """设置批量大小"""
        async for result in self.command_handlers.handle_batch_size(event, batch_size):
            yield result

    @sd.command("iter")
    async def set_n_iter(self, event: AstrMessageEvent, n_iter: int):
        """设置迭代次数"""
        async for result in self.command_handlers.handle_n_iter(event, n_iter):
            yield result

    # 模型管理命令组
    @sd.group("model")
    def model(self):
        pass

    @model.command("list")
    async def list_model(self, event: AstrMessageEvent):
        """列出可用模型"""
        async for result in self.command_handlers.handle_model_list(event):
            yield result

    @model.command("set")
    async def set_base_model(self, event: AstrMessageEvent, model_index: int):
        """设置基础模型"""
        async for result in self.command_handlers.handle_model_set(event, model_index):
            yield result

    # LoRA和Embedding命令
    @sd.command("lora")
    async def list_lora(self, event: AstrMessageEvent):
        """列出LoRA模型"""
        async for result in self.command_handlers.handle_lora_list(event):
            yield result

    @sd.command("embedding")
    async def list_embedding(self, event: AstrMessageEvent):
        """列出Embedding模型"""
        async for result in self.command_handlers.handle_embedding_list(event):
            yield result

    # 采样器命令组
    @sd.group("sampler")
    def sampler(self):
        pass

    @sampler.command("list")
    async def list_sampler(self, event: AstrMessageEvent):
        """列出采样器"""
        async for result in self.command_handlers.handle_sampler_list(event):
            yield result

    @sampler.command("set")
    async def set_sampler(self, event: AstrMessageEvent, sampler_index: int):
        """设置采样器"""
        async for result in self.command_handlers.handle_sampler_set(event, sampler_index):
            yield result

    # 上采样算法命令组
    @sd.group("upscaler")
    def upscaler(self):
        pass

    @upscaler.command("list")
    async def list_upscaler(self, event: AstrMessageEvent):
        """列出上采样算法"""
        async for result in self.command_handlers.handle_upscaler_list(event):
            yield result

    @upscaler.command("set")
    async def set_upscaler(self, event: AstrMessageEvent, upscaler_index: int):
        """设置上采样算法"""
        async for result in self.command_handlers.handle_upscaler_set(event, upscaler_index):
            yield result

    # LLM工具接口
    @llm_tool("generate_image")
    async def _llm_tool_generate_image(self, event: AstrMessageEvent, prompt: str):
        """LLM工具：根据提示词生成图像"""
        try:
            async for result in self.image_processor.generate_image_with_semaphore(event, prompt):
                yield result
        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            yield event.plain_result("❌ 图像生成失败，请检查日志")