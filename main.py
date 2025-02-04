from astrbot.api.all import *
from astrbot.api.message_components import Image
import aiohttp
import json
import base64


@register("sd_generator", "YourName", "Stable Diffusion图像生成", "1.0.0")
class SDPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.session = None  # 延迟初始化

    async def ensure_session(self):
        """确保会话初始化"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def on_disable(self):
        """插件禁用时清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _call_sd_api(self, prompt: str) -> dict:
        """调用SD API核心方法"""
        await self.ensure_session()

        payload = {
            "prompt": prompt,
            "negative_prompt": self.config["negative_prompt"],
            "width": self.config["default_width"],
            "height": self.config["default_height"],
            "steps": 20,
            "sampler_name": self.config["sampler"],
            "cfg_scale": self.config["cfg_scale"],
            "override_settings": {
                "sd_model_checkpoint": "model.safetensors"
            }
        }

        try:
            async with self.session.post(
                    f"{self.config['webui_url']}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=300
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API返回错误: {error}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise Exception(f"连接失败: {str(e)}")

    @filter.command("sdgen")
    async def generate_image(self, event: AstrMessageEvent, *, prompt: str):
        """图像生成指令

        Args:
            prompt: 生成图像的描述提示词
        """
        try:
            # 第一阶段：生成开始反馈
            yield event.plain_result("🎨 开始生成图像，预计需要20秒...")

            # 第二阶段：调用API
            response = await self._call_sd_api(prompt)

            # 第三阶段：处理结果
            if not response.get("images"):
                raise Exception("API返回数据异常")

            image_data = response["images"][0]

            # 发送base64图片
            yield event.image_result(f"base64://{image_data}")

            # 可选：发送生成参数
            info = json.loads(response["info"])
            params = [
                f"尺寸: {info['width']}x{info['height']}",
                f"采样器: {info['sampler_name']}",
                f"种子: {info['seed']}"
            ]
            yield event.plain_result("生成参数:\n" + "\n".join(params))

        except Exception as e:
            error_msg = f"⚠️ 生成失败: {str(e)}"
            if "ConnectionError" in str(e):
                error_msg += "\n请检查WebUI地址是否正确且服务已启动"
            yield event.plain_result(error_msg)
