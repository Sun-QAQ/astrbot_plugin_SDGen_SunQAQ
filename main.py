import logging
from typing import Any, Optional, Coroutine

from astrbot.api.all import *
import aiohttp
import json

logger = logging.getLogger("astrbot")


@register("SDGen", "buding", "Stable Diffusion图像生成器", "1.0.1")
class SDGenerator(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.session = None
        self._validate_config()

    def _validate_config(self):
        """配置验证"""
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

    async def ensure_session(self):
        """确保会话连接"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )

    async def on_disable(self):
        """清理资源"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def _generate_payload(self, prompt: str) -> dict:
        """构建生成参数"""
        params = self.config["default_params"]
        return {
            "prompt": prompt,
            "negative_prompt": self.config["negative_prompt"],
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "override_settings": {
                "sd_model_checkpoint": "model.safetensors"
            }
        }

    async def _call_sd_api(self, prompt: str) -> dict:
        """调用SD API"""
        await self.ensure_session()
        payload = await self._generate_payload(prompt)

        try:
            async with self.session.post(
                    f"{self.config['webui_url']}/sdapi/v1/txt2img",
                    json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")

                return await resp.json()

        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {str(e)}")

    @command_group("sd")
    def sd(self):
        pass

    @sd.command("gen")
    async def generate_image(self, event: AstrMessageEvent, prompt_start: str, *args):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """
        prompt = prompt_start.join(args)
        logger.debug(f"prompt: {prompt}")
        try:
            # 第一阶段：生成开始反馈
            yield event.plain_result("🖌️ 正在生成图像，这可能需要1-2分钟...")

            # 第二阶段：API调用
            response = await self._call_sd_api(prompt)

            # 第三阶段：结果处理
            if not response.get("images"):
                raise ValueError("API返回数据异常")

            image_data = response["images"][0]
            logger.debug(f"img: {image_data}")

            info = json.loads(response["info"])
            logger.debug(f"info: {info}")

            image_bytes = base64.b64decode(image_data)

            with open("output.jpg", "wb") as image_file:
                image_file.write(image_bytes)

            # 发送结果
            yield event.image_result("output.jpg")
            yield event.plain_result(
                f"✅ 生成成功\n"
                f"尺寸: {info['width']}x{info['height']}\n"
                f"采样器: {info['sampler_name']}\n"
                f"种子: {info['seed']}"
            )

        except Exception as e:
            logger.error(f"Generate image failed, error: {e}")
            if "Cannot connect to host" in str(e):
                error_msg = "⚠️ 生成失败! 请检查：\n1. WebUI服务是否运行\n2. 防火墙设置\n3. 配置地址是否正确"
            yield event.plain_result(error_msg)

    @sd.command("check")
    async def check_service(self, event: AstrMessageEvent):
        """服务状态检查"""
        try:
            await self.ensure_session()
            async with self.session.get(
                f"{self.config['webui_url']}/sdapi/v1/progress"
            ) as resp:
                if resp.status == 200:
                    yield event.plain_result("✅ 服务连接正常")
                else:
                    yield event.plain_result(f"⚠️ 服务异常 (状态码: {resp.status})")
        except Exception as e:
            yield event.plain_result(f"❌ 连接测试失败: {str(e)}")

    @sd.command("help")
    async def show_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        help_msg = [
            "🖼️ Stable Diffusion 插件使用指南",
            "指令列表:",
            "/sd gen [提示词] - 生成图像（示例：/sdgen 星空下的城堡）",
            "/sd check - 检查服务连接状态",
            "/sd help - 显示本帮助信息",
            "配置参数:",
            f"当前模型: {self.config['default_params']['sampler']}",
            f"默认尺寸: {self.config['default_params']['width']}x{self.config['default_params']['height']}"
        ]
        yield event.plain_result("\n".join(help_msg))
