"""API客户端模块，负责与Stable Diffusion WebUI通信"""

import asyncio
import base64
import logging

import aiohttp

logger = logging.getLogger(__name__)


class SDWebUIClient:
    """Stable Diffusion WebUI API客户端"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.session = None
        self._lock = asyncio.Lock()

    async def ensure_session(self):
        """确保会话连接"""
        async with self._lock:
            if self.session is None or self.session.closed:
                timeout = self.config_manager.get_session_timeout()
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(timeout)
                )

    async def close_session(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _call_api(self, endpoint: str, payload: dict) -> dict:
        """通用API调用函数"""
        await self.ensure_session()
        try:
            url = f"{self.config_manager.get_webui_url()}{endpoint}"
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")
                return await resp.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {str(e)}")

    async def check_availability(self) -> tuple[bool, int]:
        """检查服务可用性"""
        try:
            await self.ensure_session()
            url = f"{self.config_manager.get_webui_url()}/sdapi/v1/txt2img"
            async with self.session.get(url) as resp:
                if resp.status == 200 or resp.status == 405:
                    return True, 0
                else:
                    logger.debug(f"⚠️ Stable diffusion Webui 返回值异常，状态码: {resp.status})")
                    return False, resp.status
        except Exception as e:
            logger.debug(f"❌ 测试连接 Stable diffusion Webui 失败，报错：{e}")
            return False, 0

    async def generate_text_to_image(self, prompt: str) -> dict:
        """调用文本到图像生成API"""
        payload = self._build_generation_payload(prompt)
        return await self._call_api("/sdapi/v1/txt2img", payload)

    async def process_image_upscale(self, image_base64: str) -> str:
        """处理图像超分辨率放大"""
        params = self.config_manager.get_default_params()
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        payload = {
            "image": image_base64,
            "upscaling_resize": upscale_factor,
            "upscaler_1": upscaler,
            "resize_mode": 0,
            "show_extras_results": True,
            "upscaling_resize_w": 1,
            "upscaling_resize_h": 1,
            "upscaling_crop": False,
            "gfpgan_visibility": 0,
            "codeformer_visibility": 0,
            "codeformer_weight": 0,
            "extras_upscaler_2_visibility": 0
        }

        resp = await self._call_api("/sdapi/v1/extra-single-image", payload)
        return resp["image"]

    async def set_model(self, model_name: str) -> bool:
        """设置模型"""
        try:
            await self.ensure_session()
            url = f"{self.config_manager.get_webui_url()}/sdapi/v1/options"
            payload = {"sd_model_checkpoint": model_name}

            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.debug(f"模型已设置为: {model_name}")
                    return True
                else:
                    logger.error(f"设置模型失败 (状态码: {resp.status})")
                    return False
        except Exception as e:
            logger.error(f"设置模型异常: {e}")
            return False

    async def fetch_resources(self, resource_type: str) -> list:
        """从WebUI获取指定类型的资源列表"""
        endpoint_map = {
            "model": "/sdapi/v1/sd-models",
            "embedding": "/sdapi/v1/embeddings",
            "lora": "/sdapi/v1/loras",
            "sampler": "/sdapi/v1/samplers",
            "upscaler": "/sdapi/v1/upscalers"
        }

        if resource_type not in endpoint_map:
            logger.error(f"无效的资源类型: {resource_type}")
            return []

        try:
            await self.ensure_session()
            url = f"{self.config_manager.get_webui_url()}{endpoint_map[resource_type]}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    resources = await resp.json()
                    return self._parse_resource_data(resources, resource_type)
        except Exception as e:
            logger.error(f"获取 {resource_type} 类型资源失败: {e}")

        return []

    def _parse_resource_data(self, resources: dict, resource_type: str) -> list:
        """解析不同类型的资源数据"""
        if resource_type == "model":
            return [r["model_name"] for r in resources if "model_name" in r]
        elif resource_type == "embedding":
            return list(resources.get('loaded', {}).keys())
        elif resource_type in ["lora", "sampler", "upscaler"]:
            return [r["name"] for r in resources if "name" in r]
        else:
            return []

    def _build_generation_payload(self, prompt: str) -> dict:
        """构建图像生成参数"""
        params = self.config_manager.get_default_params()

        return {
            "prompt": prompt,
            "negative_prompt": self.config_manager.get_negative_prompt_global(),
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
        }