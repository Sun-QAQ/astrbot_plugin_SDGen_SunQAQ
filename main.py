import re
import tempfile

import aiohttp

from astrbot.api.all import *

@register("SDGen", "buding", "Stable Diffusion图像生成器", "1.0.6")
class SDGenerator(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.session = None
        self._validate_config()

    def _validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")

    async def ensure_session(self):
        """确保会话连接"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(self.config.get("session_timeout_time", 120))
            )

    async def _fetch_webui_resource(self, resource_type: str) -> list:
        """从 WebUI API 获取指定类型的资源列表"""
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
            async with self.session.get(f"{self.config['webui_url']}{endpoint_map[resource_type]}") as resp:
                if resp.status == 200:
                    resources = await resp.json()
                    logger.error(resources)

                    # 按不同类型解析返回数据
                    if resource_type == "model":
                        resource_names = [r["model_name"] for r in resources if "model_name" in r]
                    elif resource_type == "embedding":
                        resource_names = list(resources.get('loaded', {}).keys())
                    elif resource_type == "lora":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "sampler":
                        resource_names = [r["name"] for r in resources if "name" in r]
                    elif resource_type == "upscaler":
                        resource_names = [r["name"] for r in resources if "name" in r]

                    else:
                        resource_names = []

                    logger.debug(f"从 WebUI 获取到的{resource_type}资源: {resource_names}")
                    return resource_names
        except Exception as e:
            logger.error(f"获取 {resource_type} 类型资源失败: {e}")

        return []

    async def _get_sd_model_list(self):
        return await self._fetch_webui_resource("model")

    async def _get_embedding_list(self):
        return await self._fetch_webui_resource("embedding")

    async def _get_lora_list(self):
        return await self._fetch_webui_resource("lora")

    async def _get_sampler_list(self):
        """获取可用的采样器列表"""
        return await self._fetch_webui_resource("sampler")

    async def _get_upscaler_list(self):
        """获取可用的上采样算法列表"""
        return await self._fetch_webui_resource("upscaler")

    async def _generate_payload(self, prompt: str) -> dict:
        """构建生成参数"""
        params = self.config["default_params"]

        return {
            "prompt": prompt,
            "negative_prompt": self.config["negative_prompt_global"],
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
        }

    def _trans_prompt(self, prompt: str) -> str:
        """
        替换提示词中的所有下划线为空格
        """
        return prompt.replace("_", " ")

    async def _generate_prompt(self, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            prompt_guidelines = self.config["prompt_guidelines"]
            prompt_generate_text = (
                "请根据以下描述生成用于 Stable Diffusion WebUI 的提示词，"
                "请返回一条逗号分隔的 `prompt` 英文字符串，适用于 SD-WebUI，"
                "其中应包含主体、风格、光照、色彩等方面的描述，"
                "避免解释性文本，不需要 “prompt:” 等内容，不需要双引号包裹，"
                "直接返回 `prompt`，不要加任何额外说明。"
                f"{prompt_guidelines}\n"
                "描述："
            )

            response = await provider.text_chat(f"{prompt_generate_text} {prompt}", session_id=None)
            if response.completion_text:
                generated_prompt = re.sub(r"<think>[\s\S]*</think>", "", response.completion_text).strip()
                return generated_prompt

        return ""

    async def _call_sd_api(self, endpoint: str, payload: dict) -> dict:
        """通用API调用函数"""
        await self.ensure_session()
        try:
            async with self.session.post(
                    f"{self.config['webui_url']}{endpoint}",
                    json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise ConnectionError(f"API错误 ({resp.status}): {error}")
                return await resp.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"连接失败: {str(e)}")

    async def _call_t2i_api(self, prompt: str) -> dict:
        """调用 Stable Diffusion 文生图 API"""
        await self.ensure_session()
        payload = await self._generate_payload(prompt)
        return await self._call_sd_api("/sdapi/v1/txt2img", payload)

    async def _apply_image_processing(self, image_origin: str) -> str:
        """统一处理高分辨率修复与超分辨率放大"""

        # 获取配置参数
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        # 根据配置构建payload
        payload = {
            "image": image_origin,
            "upscaling_resize": upscale_factor,  # 使用配置的放大倍数
            "upscaler_1": upscaler,  # 使用配置的上采样算法
            "resize_mode": 0,  # 标准缩放模式
            "show_extras_results": True,  # 显示额外结果
            "upscaling_resize_w": 1,  # 自动计算宽度
            "upscaling_resize_h": 1,  # 自动计算高度
            "upscaling_crop": False,  # 不裁剪图像
            "gfpgan_visibility": 0,  # 不使用人脸修复
            "codeformer_visibility": 0,  # 不使用CodeFormer修复
            "codeformer_weight": 0,  # 不使用CodeFormer权重
            "extras_upscaler_2_visibility": 0  # 不使用额外的上采样算法
        }

        resp = await self._call_sd_api("/sdapi/v1/extra-single-image", payload)
        return resp["image"]

    async def _set_model(self, model_name: str) -> bool:
        """设置图像生成模型，并存入 config"""
        try:
            async with self.session.post(
                    f"{self.config['webui_url']}/sdapi/v1/options",
                    json={"sd_model_checkpoint": model_name}
            ) as resp:
                if resp.status == 200:
                    self.config["base_model"] = model_name  # 存入 config
                    logger.debug(f"模型已设置为: {model_name}")
                    return True
                else:
                    logger.error(f"设置模型失败 (状态码: {resp.status})")
                    return False
        except Exception as e:
            logger.error(f"设置模型异常: {e}")
            return False

    async def _check_webui_available(self) -> (bool, str):
        """服务状态检查"""
        try:
            await self.ensure_session()
            async with self.session.get(f"{self.config['webui_url']}/sdapi/v1/progress") as resp:
                if resp.status == 200:
                    return True, 0
                else:
                    logger.debug(f"⚠️ Stable diffusion Webui 返回值异常，状态码: {resp.status})")
                    return False, resp.status
        except Exception as e:
            logger.debug(f"❌ 测试连接 Stable diffusion Webui 失败，报错：{e}")
            return False, 0

    def _get_generation_params(self) -> str:
        """获取当前图像生成的参数"""
        positive_prompt_global = self.config.get("positive_prompt_global", "")
        negative_prompt_global = self.config.get("negative_prompt_global", "")

        params = self.config.get("default_params", {})
        width = params.get("width") or "未设置"
        height = params.get("height") or "未设置"
        steps = params.get("steps") or "未设置"
        sampler = params.get("sampler") or "未设置"
        cfg_scale = params.get("cfg_scale") or "未设置"

        base_model = self.config.get("base_model").strip() or "未设置"

        return (
            f"- 全局正面提示词: {positive_prompt_global}\n"
            f"- 全局负面提示词: {negative_prompt_global}\n"
            f"- 基础模型: {base_model}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}"
        )

    def _get_upscale_params(self) -> str:
        """获取当前图像增强（超分辨率放大）参数"""
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        return (
            f"- 放大倍数: {upscale_factor}\n"
            f"- 上采样算法: {upscaler}"
        )

    @command_group("sd")
    def sd(self):
        pass

    @sd.command("check")
    async def check(self, event: AstrMessageEvent):
        """服务状态检查"""
        try:
            webui_available, status = await self._check_webui_available()
            if webui_available:
                yield event.plain_result("✅ 同Webui连接正常")
            else:
                yield event.plain_result(f"❌ 同Webui无连接，请检查配置和Webui工作状态")
        except Exception as e:
            logger.error(f"❌ 检查可用性错误，报错{e}")
            yield event.plain_result("❌ 检查可用性错误，请查看控制台输出")

    @sd.command("gen")
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """
        try:
            # 检查webui可用性
            if not (await self._check_webui_available())[0]:
                yield event.plain_result("⚠️ 同webui无连接，目前无法生成图片！")
                return

            verbose = self.config["verbose"]
            if verbose:
                yield event.plain_result("🖌️ 生成图像阶段，这可能需要一段时间...")

            # 生成提示词
            if self.config.get("enable_generate_prompt"):
                generated_prompt = await self._generate_prompt(prompt)
                logger.debug(f"LLM generated prompt: {generated_prompt}")
                positive_prompt = self.config.get("positive_prompt_global", "") + generated_prompt
            else:
                positive_prompt = self.config.get("positive_prompt_global", "") + self._trans_prompt(prompt)
            
            #输出正向提示词
            if self.config.get("enable_show_positive_prompt", False):
                yield event.plain_result(f"正向提示词：{positive_prompt}")
            
            # 生成图像
            response = await self._call_t2i_api(positive_prompt)
            if not response.get("images"):
                raise ValueError("API返回数据异常：生成图像失败")

            image_data = response["images"][0]
            logger.debug(f"img: {image_data}")

            image_bytes = base64.b64decode(image_data)
            image = base64.b64encode(image_bytes).decode("utf-8")

            # 图像处理
            if self.config.get("enable_upscale"):
                if verbose:
                    yield event.plain_result("🖼️ 处理图像阶段，即将结束...")
                image = await self._apply_image_processing(image)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                temp_image.write(base64.b64decode(image))
                temp_image_path = temp_image.name  # 获取临时文件路径

            yield event.image_result(temp_image_path)

            if verbose:
                yield event.plain_result("✅ 图像生成成功")

            os.remove(temp_image_path)
        except ValueError as e:
            # 针对API返回异常的处理
            logger.error(f"API返回数据异常: {e}")
            yield event.plain_result(f"❌ 图像生成失败: 参数异常，API调用失败")

        except ConnectionError as e:
            # 网络连接错误处理
            logger.error(f"网络连接失败: {e}")
            yield event.plain_result("⚠️ 生成失败! 请检查网络连接和WebUI服务是否运行正常")

        except TimeoutError as e:
            # 处理超时错误
            logger.error(f"请求超时: {e}")
            yield event.plain_result("⚠️ 请求超时，请稍后再试")

        except Exception as e:
            # 捕获所有其他异常
            logger.error(f"生成图像时发生其他错误: {e}")
            yield event.plain_result(f"❌ 图像生成失败: 发生其他错误，请查阅控制台日志")

    @sd.command("verbose")
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式（verbose）"""
        try:
            # 读取当前状态并取反
            current_verbose = self.config.get("verbose", True)
            new_verbose = not current_verbose

            # 更新配置
            self.config["verbose"] = new_verbose

            # 发送反馈消息
            status = "开启" if new_verbose else "关闭"
            yield event.plain_result(f"📢 详细输出模式已{status}")
        except Exception as e:
            logger.error(f"切换详细输出模式失败: {e}")
            yield event.plain_result("❌ 切换详细模式失败，请检查配置")

    @sd.command("upscale")
    async def set_upscale(self, event: AstrMessageEvent):
        """设置图像增强模式（enable_upscale）"""
        try:
            # 获取当前的 upscale 配置值
            current_upscale = self.config.get("enable_upscale", False)

            # 切换 enable_upscale 配置
            new_upscale = not current_upscale

            # 更新配置
            self.config["enable_upscale"] = new_upscale

            # 发送反馈消息
            status = "开启" if new_upscale else "关闭"
            yield event.plain_result(f"📢 图像增强模式已{status}")

        except Exception as e:
            logger.error(f"切换图像增强模式失败: {e}")
            yield event.plain_result("❌ 切换图像增强模式失败，请检查配置")

    @sd.command("LLM")
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        try:
            current_setting = self.config.get("enable_generate_prompt", False)
            new_setting = not current_setting
            self.config["enable_generate_prompt"] = new_setting

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 提示词生成功能已{status}")
        except Exception as e:
            logger.error(f"切换生成提示词功能失败: {e}")
            yield event.plain_result("❌ 切换生成提示词功能失败，请检查配置")

    @sd.command("prompt")
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正向提示词功能"""
        try:
            current_setting = self.config.get("enable_show_positive_prompt", False)
            new_setting = not current_setting
            self.config["enable_show_positive_prompt"] = new_setting

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 显示正向提示词功能已{status}")
        except Exception as e:
            logger.error(f"切换显示正向提示词功能失败: {e}")
            yield event.plain_result("❌ 切换显示正向提示词功能失败，请检查配置")

    @sd.command("timeout")
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        try:
            if time < 10 or time > 300:
                yield event.plain_result("⚠️ 超时时间需设置在 10 到 300 秒范围内")
                return

            self.config["session_timeout_time"] = time
            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查配置")

    @sd.command("conf")
    async def show_conf(self, event: AstrMessageEvent):
        """打印当前图像生成参数，包括当前使用的模型"""
        try:
            gen_params = self._get_generation_params()  # 获取当前图像参数
            scale_params = self._get_upscale_params()   # 获取图像增强参数
            prompt_guidelines = self.config.get("prompt_guidelines").strip() or "未设置"  # 获取提示词限制

            verbose = self.config.get("verbose", True)  # 获取详略模式
            upscale = self.config.get("enable_upscale", False)  # 图像增强模式
            show_positive_prompt = self.config.get("enable_show_positive_prompt", False)  # 是否显示正向提示词
            generate_prompt = self.config.get("enable_generate_prompt", False)  # 是否启用生成提示词

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

    @sd.command("help")
    async def show_help(self, event: AstrMessageEvent):
        """显示SDGenerator插件所有可用指令及其描述"""
        help_msg = [
            "🖼️ **Stable Diffusion 插件帮助指南**",
            "该插件用于调用 Stable Diffusion WebUI 的 API 生成图像并管理相关模型资源。",
            "",
            "📜 **主要功能指令列表**:",
            "- `/sd gen [提示词]`：生成图片，例如 `/sd gen 星空下的城堡`。",
            "- `/sd check`：检查 WebUI 的连接状态。",
            "- `/sd conf`：显示当前使用配置，包括模型、参数和提示词设置。",
            "- `/sd help`：显示本帮助信息。",
            "",
            "🔧 **高级功能管理指令**:",
            "- `/sd verbose`：切换详细输出模式，用于显示图像生成步骤。",
            "- `/sd upscale`：切换图像增强模式（用于超分辨率放大或高分修复）。",
            "- `/sd LLM`：切换是否使用 LLM 自动生成提示词。",
            "- `/sd prompt`：切换是否在生成过程显示正向提示词。",
            "- `/sd timeout [秒数]`：设置连接超时时间（范围：10 到 300 秒）。",
            "",
            "🖼️ **模型与资源管理指令**:",
            "- `/sd model list`：列出 WebUI 当前可用的模型。",
            "- `/sd model set [索引]`：根据索引设置模型，索引可通过 `model list` 查询。",
            "- `/sd lora`：列出所有可用的 LoRA 模型。",
            "- `/sd embedding`：显示所有已加载的 Embedding 模型。",
            "",
            "🎨 **采样器与上采样设置指令**:",
            "- `/sd sampler list`：列出支持的采样器。",
            "- `/sd sampler set [索引]`：根据索引配置采样器，用于调整生成效果。",
            "- `/sd upscaler list`：列出支持的上采样算法。",
            "- `/sd upscaler set [索引]`：根据索引设置上采样算法。",
            "",
            "ℹ️ **注意事项**:",
            "- 如启用自动生成提示词功能，则会使用 LLM 根据提供的信息随机生成提示词。"
            "- 如未启用自动生成提示词功能，若自定义的提示词包含空格，则应使用 `_` 替代提示词中的空格。",
            "- 模型、采样器和其他资源的索引需要使用对应 `list` 命令获取后设置！",
            "- 目前在插件中设置的配置再重启时会丢失，如需保存配置请前往AstrBot面板-插件-配置，保存并重启。"
        ]
        yield event.plain_result("\n".join(help_msg))

    @sd.group("model")
    def model(self):
        pass

    @model.command("list")
    async def list_model(self, event: AstrMessageEvent):
        """
        以“1. xxx.safetensors“形式打印可用的模型
        """
        try:
            models = await self._get_sd_model_list()  # 使用统一方法获取模型列表
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            model_list = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(models))
            yield event.plain_result(f"🖼️ 可用模型列表:\n{model_list}")

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            yield event.plain_result("❌ 获取模型列表失败，请检查 WebUI 是否运行")

    @model.command("set")
    async def set_base_model(self, event: AstrMessageEvent, model_index: int):
        """
        解析用户输入的索引，并设置对应的模型
        """
        try:
            models = await self._get_sd_model_list()
            if not models:
                yield event.plain_result("⚠️ 没有可用的模型")
                return

            try:
                index = int(model_index) - 1  # 转换为 0-based 索引
                if index < 0 or index >= len(models):
                    yield event.plain_result("❌ 无效的模型索引，请检查 /sd model list")
                    return

                selected_model = models[index]
                logger.debug(f"selected_model: {selected_model}")
                if await self._set_model(selected_model):
                    self.config["base_model"] = selected_model
                    yield event.plain_result(f"✅ 模型已切换为: {selected_model}")
                else:
                    yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")

        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请查看控制台输出")

    @sd.command("lora")
    async def list_lora(self, event: AstrMessageEvent):
        """
        列出可用的 LoRA 模型
        """
        try:
            lora_models = await self._get_lora_list()
            if not lora_models:
                yield event.plain_result("没有可用的 LoRA 模型。")
            else:
                lora_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(lora_models))
                yield event.plain_result(f"可用的 LoRA 模型:\n{lora_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 LoRA 模型列表失败: {str(e)}")

    @sd.group("sampler")
    def sampler(self):
        pass

    @sampler.command("list")
    async def list_sampler(self, event: AstrMessageEvent):
        """
        列出所有可用的采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            sampler_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(samplers))
            yield event.plain_result(f"🖌️ 可用采样器列表:\n{sampler_list}")
        except Exception as e:
            yield event.plain_result(f"获取采样器列表失败: {str(e)}")

    @sampler.command("set")
    async def set_sampler(self, event: AstrMessageEvent, sampler_index: int):
        """
        设置采样器
        """
        try:
            samplers = await self._get_sampler_list()
            if not samplers:
                yield event.plain_result("⚠️ 没有可用的采样器")
                return

            try:
                index = int(sampler_index) - 1
                if index < 0 or index >= len(samplers):
                    yield event.plain_result("❌ 无效的采样器索引，请检查 /sd sampler list")
                    return

                selected_sampler = samplers[index]
                self.config["default_params"]["sampler"] = selected_sampler
                yield event.plain_result(f"✅ 已设置采样器为: {selected_sampler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    @sd.group("upscaler")
    def upscaler(self):
        pass

    @upscaler.command("list")
    async def list_upscaler(self, event: AstrMessageEvent):
        """
        列出所有可用的上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            upscaler_list = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(upscalers))
            yield event.plain_result(f"🖌️ 可用上采样算法列表:\n{upscaler_list}")
        except Exception as e:
            yield event.plain_result(f"获取上采样算法列表失败: {str(e)}")

    @upscaler.command("set")
    async def set_upscaler(self, event: AstrMessageEvent, upscaler_index: int):
        """
        设置上采样算法
        """
        try:
            upscalers = await self._get_upscaler_list()
            if not upscalers:
                yield event.plain_result("⚠️ 没有可用的上采样算法")
                return

            try:
                index = int(upscaler_index) - 1
                if index < 0 or index >= len(upscalers):
                    yield event.plain_result("❌ 无效的上采样算法索引，请检查 /sd upscaler list")
                    return

                selected_upscaler = upscalers[index]
                self.config["default_params"]["upscaler"] = selected_upscaler
                yield event.plain_result(f"✅ 已设置上采样算法为: {selected_upscaler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")


    @sd.command("embedding")
    async def list_embedding(self, event: AstrMessageEvent):
        """
        列出可用的 Embedding 模型
        """
        try:
            embedding_models = await self._get_embedding_list()
            if not embedding_models:
                yield event.plain_result("没有可用的 Embedding 模型。")
            else:
                embedding_model_list = "\n".join(f"{i + 1}. {lora}" for i, lora in enumerate(embedding_models))
                yield event.plain_result(f"可用的 Embedding 模型:\n{embedding_model_list}")
        except Exception as e:
            yield event.plain_result(f"获取 Embedding 模型列表失败: {str(e)}")

    @llm_tool("generate_image_call")
    async def generate_image_call(self, event: AstrMessageEvent, prompt: str):
        """根据提示词生成图片

        Args:
            prompt(string): 用于图片生成的提示词或提示语
        """
        try:
            # 使用 async for 遍历异步生成器的返回值
            async for result in self.generate_image(event, prompt):
                # 根据生成器的每一个结果返回响应
                yield result

        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            yield event.plain_result("❌ 图像生成失败，请查看控制台日志")
