"""配置管理模块"""

import os


class ConfigManager:
    """配置管理器"""

    def __init__(self, config):
        self.config = config

    def validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")
            self.config.save_config()

    def get_generation_params(self) -> str:
        """获取当前图像生成的参数"""
        positive_prompt_global = self.config.get("positive_prompt_global", "")
        negative_prompt_global = self.config.get("negative_prompt_global", "")

        params = self.config.get("default_params", {})
        width = params.get("width") or "未设置"
        height = params.get("height") or "未设置"
        steps = params.get("steps") or "未设置"
        sampler = params.get("sampler") or "未设置"
        cfg_scale = params.get("cfg_scale") or "未设置"
        batch_size = params.get("batch_size") or "未设置"
        n_iter = params.get("n_iter") or "未设置"

        base_model = self.config.get("base_model").strip() or "未设置"

        return (
            f"- 全局正面提示词: {positive_prompt_global}\n"
            f"- 全局负面提示词: {negative_prompt_global}\n"
            f"- 基础模型: {base_model}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}\n"
            f"- 批数量: {batch_size}\n"
            f"- 迭代次数: {n_iter}"
        )

    def get_upscale_params(self) -> str:
        """获取当前图像增强（超分辨率放大）参数"""
        params = self.config["default_params"]
        upscale_factor = params["upscale_factor"] or "2"
        upscaler = params["upscaler"] or "未设置"

        return (
            f"- 放大倍数: {upscale_factor}\n"
            f"- 上采样算法: {upscaler}"
        )

    def get_session_timeout(self):
        """获取会话超时时间"""
        return self.config.get("session_timeout_time", 120)

    def get_webui_url(self):
        """获取WebUI URL"""
        return self.config["webui_url"]

    def get_verbose_mode(self):
        """获取详细输出模式"""
        return self.config.get("verbose", True)

    def get_upscale_enabled(self):
        """获取图像增强模式状态"""
        return self.config.get("enable_upscale", False)

    def get_show_positive_prompt(self):
        """获取显示正向提示词状态"""
        return self.config.get("enable_show_positive_prompt", False)

    def get_generate_prompt_enabled(self):
        """获取生成提示词功能状态"""
        return self.config.get("enable_generate_prompt", False)

    def get_positive_prompt_add_position(self):
        """获取全局正面提示词添加位置"""
        return self.config.get("enable_positive_prompt_add_in_head_or_tail", True)

    def get_positive_prompt_global(self):
        """获取全局正面提示词"""
        return self.config.get("positive_prompt_global", "")

    def get_negative_prompt_global(self):
        """获取全局负面提示词"""
        return self.config["negative_prompt_global"]

    def get_default_params(self):
        """获取默认参数"""
        return self.config["default_params"]

    def get_prompt_guidelines(self):
        """获取提示词指导原则"""
        return self.config.get("prompt_guidelines", "")

    def get_replace_space_char(self):
        """获取空格替换字符"""
        return self.config.get("replace_space", "~")

    def update_config(self, key: str, value):
        """更新配置并保存"""
        self.config[key] = value
        self.config.save_config()

    def update_default_param(self, param: str, value):
        """更新默认参数并保存"""
        self.config["default_params"][param] = value
        self.config.save_config()