import asyncio
import os
import re

import aiohttp

from astrbot.api.all import *

try:  #由DS harness生成
    from astrbot.core.message.components import Reply  # 引用消息组件：用于从引用中取图（新版AstrBot提供）#由DS harness生成
except ImportError:  #由DS harness生成
    Reply = None  # 旧版无此组件时，_is_reply_segment退化为类型字符串匹配 #由DS harness生成


TEMP_PATH = os.path.abspath("data/temp")

@register("SDGen", "buding(AstrBot)", "Stable Diffusion图像生成器", "1.2.2")
class SDGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.session = None
        self._validate_config()
        os.makedirs(TEMP_PATH, exist_ok=True)

        # 初始化并发控制
        self.active_tasks = 0
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)  # 设定最大并发数
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

    @staticmethod
    def _select_prompt_option(group: dict, index_key: str, prefix: str, count: int = 4) -> str:
        """Select prompt by index with safe fallback."""
        index = group.get(index_key, 0)
        if not isinstance(index, int) or index < 0 or index >= count:
            index = 0
        return group.get(f"{prefix}{index}", "")

    @staticmethod
    def _compose_prompt(*segments: str) -> str:
        """Join non-empty prompt segments with commas."""
        return ",".join(segment for segment in segments if segment)

    def _validate_config(self):
        """配置验证"""
        self.config["webui_url"] = self.config["webui_url"].strip()
        if not self.config["webui_url"].startswith(("http://", "https://")):
            raise ValueError("WebUI地址必须以http://或https://开头")

        if self.config["webui_url"].endswith("/"):
            self.config["webui_url"] = self.config["webui_url"].rstrip("/")
            self.config.save_config()

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

    def _build_negative_prompt(self) -> str:
        """Assemble negative prompt from global and user presets."""
        global_group = self.config.get("global_prompt_group", {})
        user_negative_group = self.config["user_prompt_group"]["user_negative_prompt_group"]

        global_negative_prompt = (
            global_group.get("global_negative_prompt", "")
            if global_group.get("global_negative_prompt_switch", False)
            else ""
        )
        user_negative_prompt = self._select_prompt_option(
            user_negative_group, "user_negative_prompt_list", "user_negative_prompt"
        )
        return self._compose_prompt(global_negative_prompt, user_negative_prompt)

    async def _generate_payload(self, prompt: str) -> dict:
        """构建生成参数"""
        params = self.config["default_params"]
        negative_prompt = self._build_negative_prompt()

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
        }

    def _trans_prompt(self, prompt: str) -> str:
        """返回原始提示词（保留空格）"""
        return prompt

    @staticmethod
    def _extract_prompt_from_message(event: AstrMessageEvent, raw_prompt: str, subcommand: str = "gen") -> str:
        """从原始消息还原提示词，避免参数解析截断空格"""
        full = (event.message_str or "").strip()
        base = (raw_prompt or "").strip()

        if not full:
            return base

        tokens = full.split()
        if tokens and tokens[0].lstrip("/") in ("sd",):
            tokens = tokens[1:]
        # [修改] 子命令名参数化：默认"gen"保持原有行为不变，i2i传入"i2i"复用同一还原逻辑 #由DS harness生成
        if tokens and tokens[0] == subcommand:
            tokens = tokens[1:]

        fallback = " ".join(tokens).strip()
        return fallback or base

    def _build_positive_prompt(self, raw_prompt: str, generated_prompt: str) -> str:
        """Construct final positive prompt with global/user presets."""
        global_group = self.config.get("global_prompt_group", {})
        user_positive_group = self.config["user_prompt_group"]["user_positive_prompt_group"]

        global_positive_prompt = (
            global_group.get("global_positive_prompt", "")
            if global_group.get("global_positive_prompt_switch", False)
            else ""
        )
        add_global_first = global_group.get("positive_prompt_add_in_head_or_tail_switch", False)
        user_positive_prompt = self._select_prompt_option(
            user_positive_group, "user_positive_prompt_list", "user_positive_prompt"
        )

        base_prompt = (
            generated_prompt if self.config.get("enable_generate_prompt") and generated_prompt else self._trans_prompt(raw_prompt)
        )

        if add_global_first:
            return self._compose_prompt(global_positive_prompt, user_positive_prompt, base_prompt)
        return self._compose_prompt(base_prompt, global_positive_prompt, user_positive_prompt)

    async def _generate_prompt(self, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            prompt_guidelines = self.config["prompt_guidelines"]
            prompt_generate_text = (
                "请根据以下描述生成用于 Stable Diffusion WebUI 的英文提示词，"
                "请返回一条逗号分隔的 `prompt` 英文字符串，适用于 Stable Diffusion web UI，"
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
                    self.config.save_config()

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
        global_positive_prompt_switch = self.config.get("global_prompt_group").get("global_positive_prompt_switch", False)  # 获取全局正面提示词开关状态
        global_negative_prompt_switch = self.config.get("global_prompt_group").get("global_negative_prompt_switch", False)  # 获取全局负面提示词开关状态
        global_positive_prompt = self.config.get("global_prompt_group").get("global_positive_prompt", "") # 获取全局正面提示词
        global_negative_prompt = self.config.get("global_prompt_group").get("global_negative_prompt", "")   #获取全局负面提示词

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
            f"- 全局正面提示词: {'开启' if global_positive_prompt_switch else '关闭'}\n"
            f"- 全局正面提示词: {global_positive_prompt}\n"
            f"- 全局负面提示词: {'开启' if global_negative_prompt_switch else '关闭'}\n"
            f"- 全局负面提示词: {global_negative_prompt}\n"
            f"- 基础模型: {base_model}\n"
            f"- 图片尺寸: {width}x{height}\n"
            f"- 步数: {steps}\n"
            f"- 采样器: {sampler}\n"
            f"- CFG比例: {cfg_scale}\n"
            f"- 批数量: {batch_size}\n"
            f"- 迭代次数: {n_iter}"
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

    @sd.command("check")    # 服务状态检查
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
            yield event.plain_result("❌ 检查可用性错误，请检查日志")

    async def _run_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        allow_generate_prompt: bool,
        allow_extract_prompt: bool
    ):
        """Shared image generation logic for command/tool callers."""
        async with self.task_semaphore:
            self.active_tasks += 1
            try:
                if allow_extract_prompt:
                    prompt = self._extract_prompt_from_message(event, prompt)
                else:
                    prompt = (prompt or "").strip()
                if not prompt:
                    yield event.plain_result("⚠️ 需要提供提示词")
                    return
                # 检查webui可用性
                if not (await self._check_webui_available())[0]:
                    yield event.plain_result("⚠️ 同webui无连接，目前无法生成图片！")
                    return

                verbose = self.config["verbose"]
                if verbose:
                    yield event.plain_result("🖌️ 生成图像阶段，这可能需要一段时间...")

                # 生成正面提示词，决定到底是使用LLM生成还是用户直接提供
                generated_prompt = ""
                if allow_generate_prompt and self.config.get("enable_generate_prompt"):
                    generated_prompt = await self._generate_prompt(prompt)
                    logger.debug(f"LLM generated prompt: {generated_prompt}")

                positive_prompt = self._build_positive_prompt(prompt, generated_prompt)

                #输出正面提示词
                if self.config.get("enable_show_positive_prompt", False):
                    yield event.plain_result(f"正面提示词：{positive_prompt}")

                # 生成图像
                response = await self._call_t2i_api(positive_prompt)
                if not response.get("images"):
                    raise ValueError("API返回数据异常：生成图像失败")

                images = response["images"]

                if len(images) == 1:

                    image_data = response["images"][0]

                    image_bytes = base64.b64decode(image_data)
                    image = base64.b64encode(image_bytes).decode("utf-8")

                    # 图像处理
                    if self.config.get("enable_upscale"):
                        if verbose:
                            yield event.plain_result("🖼️ 处理图像阶段，即将结束...")
                        image = await self._apply_image_processing(image)

                    yield event.chain_result([Image.fromBase64(image)])
                else:
                    chain = []

                    if self.config.get("enable_upscale") and verbose:
                        yield event.plain_result("🖼️ 处理图像阶段，即将结束...")

                    for image_data in images:
                        image_bytes = base64.b64decode(image_data)
                        image = base64.b64encode(image_bytes).decode("utf-8")

                        # 图像处理
                        if self.config.get("enable_upscale"):
                            image = await self._apply_image_processing(image)

                        # 添加到链对象
                        chain.append(Image.fromBase64(image))

                    # 将链式结果发送给事件
                    yield event.chain_result(chain)

                if verbose:
                    yield event.plain_result("✅ 图像生成成功")

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
                yield event.plain_result(f"❌ 图像生成失败: 发生其他错误，请检查日志")
            finally:
                self.active_tasks -= 1

    @sd.command("gen")  # 生成图像指令
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """生成图像指令
        Args:
            prompt: 图像描述提示词
        """
        async for result in self._run_generate_image(
            event,
            prompt,
            allow_generate_prompt=True,
            allow_extract_prompt=True
        ):
            yield result

    @staticmethod
    def _is_reply_segment(seg) -> bool:
        """识别引用消息段：优先 isinstance(Reply)，旧版 AstrBot 无此类时按类型值匹配。#由DS harness生成"""
        if Reply is not None:
            return isinstance(seg, Reply)
        t = str(getattr(seg, "type", "")).lower()
        return t == "reply" or t.endswith("componenttype.reply")

    def _find_i2i_image(self, event: AstrMessageEvent):
        """[i2i] 定位输入图片。优先级：引用消息中的图 > 当前消息直接附带的图（宽容降级）。#由DS harness生成
        返回 (Image组件或None, 来源/原因描述)。"""
        # [新增] getattr兜底：AstrBot版本差异导致属性路径不同时优雅降级为"未找到图片"而非崩溃 #由DS harness生成
        segments = getattr(getattr(event, "message_obj", None), "message", None) or []
        reply_seg = None
        first_direct = None
        for seg in segments:                        # 一趟扫描收集两类候选 #由DS harness生成
            if self._is_reply_segment(seg) and reply_seg is None:
                reply_seg = seg                     # AstrBot已自动get_msg，chain里是原消息完整段列表
            elif isinstance(seg, Image) and first_direct is None:
                first_direct = seg                  # 当前消息直接附带的图

        if reply_seg is not None:                   # —— 引用分支：宽容降级链 —— #由DS harness生成
            chain = getattr(reply_seg, "chain", None) or []
            for sub in chain:                       # 1. 引用里的第一张图（多图取首张）
                if isinstance(sub, Image):
                    return sub, "引用消息中的图片"
            if not chain:                           # 2. chain为空 = get_msg失败（原消息太旧/被撤回）
                if first_direct is not None:
                    return first_direct, "随消息发送的图片（引用内容获取失败，已改用直发图）"
                return None, "无法获取引用的消息内容，请重新引用最近的图片"
            if first_direct is not None:            # 3. 引用里没图 → 降级用直发图
                return first_direct, "随消息发送的图片（引用中未找到图片，已改用直发图）"
            return None, "引用的消息里没有图片，请引用图片或把图片和指令一起发送"

        if first_direct is not None:                # —— 无引用分支：用直发图 —— #由DS harness生成
            return first_direct, "随消息发送的图片"
        return None, "未找到图片，请引用一张图片，或将图片与指令一起发送"

    @sd.command("i2i")  # i2i：V1测试期——仅验证"引用→取图"链路，不调用SD WebUI #由DS harness生成
    async def i2i(self, event: AstrMessageEvent, prompt: str):
        """[V1测试期] 从引用/直发消息中取出输入图片并回显确认（提示词仅回显验证，暂不使用）"""  #由DS harness生成
        full_prompt = self._extract_prompt_from_message(event, prompt, subcommand="i2i")  # 还原多词提示词，与/sd gen同机制 #由DS harness生成
        img, source_desc = self._find_i2i_image(event)
        if img is None:
            yield event.plain_result(f"⚠️ {source_desc}")
            return
        try:
            b64 = await img.convert_to_base64()     # AstrBot内置：自动处理本地文件/URL下载 #由DS harness生成
        except Exception as e:                      # 防御：临时文件被清理、CDN下载失败等
            logger.error(f"[i2i测试] 图片转base64失败: {e}")
            yield event.plain_result("❌ 图片内容获取失败，原消息可能过旧或图片已失效，请重试")
            return
        size_kb = len(b64) * 3 / 4 / 1024           # base64长度反推二进制大小（约值）#由DS harness生成
        yield event.plain_result(
            f"✅ [i2i测试期] 成功获取输入图片（{source_desc}）\n"
            f"- 图片大小：约 {size_kb:.0f} KB\n"
            f"- 收到提示词：{full_prompt if full_prompt else '（无）'}\n"
            "取图链路验证通过，下方回显原图供核对（测试期行为，正式版移除）："
        )
        yield event.chain_result(Image.fromBase64(b64))  # 回显确认；Image.fromBase64插件t2i输出已在用 #由DS harness生成

    @sd.command("verbose")  # 切换详细输出模式
    async def set_verbose(self, event: AstrMessageEvent):
        """切换详细输出模式（verbose）"""
        try:
            # 读取当前状态并取反
            current_verbose = self.config.get("verbose", True)
            new_verbose = not current_verbose

            # 更新配置
            self.config["verbose"] = new_verbose
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_verbose else "关闭"
            yield event.plain_result(f"📢 详细输出模式已{status}")
        except Exception as e:
            logger.error(f"切换详细输出模式失败: {e}")
            yield event.plain_result("❌ 切换详细模式失败，请检查日志")

    @sd.command("upscale") # 切换图像增强模式
    async def set_upscale(self, event: AstrMessageEvent):
        """设置图像增强模式（enable_upscale）"""
        try:
            # 获取当前的 upscale 配置值
            current_upscale = self.config.get("enable_upscale", False)

            # 切换 enable_upscale 配置
            new_upscale = not current_upscale

            # 更新配置
            self.config["enable_upscale"] = new_upscale
            self.config.save_config()

            # 发送反馈消息
            status = "开启" if new_upscale else "关闭"
            yield event.plain_result(f"📢 图像增强模式已{status}")

        except Exception as e:
            logger.error(f"切换图像增强模式失败: {e}")
            yield event.plain_result("❌ 切换图像增强模式失败，请检查日志")

    @sd.command("LLM")  # 切换生成提示词功能
    async def set_generate_prompt(self, event: AstrMessageEvent):
        """切换生成提示词功能"""
        try:
            current_setting = self.config.get("enable_generate_prompt", False)
            new_setting = not current_setting
            self.config["enable_generate_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 提示词生成功能已{status}")
        except Exception as e:
            logger.error(f"切换生成提示词功能失败: {e}")
            yield event.plain_result("❌ 切换生成提示词功能失败，请检查日志")

    @sd.command("headtail") # 切换全局正面提示词添加位置
    async def switch_positive_prompt_add_in_head_or_tail(self, event: AstrMessageEvent):
        """切换全局正面提示词添加位置"""
        try:
            current_setting = self.config.get("global_prompt_group").get("positive_prompt_add_in_head_or_tail_switch", False)
            new_setting = not current_setting
            self.config["global_prompt_group"]["positive_prompt_add_in_head_or_tail_switch"] = new_setting
            self.config.save_config()

            status = "头部" if new_setting else "尾部"
            yield event.plain_result(f"📢 全局正面提示词现将添加在 {status}")
        except Exception as e:
            logger.error(f"切换全局正面提示词位置失败: {e}")
            yield event.plain_result("❌ 切换全局正面提示词位置失败，请检查日志")

    @sd.command("prompt") # 切换显示正面提示词功能
    async def set_show_prompt(self, event: AstrMessageEvent):
        """切换显示正面提示词功能"""
        try:
            current_setting = self.config.get("enable_show_positive_prompt", False)
            new_setting = not current_setting
            self.config["enable_show_positive_prompt"] = new_setting
            self.config.save_config()

            status = "开启" if new_setting else "关闭"
            yield event.plain_result(f"📢 显示正面提示词功能已{status}")
        except Exception as e:
            logger.error(f"切换显示正面提示词功能失败: {e}")
            yield event.plain_result("❌ 切换显示正面提示词功能失败，请检查日志")

    @sd.command("pprompt")  # 设置生效的用户预设正面提示词
    async def set_user_positive_prompt(self, event: AstrMessageEvent, pprompt: int):
        """设置生效的用户预设正面提示词"""
        try:
            if pprompt not in [0, 1, 2, 3]:
                yield event.plain_result("⚠️ 可选择的用户正面提示词序号仅有 0、1、2、3")
                return

            self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt_list"] = pprompt
            self.config.save_config()

            yield event.plain_result(f"➕{pprompt} 现在使用：用户预设正面提示词{pprompt}")
        except Exception as e:
            logger.error(f"设置用户正面提示词失败: {e}")
            yield event.plain_result("❌ 设置用户预设正面提示词失败，请检查日志")

    @sd.command("nprompt")  # 设置生效的用户预设负面提示词
    async def set_user_negative_prompt(self, event: AstrMessageEvent, nprompt: int):
        """设置生效的用户预设负面提示词"""
        try:
            if nprompt not in [0, 1, 2, 3]:
                yield event.plain_result("⚠️ 可选择的用户负面提示词序号仅有 0、1、2、3")
                return

            self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt_list"] = nprompt
            self.config.save_config()

            yield event.plain_result(f"➖{nprompt} 现在使用：用户预设负面提示词{nprompt}")
        except Exception as e:
            logger.error(f"设置用户负面提示词失败: {e}")
            yield event.plain_result("❌ 设置用户预设负面提示词失败，请检查日志")


    @sd.command("timeout")  # 设置会话超时时间
    async def set_timeout(self, event: AstrMessageEvent, time: int):
        """设置会话超时时间"""
        try:
            if time < 10 or time > 1800:
                yield event.plain_result("⚠️ 超时时间需设置在 10秒 到 1800秒 范围内")
                return

            self.config["session_timeout_time"] = time
            self.config.save_config()

            yield event.plain_result(f"⏲️ 会话超时时间已设置为 {time} 秒")
        except Exception as e:
            logger.error(f"设置会话超时时间失败: {e}")
            yield event.plain_result("❌ 设置会话超时时间失败，请检查日志")

    @sd.command("conf") # 输出当前各项配置
    async def show_conf(self, event: AstrMessageEvent):
        """打印当前图像生成参数，包括当前使用的模型"""
        try:
            global_positive_prompt_switch = self.config.get("global_prompt_group").get("global_positive_prompt_switch", False)  # 获取全局正面提示词开关状态
            global_negative_prompt_switch = self.config.get("global_prompt_group").get("global_negative_prompt_switch", False)  # 获取全局负面提示词开关状态

            pprompt = self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt_list"]  # 获取生效的用户预设正面提示词序号
            nprompt = self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt_list"]  # 获取生效的用户预设负面提示词序号

            user_positive_prompt0 = self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt0"] # 获取正面提示词组0
            user_positive_prompt1 = self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt1"] # 获取正面提示词组1
            user_positive_prompt2 = self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt2"] # 获取正面提示词组2
            user_positive_prompt3 = self.config["user_prompt_group"]["user_positive_prompt_group"]["user_positive_prompt3"] # 获取正面提示词组3
            user_negative_prompt0 = self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt0"] # 获取负面提示词组0
            user_negative_prompt1 = self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt1"] # 获取负面提示词组1
            user_negative_prompt2 = self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt2"] # 获取负面提示词组2
            user_negative_prompt3 = self.config["user_prompt_group"]["user_negative_prompt_group"]["user_negative_prompt3"] # 获取负面提示词组3

            gen_params = self._get_generation_params()  # 获取当前图像参数
            scale_params = self._get_upscale_params()   # 获取图像增强参数
            prompt_guidelines = self.config.get("prompt_guidelines").strip() or "未设置"  # 获取提示词限制

            positive_prompt_add_in_head_or_tail_switch = self.config.get("global_prompt_group").get('positive_prompt_add_in_head_or_tail_switch',False) # 获取全局正面提示词添加位置

            verbose = self.config.get("verbose", True)  # 获取详略模式
            upscale = self.config.get("enable_upscale", False)  # 图像增强模式
            show_positive_prompt = self.config.get("enable_show_positive_prompt", False)  # 是否显示正面提示词
            generate_prompt = self.config.get("enable_generate_prompt", False)  # 是否启用生成提示词

            conf_message = (
                f"⚙️  图像生成参数:\n{gen_params}\n\n"
                f"⬅️➡️  全局正面提示词加在 {'头部' if positive_prompt_add_in_head_or_tail_switch else '尾部'}\n\n"
                f"➕  生效的用户预设正面提示词序号:{pprompt}\n"
                f"➖  生效的用户预设负面提示词序号:{nprompt}\n\n"
                f"➕0  用户预设正面提示词组0:{user_positive_prompt0}\n"
                f"➕1  用户预设正面提示词组1:{user_positive_prompt1}\n"
                f"➕2  用户预设正面提示词组2:{user_positive_prompt2}\n"
                f"➕3  用户预设正面提示词组3:{user_positive_prompt3}\n"
                f"➖0  用户预设负面提示词组0:{user_negative_prompt0}\n"
                f"➖1  用户预设负面提示词组1:{user_negative_prompt1}\n"
                f"➖2  用户预设负面提示词组2:{user_negative_prompt2}\n"
                f"➖3  用户预设负面提示词组3:{user_negative_prompt3}\n\n"
                f"🔍  图像增强参数:\n{scale_params}\n\n"
                f"🛠️  提示词附加要求: {prompt_guidelines}\n\n"
                f"📢  详细输出模式: {'开启' if verbose else '关闭'}\n\n"
                f"🔧  图像增强模式: {'开启' if upscale else '关闭'}\n\n"
                f"📝  正面提示词显示: {'开启' if show_positive_prompt else '关闭'}\n\n"
                f"🤖  提示词生成模式: {'开启' if generate_prompt else '关闭'}"
            )

            yield event.plain_result(conf_message)
        except Exception as e:
            logger.error(f"获取生成参数失败: {e}")
            yield event.plain_result("❌ 获取图像生成参数失败，请检查配置是否正确")

    @sd.command("help") # 帮助指令
    async def show_help(self, event: AstrMessageEvent):
        """显示SDGenerator插件所有可用指令及其描述"""
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
            "➕➖ **正负提示词设置指令**:",
            "- `/sd headtail`：切换全局正面提示词添加位置（头部或尾部）。",
            "- `/sd pprompt`： 设置生效的用户预设正面提示词（0，1，2，3）。",
            "- `/sd nprompt`： 设置生效的用户预设负面提示词（0，1，2，3）。",
            "",
            "🔧 **高级功能指令**:",
            "- `/sd verbose`：切换详细输出模式，用于实时告知目前AI生图进行到了哪个阶段。",
            "- `/sd upscale`：切换图像增强模式（用于超分辨率放大或高分修复）。",
            "- `/sd LLM`：开启后，在使用/sd gen指令时，将内容先发送给LLM，再由LLM来生成正面提示词",
            "- `/sd prompt`：开启时，用户发起AI生图请求后，将发送一条消息，内容为送入到Stable diffusion的正面提示词",
            "- `/sd timeout [秒数]`：设置连接超时时间（建议范围：10 到 1800 秒）。",
            "- `/sd res  [宽度] [高度]`：设置图像生成的分辨率（高度和宽度均支持:1-2048之间的任意整数）。",
            "- `/sd step [步数]`：设置图像生成的步数（范围：10 到 50 步）。",
            "- `/sd batch [数量]`：设置发出AI生图请求后，每轮生成的图片数量（范围： 1 到 10 张）。"
            "- `/sd iter [次数]`：设置迭代次数（范围： 1 到 5 次）。"
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
            "- 如未启用自动生成提示词功能，若提供的自定义提示词中包含空格，则应使用 “~”（英文波浪号） 替代所有提示词中的空格，否则输入的自定义提示词组将在空格处中断。你可以在配置中修改想使用的字符。",
            "- 模型、采样器和其他资源的索引需要使用对应 `list` 命令获取后设置！",
        ]
        yield event.plain_result("\n".join(help_msg))

    @sd.command("res") # 设置生成图像的宽和高
    async def set_resolution(self, event: AstrMessageEvent, width: int,height: int ):
        """设置分辨率"""
        try:
            if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1 or height > 2048 or width > 2048:
                yield event.plain_result("⚠️ 分辨率仅支持:1-2048之间的任意整数")
                return

            self.config["default_params"]["height"] = height
            self.config["default_params"]["width"] = width
            self.config.save_config()

            yield event.plain_result(f"✅ 图像生成的分辨率已设置为: 宽度——{width}，高度——{height}")
        except Exception as e:
            logger.error(f"设置分辨率失败: {e}")
            yield event.plain_result("❌ 设置分辨率失败，请检查日志")

    @sd.command("step")# 设置生成图像的步数
    async def set_step(self, event: AstrMessageEvent, step: int):
        """设置步数"""
        try:
            if step < 10 or step > 50:
                yield event.plain_result("⚠️ 步数需设置在 10 到 50 之间")
                return

            self.config["default_params"]["steps"] = step
            self.config.save_config()

            yield event.plain_result(f"✅ 步数已设置为: {step}")
        except Exception as e:
            logger.error(f"设置步数失败: {e}")
            yield event.plain_result("❌ 设置步数失败，请检查日志")

    @sd.command("batch") # 设置一次性生成的图片数量
    async def set_batch_size(self, event: AstrMessageEvent, batch_size: int):
        """设置批量生成的图片数量"""
        try:
            if batch_size < 1 or batch_size > 10:
                yield event.plain_result("⚠️ 图片生成的批数量需设置在 1 到 10 之间")
                return

            self.config["default_params"]["batch_size"] = batch_size
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成批数量已设置为: {batch_size}")
        except Exception as e:
            logger.error(f"设置批量生成数量失败: {e}")
            yield event.plain_result("❌ 设置图片生成批数量失败，请检查日志")

    @sd.command("iter") # 设置生成图像的迭代次数
    async def set_n_iter(self, event: AstrMessageEvent, n_iter: int):
        """设置生成迭代次数"""
        try:
            if n_iter < 1 or n_iter > 5:
                yield event.plain_result("⚠️ 图片生成的迭代次数需设置在 1 到 5 之间")
                return

            self.config["default_params"]["n_iter"] = n_iter
            self.config.save_config()

            yield event.plain_result(f"✅ 图片生成的迭代次数已设置为: {n_iter}")
        except Exception as e:
            logger.error(f"设置生成迭代次数失败: {e}")
            yield event.plain_result("❌ 设置图片生成的迭代次数失败，请检查日志")

    @sd.group("model") #引出模型设置子命令
    def model(self):
        pass

    @model.command("list") # 列出可用的生图模型
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

    @model.command("set") # 设置使用哪个生图模型
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
                    yield event.plain_result("❌ 无效的模型索引，请使用 /sd model list 获取")
                    return

                selected_model = models[index]
                logger.debug(f"selected_model: {selected_model}")
                if await self._set_model(selected_model):
                    yield event.plain_result(f"✅ 模型已切换为: {selected_model}")
                else:
                    yield event.plain_result("⚠️ 切换模型失败，请检查 WebUI 状态")

            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")

        except Exception as e:
            logger.error(f"切换模型失败: {e}")
            yield event.plain_result("❌ 切换模型失败，请检查日志")

    @sd.command("lora") # 列出可用的 LoRA 模型
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

    @sd.group("sampler") # 引出采样器设置子命令
    def sampler(self):
        pass

    @sampler.command("list") # 列出可用的采样器
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

    @sampler.command("set") # 设置采样器
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
                    yield event.plain_result("❌ 无效的采样器索引，请使用 /sd sampler list 获取")
                    return

                selected_sampler = samplers[index]
                self.config["default_params"]["sampler"] = selected_sampler
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置采样器为: {selected_sampler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置采样器失败: {str(e)}")

    @sd.group("upscaler") # 引出上采样算法设置子命令
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

    @upscaler.command("set") # 设置上采样算法
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
                self.config.save_config()

                yield event.plain_result(f"✅ 已设置上采样算法为: {selected_upscaler}")
            except ValueError:
                yield event.plain_result("❌ 请输入有效的数字索引")
        except Exception as e:
            yield event.plain_result(f"设置上采样算法失败: {str(e)}")


    @sd.command("embedding") # 列出可用的 Embedding 模型
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

    @llm_tool("generate_image") # LLM可调用的图像生成工具函数
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """Generate images using Stable Diffusion based on the given prompt.
        This function should only be called when the prompt contains keywords like "generate," "draw," or "create."
        It should not be mistakenly used for image searching.
        The prompt should be ready for Stable Diffusion; no additional prompt generation is performed.

        Args:
            prompt (string): The prompt or description used for generating images.
        """
        try:
            # 使用 async for 遍历异步生成器的返回值
            async for result in self._run_generate_image(
                event,
                prompt,
                allow_generate_prompt=False,
                allow_extract_prompt=False
            ):
                # 根据生成器的每一个结果返回响应
                yield result

        except Exception as e:
            logger.error(f"调用 generate_image 时出错: {e}")
            yield event.plain_result("❌ 图像生成失败，请检查日志")
