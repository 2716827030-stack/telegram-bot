"""Telegram bot: photo -> prompt -> RunningHub image-to-image (优化版).

主要改进:
1. DuckDuckGoose 隐写术检测与自动解码
2. 即时响应 + 后台异步处理（不再阻塞）
3. 完善的错误处理与超时机制（防止卡死）
4. 并发任务队列（支持多任务同时处理）
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, BotCommand
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler,
    filters, CallbackQueryHandler, JobQueue
)

from bot.env_bootstrap import bootstrap_env, project_root
from bot.runninghub import RunningHubClient, RunningHubConfig, RunningHubError
from bot.steganography import detect_and_decode, extract_video_frames, extract_video_frames_opencv

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

PENDING_RH_IMAGES = "pending_rh_images"  # 多张待处理图片的列表
UPLOADING_MESSAGE = "uploading_message"
PENDING_PROMPT = "pending_prompt"

# 用于生成递增任务ID
task_counter = 0


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationTask:
    """生成任务数据类"""
    task_id: str
    user_id: int
    chat_id: int
    message_id: int
    rh_image_name: str
    prompt: str
    original_image_bytes: Optional[bytes] = None  # 保存原始图片字节，用于读取尺寸
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_urls: list[str] = field(default_factory=list)
    decoded_text: Optional[str] = None
    decoded_result: Optional[tuple] = None  # (type, data)
    video_frames: list[bytes] = field(default_factory=list)  # 视频帧的列表
    is_stego: bool = False  # 是否检测到隐写
    cached_images: list[bytes] = field(default_factory=list)  # 缓存的生成图片，避免重复下载


class TaskQueue:
    """并发任务队列管理器"""

    def __init__(self, max_concurrent: int = 3):
        self._tasks: dict[str, GenerationTask] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._running_count = 0
        self._last_activity = datetime.now()  # 保活追踪
    
    async def add_task(self, task: GenerationTask) -> str:
        """添加新任务"""
        self.mark_active()
        async with self._lock:
            self._tasks[task.task_id] = task
        return task.task_id
    
    async def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """获取任务状态"""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def get_user_tasks(self, user_id: int) -> list[GenerationTask]:
        """获取用户的所有任务"""
        async with self._lock:
            return [t for t in self._tasks.values() if t.user_id == user_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    return True
        return False

    def mark_active(self) -> None:
        """标记活跃时间（有任务提交或完成时调用）"""
        self._last_activity = datetime.now()

    @property
    def idle_seconds(self) -> float:
        return (datetime.now() - self._last_activity).total_seconds()

    async def remove_completed_tasks(self, older_than_minutes: int = 30):
        """清理已完成的任务"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(minutes=older_than_minutes)
        async with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and task.completed_at and task.completed_at < cutoff
            ]
            for tid in to_remove:
                del self._tasks[tid]
    
    async def process_task(
        self,
        task_id: str,
        rh_client: RunningHubClient,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """异步处理任务 - 使用安全方式"""
        task = await self.get_task(task_id)
        if not task or task.status == TaskStatus.CANCELLED:
            return
        
        # 获取必要信息
        bot = context.bot
        chat_id = task.chat_id

        async with self._semaphore:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            self._running_count += 1
            
            try:
                logger.info(f"开始处理任务 {task_id}")
                
                # 读取原始图片尺寸，传给 RunningHub
                extra_node_info = []
                if task.original_image_bytes:
                    try:
                        from PIL import Image
                        img = Image.open(io.BytesIO(task.original_image_bytes))
                        width, height = img.size
                        logger.info(f"原始图片尺寸: {width}x{height}")
                        extra_node_info = [
                            {"nodeId": "11", "fieldName": "width", "fieldValue": width},
                            {"nodeId": "11", "fieldName": "height", "fieldValue": height},
                        ]
                    except Exception as e:
                        logger.error(f"读取图片尺寸失败: {e}")
                
                # 创建 RunningHub 任务
                task_id_rh = await asyncio.wait_for(
                    rh_client.create_task(task.rh_image_name, task.prompt, extra_node_info=extra_node_info),
                    timeout=30.0
                )
                
                # 等待输出
                outputs = await asyncio.wait_for(
                    rh_client.wait_for_outputs(task_id_rh),
                    timeout=600.0
                )
                
                urls = [o.get("fileUrl") for o in outputs if o.get("fileUrl")]
                if not urls:
                    task.status = TaskStatus.FAILED
                    task.error_message = "任务完成但未返回图片 URL"
                    task.completed_at = datetime.now()
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=f"❌ 任务 #{task_id} 失败: 任务完成但未返回图片"
                        )
                    except:
                        pass
                    return
                
                task.result_urls = urls

                # 并发下载所有结果图片
                import httpx
                import os

                async def _download_one(client: httpx.AsyncClient, url: str) -> bytes | None:
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        logger.info(f"下载完成: {url} ({len(resp.content)//1024}KB)")
                        return resp.content
                    except Exception as e:
                        logger.error(f"下载失败 {url}: {e}")
                        return None

                logger.info(f"并发下载 {len(urls)} 张结果图片...")
                async with httpx.AsyncClient(timeout=120.0) as dl_client:
                    results = await asyncio.wait_for(
                        asyncio.gather(*[_download_one(dl_client, u) for u in urls]),
                        timeout=180.0
                    )
                task.cached_images = [r for r in results if r is not None]
                logger.info(f"下载完成: {len(task.cached_images)}/{len(urls)} 张")

                if not task.cached_images:
                    task.status = TaskStatus.FAILED
                    task.error_message = "无法下载结果图片"
                    task.completed_at = datetime.now()
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=f"❌ 任务 #{task_id} 失败: 无法下载结果图片"
                        )
                    except:
                        pass
                    return

                # 检测隐写（使用缓存的第一张图）
                task.decoded_result = None
                for idx, image_bytes in enumerate(task.cached_images):
                    try:
                        is_stego, decoded_text, raw_data, ext = detect_and_decode(image_bytes)
                        if is_stego:
                            task.is_stego = True
                            logger.info(f"检测到隐写内容！扩展名: {ext}")

                            if ext.lower() in ["png", "jpg", "jpeg", "bmp", "webp", "gif"]:
                                temp_dir = os.path.join(os.path.dirname(__file__), "temp")
                                os.makedirs(temp_dir, exist_ok=True)
                                temp_file = os.path.join(temp_dir, f"decoded_{task_id}.{ext}")
                                with open(temp_file, "wb") as f:
                                    f.write(raw_data)
                                task.decoded_result = ("image", temp_file)
                            elif ext.lower() == "mp4":
                                frames = extract_video_frames(raw_data)
                                if not frames:
                                    frames = extract_video_frames_opencv(raw_data)
                                task.video_frames = frames
                            elif ext.lower() == "txt":
                                task.decoded_result = ("text", decoded_text)
                            else:
                                temp_dir = os.path.join(os.path.dirname(__file__), "temp")
                                os.makedirs(temp_dir, exist_ok=True)
                                temp_file = os.path.join(temp_dir, f"decoded_{task_id}.{ext}")
                                with open(temp_file, "wb") as f:
                                    f.write(raw_data)
                                task.decoded_result = ("file", temp_file)
                            break
                    except Exception as e:
                        logger.error(f"隐写检测失败 (图片{idx+1}): {e}")
                        continue

                # 发送结果
                try:
                    await self._send_result_safe(task, bot)
                except Exception:
                    logger.exception(f"发送结果异常 task_id={task_id}")
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=f"✅ 任务 #{task_id} 已完成\n提示词: {task.prompt}"
                        )
                    except:
                        pass

                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error_message = "任务处理超时"
                task.completed_at = datetime.now()
                logger.error(f"任务 {task_id} 处理超时")
                try:
                    await bot.send_message(chat_id=chat_id, text=f"❌ 任务 #{task_id} 超时")
                except:
                    pass

            except RunningHubError as e:
                task.status = TaskStatus.FAILED
                task.error_message = f"RunningHub错误: {e}"
                task.completed_at = datetime.now()
                logger.exception(f"任务 {task_id} RunningHub错误")
                try:
                    await bot.send_message(chat_id=chat_id, text=f"❌ 任务 #{task_id} 失败: {e}")
                except:
                    pass

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = f"处理错误: {str(e)}"
                task.completed_at = datetime.now()
                logger.exception(f"任务 {task_id} 处理异常")
                try:
                    await bot.send_message(chat_id=chat_id, text=f"❌ 任务 #{task_id} 失败: {e}")
                except:
                    pass
                
            finally:
                self._running_count -= 1
                self.mark_active()
    
    @staticmethod
    def _compress_image(image_bytes: bytes, max_mb: float = 0.1) -> bytes:
        """压缩图片为 JPEG，目标 100KB 以下"""
        max_size = int(max_mb * 1024 * 1024)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            # 宽高限制 1920
            w, h = img.size
            if w > 1920 or h > 1920:
                ratio = 1920 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)
            for quality in range(85, 9, -10):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                compressed = buf.getvalue()
                if len(compressed) <= max_size or quality <= 15:
                    logger.info(f"图片压缩: {len(image_bytes)//1024}KB → {len(compressed)//1024}KB (quality={quality})")
                    return compressed
            return compressed
        except Exception as e:
            logger.error(f"图片压缩失败: {e}")
            return image_bytes

    async def _send_result_safe(self, task: GenerationTask, bot):
        """发送一个媒体组，失败只发文字"""
        from telegram import InputMediaPhoto

        chat_id = task.chat_id
        caption = (task.prompt or "")[:1024]

        raw_images: list[bytes] = []
        if task.video_frames:
            raw_images = task.video_frames[:2]
        elif task.decoded_result and task.decoded_result[0] == "image":
            with open(task.decoded_result[1], "rb") as f:
                raw_images = [f.read()]
        elif task.cached_images:
            raw_images = task.cached_images

        if not raw_images:
            await bot.send_message(chat_id=chat_id, text=f"✓ 任务 #{task.task_id} 已完成")
            return

        compressed = [self._compress_image(img, max_mb=0.5) for img in raw_images]
        logger.info(f"[结果] {len(compressed)}图 大小={[len(x)//1024 for x in compressed]}KB")
        try:
            media = []
            for idx, data in enumerate(compressed):
                c = caption if idx == 0 else None
                media.append(InputMediaPhoto(data, caption=c))
            await bot.send_media_group(chat_id=chat_id, media=media, read_timeout=120, write_timeout=120)
            logger.info(f"[结果] 媒体组成功 task_id={task.task_id}")
        except Exception as e:
            logger.error(f"[结果] 媒体组失败: {e}")
            try:
                await bot.send_message(chat_id=chat_id, text=f"✓ 任务 #{task.task_id} 已完成")
            except:
                pass


task_queue = TaskQueue(max_concurrent=20)


def _load_config() -> RunningHubConfig:
    api = os.environ.get("RUNNINGHUB_API_KEY", "").strip()
    wf = os.environ.get("RUNNINGHUB_WORKFLOW_ID", "").strip()
    if not api or not wf:
        raise SystemExit("Set RUNNINGHUB_API_KEY and RUNNINGHUB_WORKFLOW_ID in .env")
    return RunningHubConfig(
        api_key=api,
        workflow_id=wf,
        base_url=os.environ.get("RUNNINGHUB_BASE_URL", "https://www.runninghub.ai").strip(),
        access_password=(os.environ.get("RUNNINGHUB_ACCESS_PASSWORD") or "").strip() or None,
        load_image_node_id=os.environ.get("RUNNINGHUB_LOAD_IMAGE_NODE_ID", "16").strip(),
        load_image_field=os.environ.get("RUNNINGHUB_LOAD_IMAGE_FIELD", "image").strip(),
        prompt_node_id=os.environ.get("RUNNINGHUB_PROMPT_NODE_ID", "5").strip(),
        prompt_field=os.environ.get("RUNNINGHUB_PROMPT_FIELD", "prompt").strip(),
    )


def _get_rh_client(context: ContextTypes.DEFAULT_TYPE) -> RunningHubClient:
    if context.bot_data.get("rh_client") is None:
        context.bot_data["rh_client"] = RunningHubClient(context.bot_data["rh_cfg"])
    return context.bot_data["rh_client"]


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "欢迎使用 AI 生图 Bot！\n\n"
        "📝 使用方法：\n"
        "1. 发送一张图片\n"
        "2. 输入提示词\n"
        "3. Bot 会在后台处理，完成后自动通知你\n\n"
        "✨ 特性：\n"
        "- 支持 DuckDuckGoose 隐写术检测与自动解码\n"
        "- 即时响应，后台处理\n"
        "- 可同时提交多个任务\n\n"
        "📋 命令：\n"
        "/status - 查看任务状态\n"
        "/cancel - 取消当前任务\n"
        "/tasks - 查看所有任务\n"
        "/help - 显示帮助"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "📖 帮助信息\n\n"
        "发送图片后输入提示词，Bot 会帮你生成新图片。\n"
        "如果生成的图片包含隐藏信息（如 DuckDuckGoose 隐写），"
        "Bot 会自动检测并解码。\n\n"
        "其他命令：\n"
        "/status - 查看当前任务状态\n"
        "/cancel [任务ID] - 取消指定任务\n"
        "/tasks - 查看所有任务列表\n"
        "/help - 显示帮助"
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    
    tasks = await task_queue.get_user_tasks(update.effective_user.id)
    
    if not tasks:
        await update.message.reply_text("当前没有任务。")
        return
    
    status_text = "📊 任务状态：\n\n"
    
    for task in sorted(tasks, key=lambda t: t.created_at, reverse=True)[:15]:
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.PROCESSING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫"
        }.get(task.status, "❓")
        
        status_text += f"{status_emoji} 任务 #{task.task_id}\n"
        status_text += f"   状态: {task.status.value}\n"
        status_text += f"   提示词: {task.prompt[:30]}...\n"
        
        if task.started_at:
            duration = (task.completed_at or datetime.now()) - task.started_at
            status_text += f"   耗时: {duration.total_seconds():.1f}秒\n"
        
        if task.error_message:
            status_text += f"   错误: {task.error_message}\n"
        
        status_text += "\n"
    
    await update.message.reply_text(status_text)


async def tasks_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    
    tasks = await task_queue.get_user_tasks(update.effective_user.id)
    
    if not tasks:
        await update.message.reply_text("当前没有任务。")
        return
    
    keyboard = []
    for task in sorted(tasks, key=lambda t: t.created_at, reverse=True)[:10]:
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.PROCESSING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫"
        }.get(task.status, "❓")
        
        keyboard.append([
            InlineKeyboardButton(
                f"{status_emoji} #{task.task_id} - {task.prompt[:20]}...",
                callback_data=f"task_{task.task_id}"
            )
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("最近的任务：", reply_markup=reply_markup)


async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    
    args = context.args
    user_id = update.effective_user.id
    
    if args:
        task_id = args[0]
        success = await task_queue.cancel_task(task_id)
        if success:
            task = await task_queue.get_task(task_id)
            if task and task.user_id == user_id:
                await update.message.reply_text(f"任务 #{task_id} 已取消。")
            else:
                await update.message.reply_text("该任务不属于你或不存在。")
        else:
            await update.message.reply_text("取消失败，任务可能已完成或不存在。")
    else:
        tasks = await task_queue.get_user_tasks(user_id)
        pending_tasks = [
            t for t in tasks
            if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING)
        ]
        
        if not pending_tasks:
            await update.message.reply_text("没有正在处理的任务可取消。")
            return
        
        keyboard = []
        for task in pending_tasks:
            keyboard.append([
                InlineKeyboardButton(
                    f"取消 #task.task_id: {task.prompt[:30]}...",
                    callback_data=f"cancel_{task.task_id}"
                )
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("选择要取消的任务：", reply_markup=reply_markup)


async def on_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    
    await query.answer()
    
    user_id = query.from_user.id
    
    if query.data.startswith("cancel_"):
        task_id = query.data[7:]
        task = await task_queue.get_task(task_id)
        
        if task and task.user_id == user_id:
            success = await task_queue.cancel_task(task_id)
            if success:
                await query.edit_message_text(f"任务已取消。")
            else:
                await query.edit_message_text("取消失败。")
        else:
            await query.edit_message_text("该任务不属于你或不存在。")
    
    elif query.data.startswith("task_"):
        task_id = query.data[5:]
        task = await task_queue.get_task(task_id)
        
        if not task:
            await query.edit_message_text("任务不存在。")
            return
        
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.PROCESSING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫"
        }.get(task.status, "❓")
        
        text = f"任务详情：\n"
        text += f"ID: {task.task_id}\n"
        text += f"状态: {status_emoji} {task.status.value}\n"
        text += f"提示词: {task.prompt}\n"
        
        if task.started_at:
            duration = (task.completed_at or datetime.now()) - task.started_at
            text += f"耗时: {duration.total_seconds():.1f}秒\n"
        
        if task.error_message:
            text += f"错误: {task.error_message}\n"
        
        if task.decoded_text:
            text += f"\n🔓 解码内容: {task.decoded_text}\n"
        
        if task.result_urls:
            text += f"\n📷 结果: {task.result_urls[0]}"
        
        keyboard = []
        if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
            keyboard.append([
                InlineKeyboardButton("取消任务", callback_data=f"cancel_{task.task_id}")
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        await query.edit_message_text(text, reply_markup=reply_markup)


async def _download_and_upload_background(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    photo_obj,
    name: str
):
    """后台下载 Telegram 图片并上传到 RunningHub"""
    # 提前捕获，防止后台任务中 update 过期
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    msg_id = update.message.id if update.message else None

    try:
        # 后台下载 Telegram 图片
        tg_file = await photo_obj.get_file()
        buf = await tg_file.download_as_bytearray()

        # 保存原始图片字节
        img_bytes = bytes(buf)

        # 后台上传到 RunningHub
        rh = _get_rh_client(context)
        rh_name = await asyncio.wait_for(
            rh.upload_image(img_bytes, name),
            timeout=30.0
        )

        # 检查是否有待处理的提示词！（不 pop，允许多个上传任务共用）
        pending_prompt = context.user_data.get(PENDING_PROMPT)
        if pending_prompt:
            # 立即用已有的提示词创建任务
            global task_counter
            task_counter += 1
            task_id = str(task_counter)
            task = GenerationTask(
                task_id=task_id,
                user_id=user_id,
                chat_id=chat_id,
                message_id=msg_id,
                rh_image_name=rh_name,
                prompt=pending_prompt,
                original_image_bytes=img_bytes,
            )
            await task_queue.add_task(task)
            asyncio.create_task(task_queue.process_task(task_id, rh, context))
            logger.info(f"自动创建任务 #{task_id}（上传完成，提示词已就绪）")
            # 如果没有更多待处理图和上传中的图，清理提示词
            if not context.user_data.get(PENDING_RH_IMAGES) and UPLOADING_MESSAGE not in context.user_data:
                context.user_data.pop(PENDING_PROMPT, None)
        else:
            # 追加到待处理图片列表，等待提示词
            pending_list = context.user_data.setdefault(PENDING_RH_IMAGES, [])
            pending_list.append({"rh_name": rh_name, "image_bytes": img_bytes})
            if len(pending_list) == 1:
                uploading_msg_id = context.user_data.pop(UPLOADING_MESSAGE, None)
                if uploading_msg_id:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=uploading_msg_id,
                        text=f"✅ {len(pending_list)}张图片已处理完毕！\n\n请直接回复提示词（文字），我将开始生成图片。\n"
                             "💡 发送多张图片后只需输入一次提示词即可同时处理。"
                    )
            else:
                uploading_msg_id = context.user_data.get(UPLOADING_MESSAGE)
                if uploading_msg_id and isinstance(uploading_msg_id, int):
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=uploading_msg_id,
                        text=f"✅ {len(pending_list)}张图片已处理完毕！\n\n请直接回复提示词（文字），我将开始生成图片。\n"
                             "💡 发送多张图片后只需输入一次提示词即可同时处理。"
                    )
    except Exception as e:
        logger.exception("后台处理失败")
        context.user_data.pop(PENDING_PROMPT, None)
        uploading_msg_id = context.user_data.pop(UPLOADING_MESSAGE, None)
        if uploading_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=uploading_msg_id,
                text=f"❌ 处理失败：{str(e)[:100]}，请重试。"
            )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        return
    
    # 新照片到来，清掉旧提示词（新图需要新提示词）
    context.user_data.pop(PENDING_PROMPT, None)

    # 媒体组：只对第一张回复"图片已收到"，其余静默处理
    mg_id = update.message.media_group_id
    if mg_id:
        seen = context.application.bot_data.setdefault("_seen_media_groups", {})
        if mg_id in seen:
            context.user_data[UPLOADING_MESSAGE] = True  # 占位，不回复
        else:
            seen[mg_id] = True
            context.user_data[UPLOADING_MESSAGE] = True
            msg = await update.message.reply_text("✅ 图片已收到！正在处理中...")
            context.user_data[UPLOADING_MESSAGE] = msg.message_id
    else:
        context.user_data[UPLOADING_MESSAGE] = True
        msg = await update.message.reply_text("✅ 图片已收到！正在处理中...")
        context.user_data[UPLOADING_MESSAGE] = msg.message_id

    # 2. 准备数据
    photo = update.message.photo[-1]
    name = f"tg_{update.effective_user.id}_{photo.file_unique_id}.jpg"

    # 3. 完全后台处理（下载+上传都在后台）
    asyncio.create_task(_download_and_upload_background(update, context, photo, name))


async def on_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document or not update.effective_user:
        return
    
    doc = update.message.document
    mime = (doc.mime_type or "").lower()
    if not mime.startswith("image/"):
        return
    
    # 新照片到来，清掉旧提示词
    context.user_data.pop(PENDING_PROMPT, None)
    # 提前占位，防止 await 期间 on_text 看不到 UPLOADING_MESSAGE
    context.user_data[UPLOADING_MESSAGE] = True
    msg = await update.message.reply_text("✅ 图片已收到！正在处理中...")
    context.user_data[UPLOADING_MESSAGE] = msg.message_id

    # 2. 准备数据
    suffix = Path(doc.file_name or "image").suffix or ".png"
    name = f"tg_{update.effective_user.id}_{doc.file_unique_id}{suffix}"

    # 3. 完全后台处理（下载+上传都在后台）
    asyncio.create_task(_download_and_upload_background(update, context, doc, name))


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global task_counter
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if text.startswith("/"):
        return
    
    # 检查是否还在上传中！如果是，就保存提示词！
    if UPLOADING_MESSAGE in context.user_data:
        context.user_data[PENDING_PROMPT] = text

        # 同时处理已完成上传的图片
        done_list = context.user_data.pop(PENDING_RH_IMAGES, None)
        if done_list:
            rh = _get_rh_client(context)
            task_ids = []
            for entry in done_list:
                task_counter += 1
                task_id = str(task_counter)
                task = GenerationTask(
                    task_id=task_id,
                    user_id=update.effective_user.id,
                    chat_id=update.message.chat_id,
                    message_id=update.message.id,
                    rh_image_name=entry["rh_name"],
                    prompt=text,
                    original_image_bytes=entry["image_bytes"],
                )
                await task_queue.add_task(task)
                asyncio.create_task(task_queue.process_task(task_id, rh, context))
                task_ids.append(task_id)
            await update.message.reply_text(
                f"✅ {len(task_ids)}个任务已提交！（剩余图片上传中，将自动处理）\n\n"
                f"提示词: {text[:50]}{'...' if len(text) > 50 else ''}",
                reply_to_message_id=update.message.id
            )
        else:
            await update.message.reply_text(
                f"✅ 提示词已保存！等图片上传完成后自动开始处理。\n\n提示词: {text[:50]}{'...' if len(text) > 50 else ''}",
                reply_to_message_id=update.message.id
            )
        return
    
    pending_list = context.user_data.pop(PENDING_RH_IMAGES, None)
    if not pending_list:
        await update.message.reply_text(
            "请先发送一张图片作为参考图。",
            reply_to_message_id=update.message.id
        )
        return

    # 为每张待处理的图片创建任务
    rh = _get_rh_client(context)
    task_ids = []
    for entry in pending_list:
        task_counter += 1
        task_id = str(task_counter)
        task = GenerationTask(
            task_id=task_id,
            user_id=update.effective_user.id,
            chat_id=update.message.chat_id,
            message_id=update.message.id,
            rh_image_name=entry["rh_name"],
            prompt=text,
            original_image_bytes=entry["image_bytes"],
        )
        await task_queue.add_task(task)
        asyncio.create_task(task_queue.process_task(task_id, rh, context))
        task_ids.append(task_id)

    count = len(task_ids)
    await update.message.reply_text(
        f"✅ {count}个任务已提交！\n\n"
        f"提示词: {text[:50]}{'...' if len(text) > 50 else ''}\n\n"
        f"⏳ 后台处理中，完成后会自动通知你。\n"
        f"你可以继续发送其他图片或提示词。\n\n"
        f"任务ID: {', '.join(f'#{tid}' for tid in task_ids)}\n"
        f"查看任务状态: /status"
    )


async def _keep_alive(context: ContextTypes.DEFAULT_TYPE) -> None:
    """保活：空闲超过14分钟时发一个轻量任务暖暖模型（已关闭）"""
    if task_queue.idle_seconds < 840:
        return
    if task_queue._running_count > 0:
        return
    logger.info("保活：模型可能已凉，发送暖机任务...")
    try:
        from PIL import Image as PILImage
        import io as _io
        tiny = _io.BytesIO()
        PILImage.new("RGB", (1, 1)).save(tiny, format="JPEG")
        rh = context.bot_data.get("rh_client") or RunningHubClient(context.bot_data["rh_cfg"])
        rh_name = await asyncio.wait_for(
            rh.upload_image(tiny.getvalue(), "keepalive.jpg"),
            timeout=30.0,
        )
        task_id_rh = await asyncio.wait_for(
            rh.create_task(rh_name, "keepalive"),
            timeout=30.0,
        )
        await asyncio.wait_for(
            rh.wait_for_outputs(task_id_rh),
            timeout=300.0,
        )
        task_queue.mark_active()
        logger.info("保活完成")
    except Exception as e:
        logger.error(f"保活失败: {e}")


async def post_init(application: Application) -> None:
    application.bot_data["rh_cfg"] = _load_config()
    application.bot_data["rh_client"] = RunningHubClient(application.bot_data["rh_cfg"])

    # 保活已关闭（注释掉下面这行即可重新开启）
    # application.job_queue.run_repeating(_keep_alive, interval=900, first=300)

    # 每 10 分钟清理已完成超过 30 分钟的任务
    async def _cleanup_old_tasks(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await task_queue.remove_completed_tasks(older_than_minutes=30)

    application.job_queue.run_repeating(_cleanup_old_tasks, interval=600, first=60)

    commands = [
        BotCommand("start", "开始使用，显示欢迎信息"),
        BotCommand("help", "显示帮助信息"),
        BotCommand("status", "查看任务状态"),
        BotCommand("tasks", "查看所有任务列表"),
        BotCommand("cancel", "取消任务"),
    ]
    await application.bot.set_my_commands(commands)

    logger.info("Bot 初始化完成")


async def post_shutdown(application: Application) -> None:
    client = application.bot_data.get("rh_client")
    if isinstance(client, RunningHubClient):
        await client.aclose()
    logger.info("Bot 已关闭")


def main() -> None:
    bootstrap_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        env_path = project_root() / ".env"
        raise SystemExit(
            "缺少 TELEGRAM_BOT_TOKEN。\n"
            f"1) 在文件夹里放好配置文件（推荐文件名：{env_path.name}，路径：{env_path}）。\n"
            "2) 变量名须为 TELEGRAM_BOT_TOKEN=你的token（中间是下划线，不要空格）。\n"
            "   若你习惯写成「TELEGRAM BOT TOKEN=…」，保存为 .env 后本程序也会自动识别。\n"
            "3) 虚拟环境路径是 .venv（前面有个点），运行： .\\.venv\\Scripts\\python run.py\n"
            f"当前工作目录：{Path.cwd()}"
        )

    app = (
        Application.builder()
        .token(token)
        .job_queue(JobQueue())
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    # /setmenu：手动为当前群/话题设置命令菜单
    async def setmenu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_type = update.effective_chat.type if update.effective_chat else "private"
        if chat_type == "private":
            await update.message.reply_text("在群里发这个命令才有用。")
            return
        from telegram import BotCommandScopeChat
        cmds = [
            BotCommand("start", "开始使用"),
            BotCommand("help", "显示帮助"),
            BotCommand("status", "查看任务状态"),
            BotCommand("tasks", "查看所有任务"),
            BotCommand("cancel", "取消任务"),
            BotCommand("setmenu", "设置群菜单"),
        ]
        try:
            await context.bot.set_my_commands(
                cmds,
                scope=BotCommandScopeChat(chat_id=update.effective_chat.id),
            )
            await update.message.reply_text("✅ 菜单已设置")
        except Exception as e:
            await update.message.reply_text(f"❌ 设置失败: {e}")

    app.add_handler(CommandHandler("setmenu", setmenu_cmd))
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("tasks", tasks_cmd))
    app.add_handler(CallbackQueryHandler(on_callback_query))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # 被拉入新群/话题时自动设置命令菜单
    from telegram import BotCommandScopeChat
    from telegram.ext import ChatMemberHandler

    async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.my_chat_member:
            return
        new_status = update.my_chat_member.new_chat_member.status
        if new_status in ("member", "administrator"):
            cmds = [
                BotCommand("start", "开始使用"),
                BotCommand("help", "显示帮助"),
                BotCommand("status", "查看任务状态"),
                BotCommand("tasks", "查看所有任务"),
                BotCommand("cancel", "取消任务"),
            ]
            try:
                await context.bot.set_my_commands(
                    cmds,
                    scope=BotCommandScopeChat(chat_id=update.effective_chat.id),
                )
                logger.info(f"已为群 {update.effective_chat.id} 设置命令菜单")
            except Exception as e:
                logger.error(f"设置群命令菜单失败: {e}")

    app.add_handler(ChatMemberHandler(on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))

    logger.info("Bot 启动中...")
    logger.info("功能说明：")
    logger.info("- DuckDuckGoose 隐写术检测与自动解码")
    logger.info("- 即时响应 + 后台异步处理")
    logger.info("- 并发任务队列（最多10个任务同时处理）")
    logger.info("- 完善的错误处理与超时机制")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
