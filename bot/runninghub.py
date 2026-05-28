"""RunningHub ComfyUI OpenAPI client (upload + create + poll outputs)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RunningHubConfig:
    api_key: str
    workflow_id: str
    base_url: str = "https://www.runninghub.ai"
    access_password: str | None = None
    load_image_node_id: str = "16"
    load_image_field: str = "image"
    prompt_node_id: str = "5"
    prompt_field: str = "prompt"
    poll_interval_sec: float = 3.0
    poll_timeout_sec: float = 900.0


class RunningHubError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class RunningHubClient:
    def __init__(self, cfg: RunningHubConfig):
        import os
        self._cfg = cfg
        proxy = (
            os.environ.get("HTTPS_PROXY", "").strip()
            or os.environ.get("https_proxy", "").strip()
            or os.environ.get("ALL_PROXY", "").strip()
        )
        if proxy:
            transport = httpx.AsyncHTTPTransport(proxy=proxy)
            self._client = httpx.AsyncClient(
                base_url=cfg.base_url.rstrip("/"),
                timeout=120.0,
                transport=transport,
                trust_env=True,
            )
        else:
            self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=120.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def upload_image(self, file_bytes: bytes, filename: str) -> str:
        """Upload input image; returns `fileName` for LoadImage node."""
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"apiKey": self._cfg.api_key}
        r = await self._client.post("/task/openapi/upload", files=files, data=data)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise RunningHubError(body.get("msg") or "upload failed", code=body.get("code"))
        fn = (body.get("data") or {}).get("fileName")
        if not fn:
            raise RunningHubError("upload response missing data.fileName")
        return str(fn)

    async def create_task(self, rh_image_filename: str, prompt: str = None, prompts: list[str] = None, extra_node_info: list[dict] | None = None) -> str:
        import os
        node_info_list = [
            {
                "nodeId": self._cfg.load_image_node_id,
                "fieldName": self._cfg.load_image_field,
                "fieldValue": rh_image_filename,
            },
        ]

        # 支持多个提示词（环境变量 RUNNINGHUB_PROMPT_NODE_IDS 指定多个节点ID，逗号分隔）
        prompt_node_ids_str = os.environ.get("RUNNINGHUB_PROMPT_NODE_IDS", self._cfg.prompt_node_id)
        prompt_node_ids = [pid.strip() for pid in prompt_node_ids_str.split(",")]

        if prompts and len(prompts) > 1:
            for i, p in enumerate(prompts):
                if i < len(prompt_node_ids):
                    node_info_list.append({
                        "nodeId": prompt_node_ids[i],
                        "fieldName": self._cfg.prompt_field,
                        "fieldValue": p,
                    })
        elif prompt:
            node_info_list.append({
                "nodeId": self._cfg.prompt_node_id,
                "fieldName": self._cfg.prompt_field,
                "fieldValue": prompt,
            })
        if extra_node_info:
            node_info_list.extend(extra_node_info)
        
        payload: dict[str, Any] = {
            "apiKey": self._cfg.api_key,
            "workflowId": self._cfg.workflow_id,
            "nodeInfoList": node_info_list,
        }
        if self._cfg.access_password:
            payload["accessPassword"] = self._cfg.access_password

        r = await self._client.post(
            "/task/openapi/create",
            json=payload,
            headers={"Host": "www.runninghub.ai"},
        )
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise RunningHubError(body.get("msg") or "create task failed", code=body.get("code"))
        tid = (body.get("data") or {}).get("taskId")
        if tid is None:
            raise RunningHubError("create response missing data.taskId")
        return str(tid)

    async def wait_for_outputs(self, task_id: str) -> list[dict[str, Any]]:
        deadline = asyncio.get_event_loop().time() + self._cfg.poll_timeout_sec
        while asyncio.get_event_loop().time() < deadline:
            r = await self._client.post(
                "/task/openapi/outputs",
                json={"apiKey": self._cfg.api_key, "taskId": task_id},
            )
            r.raise_for_status()
            body = r.json()
            code = body.get("code")
            if code == 0:
                data = body.get("data")
                if isinstance(data, list) and data:
                    return data
                if isinstance(data, list):
                    await asyncio.sleep(self._cfg.poll_interval_sec)
                    continue
                if data is None:
                    await asyncio.sleep(self._cfg.poll_interval_sec)
                    continue
                raise RunningHubError("outputs success but unexpected data shape")
            if code == 804:
                await asyncio.sleep(self._cfg.poll_interval_sec)
                continue
            raise RunningHubError(body.get("msg") or "outputs error", code=code)

        raise RunningHubError("Timed out waiting for RunningHub task outputs")
