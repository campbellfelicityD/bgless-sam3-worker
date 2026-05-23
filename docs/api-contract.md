# SAM3 + MatAnyone RunPod 处理器 — 输入/输出契约

> **状态**: v0.1 草案 (2026-05-23)
> **作用**: bgless 后端 `dispatchSAM3()` ↔ RunPod Serverless handler 之间的稳定合约
> **更新规则**: 改动需向后兼容；不兼容改动 → bump `version` 字段 + 后端 dispatcher 双跑

---

## 0. 端点

| 用途 | URL | 鉴权 |
|---|---|---|
| 同步推理（≤30s 任务，preview 用） | `POST https://api.runpod.ai/v2/{endpoint_id}/runsync` | `Authorization: Bearer ${RUNPOD_API_KEY}` |
| 异步推理（完整视频） | `POST https://api.runpod.ai/v2/{endpoint_id}/run` | 同上 |
| 状态查询 | `GET  https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}` | 同上 |
| 取消 | `POST https://api.runpod.ai/v2/{endpoint_id}/cancel/{job_id}` | 同上 |
| Webhook（推荐，省轮询） | RunPod 调用我们的 `POST {BGLESS_API}/api/runpod/webhook` | 用 HMAC 共享密钥校验 |

---

## 1. 输入 schema (`input` 字段)

RunPod 要求所有自定义参数包在 `{"input": {...}}` 里。

```jsonc
{
  "input": {
    // ─── 必填 ─────────────────────────────────────────────────
    "version": "1",                       // 合约版本，handler 用它做向后兼容
    "video_url": "https://...",           // 源视频 R2 签名 URL；handler 不能也不应保存域名/凭证
    "job_id": "bgl_abc123",               // bgless 端业务 job id (回写日志、debug 用)

    // ─── 模型选择 ─────────────────────────────────────────────
    "model": "sam3-pro",                  // 见 §2 模型档位表
    "preview": false,                     // true = 只跑前 N 秒 (preview_duration)，跳过 matting refine
    "preview_duration": 2.0,              // 秒，preview=true 时生效

    // ─── 主体选择 (SAM3 prompts) ─────────────────────────────
    "prompt": {
      "mode": "auto",                     // "auto" | "text" | "box" | "point" | "mask"
      "text": "the person",               // mode=text；自然语言概念，已过 moderation
      "box":  [[x1,y1,x2,y2], ...],       // mode=box；归一化 0-1 坐标，多个 = 多对象
      "points": [                         // mode=point；第一帧标点
        {"x": 0.5, "y": 0.5, "label": 1}, // label: 1=前景 0=背景
        {"x": 0.2, "y": 0.8, "label": 0}
      ],
      "mask_url": "https://...",          // mode=mask；外部预生成 mask 序列 (zip of PNG)
      "frame_index": 0,                   // box/point/mask 对应的参考帧索引
      "negative_text": null               // 可选：text 模式下排除概念，如 "the shadow"
    },

    // ─── 输出格式 ─────────────────────────────────────────────
    "output": {
      "format": "webm",                   // "webm" | "mov" | "mp4" | "gif" | "webp" | "png_sequence"
      "max_dimension": 1920,              // 输出最长边，超过则等比缩放
      "fps": null,                        // null = 保持源；否则 12/24/30/60
      "preserve_audio": true,             // 是否保留原音轨 (webm/mov/mp4 有效)
      "quality": "high"                   // "draft" | "standard" | "high" | "lossless"
    },

    // ─── 背景合成 ─────────────────────────────────────────────
    "background": {
      "type": "transparent",              // "transparent" | "color" | "image" | "video"
      "color": [0.0, 0.0, 0.0],           // type=color, RGB 0-1
      "image_url": null,                  // type=image, R2 URL
      "video_url": null,                  // type=video, R2 URL，循环或裁剪到主视频长度
      "fit": "cover",                     // image/video 适配：cover/contain/stretch
      "blur_original": false              // type=image/video 时是否同时模糊原背景做光晕
    },

    // ─── 高级 ────────────────────────────────────────────────
    "refine": {
      "matting_model": "matanyone",       // "none" | "guided_filter" | "matanyone"；见 §2
      "edge_smoothing": 0.5,              // 0-1，guided filter 半径 / matanyone temporal weight
      "feather_px": 0,                    // 后处理 alpha 通道羽化像素
      "trimap_dilate": 8,                 // SAM3 mask → trimap 的 unknown 带宽
      "stability_threshold": 0.9          // SAM3 mask score 阈值；低于这个值帧标记为低置信
    },

    // ─── 回调 ────────────────────────────────────────────────
    "callback_url": null,                 // 可选；handler 完成时 POST 这里 (HMAC 签名)
    "progress_callback": false            // true = 每 5% 进度 POST 一次 callback_url
  }
}
```

### 1.1 字段约束

| 字段 | 约束 | 失败行为 |
|---|---|---|
| `video_url` | 必须 HTTPS；HEAD 请求 `Content-Length` ≤ 500 MB | 返回 `ERR_VIDEO_TOO_LARGE` |
| `video_url` | 必须能在 30s 内下载完 | 返回 `ERR_DOWNLOAD_TIMEOUT` |
| `prompt.text` | ≤ 200 chars，UTF-8；调用前后端必须先过 moderation | handler 信任已过审 |
| `prompt.box` | 每个 box `x2>x1 & y2>y1`，且 `0 ≤ * ≤ 1` | 返回 `ERR_INVALID_PROMPT` |
| `output.max_dimension` | 256–4096 | 截断到范围内 |
| `output.fps` | 6–120 或 null | 默认保持源 |
| `background.image_url` | 同 video_url 校验 | `ERR_DOWNLOAD_BACKGROUND_FAILED` |

### 1.2 默认值（handler 内部补齐）

省略字段按这套默认补：
```jsonc
{
  "version": "1",
  "model": "sam3-pro",
  "preview": false,
  "preview_duration": 2.0,
  "prompt": { "mode": "auto", "frame_index": 0 },
  "output": { "format": "webm", "max_dimension": 1920, "fps": null,
              "preserve_audio": true, "quality": "high" },
  "background": { "type": "transparent", "fit": "cover", "blur_original": false },
  "refine": { "matting_model": "matanyone", "edge_smoothing": 0.5,
              "feather_px": 0, "trimap_dilate": 8, "stability_threshold": 0.9 }
}
```

---

## 2. 模型档位表（`model` 字段合法值）

| `model` | 内部 pipeline | 显存 | 速度（1080p/30fps/10s） | 目标 SKU |
|---|---|---|---|---|
| `sam3-tiny` | SAM3-Tiny → guided filter | ~6 GB | ~12 s | preview / light 档 |
| `sam3-base` | SAM3-Base → guided filter | ~10 GB | ~25 s | standard 档 |
| `sam3-pro` | SAM3-Large → trimap → MatAnyone | ~22 GB | ~80 s | **Pro 档（主推）** |
| `sam3-human` | SAM3-Base (auto, person prior) → MatAnyone | ~14 GB | ~35 s | human 档 |
| `rvm-light` | RVM MobileNetV3 ONNX（CPU 兜底） | 0 (CPU) | ~8 s | fallback / 极简档 |

> `refine.matting_model` 与 `model` 的关系：`sam3-pro` 强制 `matanyone`；其他档可以被 input 覆盖（如设 `none` 跳过 refine 出更快但更糙的结果）。

---

## 3. 输出 schema

### 3.1 成功（HTTP 200 / RunPod status=COMPLETED）

```jsonc
{
  "output": {
    "version": "1",
    "job_id": "bgl_abc123",                  // 回传业务 job_id

    "result": {
      "output_url": "https://cdn.removebgvideo.com/outputs/bgl_abc123.webm",
      "output_format": "webm",
      "duration_seconds": 12.34,
      "frame_count": 370,
      "width": 1920,
      "height": 1080,
      "file_size_bytes": 8421376,
      "has_alpha": true,
      "preview": false                       // 是否是 preview 输出
    },

    "stats": {
      "pipeline_ms": {
        "download": 1234,
        "sam3_inference": 18500,
        "matting": 12300,
        "compose_encode": 9800,
        "upload": 1800,
        "total": 43632
      },
      "gpu_peak_memory_mb": 19800,
      "model_version": "sam3-large@1.0.0+matanyone@0.3.1",
      "low_confidence_frames": [12, 13, 14], // stability_threshold 之下的帧索引
      "warnings": [
        "Frame 47: multiple subjects detected; using highest-score mask"
      ]
    },

    "debug": {                               // 仅 BGLESS_DEBUG=1 时返回
      "first_frame_mask_url": "https://...",
      "trimap_sample_url": "https://..."
    }
  }
}
```

### 3.2 失败（RunPod status=FAILED）

```jsonc
{
  "error": "ERR_DOWNLOAD_TIMEOUT",          // 见 §4
  "message": "Failed to download video_url within 30s",
  "retryable": true,
  "details": {
    "url_host": "r2.example.com",
    "elapsed_ms": 30123
  }
}
```

### 3.3 进度回调（启用 `progress_callback` 时）

每 5% 进度向 `callback_url` POST：
```jsonc
{
  "job_id": "bgl_abc123",
  "stage": "sam3_inference",                 // download | sam3_inference | matting | compose_encode | upload
  "progress": 0.45,                          // 0–1
  "eta_seconds": 18,
  "timestamp": "2026-05-23T14:30:12Z"
}
```
Header `X-Bgless-Signature: sha256=<hmac>` 校验。

---

## 4. 错误码表

| code | retryable | 含义 | 处理建议 |
|---|---|---|---|
| `ERR_VIDEO_TOO_LARGE` | ❌ | 源 > 500 MB | 前端拒绝上传 |
| `ERR_VIDEO_TOO_LONG` | ❌ | 源 > 60 min | 前端拒绝 |
| `ERR_DOWNLOAD_TIMEOUT` | ✅ | 30s 内拉不到源 | bgless 重试 1 次 |
| `ERR_DOWNLOAD_FAILED` | ✅ | HTTP 4xx/5xx | 重试 1 次 |
| `ERR_DOWNLOAD_BACKGROUND_FAILED` | ✅ | 背景图/视频拉不到 | 重试 1 次 |
| `ERR_INVALID_PROMPT` | ❌ | prompt 字段格式错 | 前端校验补齐 |
| `ERR_INVALID_INPUT` | ❌ | 其他字段约束失败 | 前端校验 |
| `ERR_NO_SUBJECT_FOUND` | ❌ | SAM3 找不到任何符合 prompt 的对象 | 引导用户改 prompt |
| `ERR_MODEL_LOAD_FAILED` | ✅ | 权重缺失/损坏 | 报警 + 重启容器 |
| `ERR_GPU_OOM` | ✅ | 显存溢出 | 自动降一档模型重试 1 次 |
| `ERR_FFMPEG_ENCODE_FAILED` | ✅ | 编码失败 | 重试 1 次 |
| `ERR_UPLOAD_FAILED` | ✅ | 写 R2 失败 | 重试 2 次 |
| `ERR_UNKNOWN` | ✅ | 兜底 | 报警 + 人工 |

后端 dispatcher 看到 `retryable=true` 时按指数退避重试，最多 3 次。

---

## 5. Webhook（推荐方案，省轮询费）

### bgless → RunPod 创建任务时附带：
```json
{
  "input": { /* §1 */ },
  "webhook": "https://api.bgless.com/api/runpod/webhook"
}
```

### RunPod → bgless 回调（任务终态时）：
```jsonc
{
  "id": "runpod-job-uuid",
  "status": "COMPLETED",                     // COMPLETED | FAILED | CANCELLED
  "output": { /* §3.1 或 §3.2 的整个 output 对象 */ },
  "executionTime": 43632,                    // ms
  "delayTime": 1200                          // 排队时长 ms
}
```

我们端：
- `POST /api/runpod/webhook` 校验 `X-RunPod-Signature` (HMAC-SHA256，secret 来自 RunPod 控制台)
- 根据 `output.output.job_id` 找到业务 job → 更新 status → 通知前端（SSE 或下次轮询返回）

---

## 6. 环境变量（handler 端）

handler 不接受任何业务凭证从 input 传入；所有敏感配置走 RunPod endpoint 的环境变量：

```bash
# 必需
HF_TOKEN=hf_xxx                              # 拉 SAM3 权重 (gated)
SAM3_WEIGHTS_DIR=/runpod-volume/sam3         # network volume，避免每次冷启动重下
MATANYONE_WEIGHTS_DIR=/runpod-volume/matanyone

# R2 输出 bucket
R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_OUTPUT_BUCKET=bgless-sam3-outputs
R2_PUBLIC_BASE=https://cdn.bgless.com       # 拼成公开 URL 返回

# 可选
BGLESS_DEBUG=0                               # 1 = 返回 debug.* 字段
WEBHOOK_HMAC_SECRET=...                      # 签 callback_url 用
SENTRY_DSN=...                               # 异常上报
```

---

## 7. 版本演进

| version | 时间 | 变更 |
|---|---|---|
| `1` | 2026-05-23 | 初版：sam3-tiny/base/pro/human + rvm-light，5 种 prompt 模式 |

不兼容改动：bump version，handler 同时支持 N 和 N-1，老调用方不受影响。

---

## 8. 与 bgless 后端 dispatcher 的对应关系

后端 `workers/transcoder/index.ts` 里：

```ts
// 待添加：dispatchSAM3.ts
type SAM3Input = z.infer<typeof SAM3InputSchema>;  // 用 zod 校验
async function dispatchSAM3(input: SAM3Input): Promise<{ runpod_job_id: string }> {
  const body = { input, webhook: env.BGLESS_WEBHOOK_URL };
  const r = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_SAM3_ENDPOINT}/run`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${env.RUNPOD_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`RunPod dispatch failed: ${r.status}`);
  const { id } = await r.json();
  return { runpod_job_id: id };
}
```

zod schema 见 `workers/transcoder/sam3-schema.ts`（待创建，与本文档 §1 字段一一对应）。

---

## 附录 A：与 RemoveBGVideo `/api/process` 字段对照

| RemoveBGVideo | bgless SAM3 input |
|---|---|
| `video_url` | `video_url` |
| `model: "original"\|"light"\|"pro"\|"human"` | `model: "sam3-base"\|"sam3-tiny"\|"sam3-pro"\|"sam3-human"` |
| `text_prompt` | `prompt.text` (mode=text) |
| `bg_type` | `background.type` |
| `bg_color` | `background.color` |
| `bg_image_url` / `bg_video_url` | `background.image_url` / `video_url` |
| `output_format` | `output.format` |
| `edge_smoothing` | `refine.edge_smoothing` |
| `quality` | `output.quality` |
| `is_preview` / `preview_duration` | `preview` / `preview_duration` |
| —（无） | `prompt.box` / `prompt.points` / `prompt.mask_url` ✨ 我们的差异化 |
| —（无） | `refine.matting_model` 显式可控 ✨ |
| —（无） | webhook + HMAC ✨ (他们只有轮询) |

我们差异化抓手：**多模态 prompt + 显式 matting 控制 + webhook 节流**。
