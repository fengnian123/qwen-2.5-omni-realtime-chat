
# 基于fastrtc框架调用qwen-2.5-omni-realtime实现实时语音对话、视频对话等多种功能

<img style="display: block; padding-right: 5px; height: 20px;" alt="Static Badge" src="https://img.shields.io/pypi/v/fastrtc"> 
<a href="https://github.com/fengnian123/qwen-2.5-omni-realtime-chat" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/github-white?logo=github&logoColor=black"></a>

> 基于 Fastrtc 框架与 Qwen-2.5-Omni-Realtime 模型，实现低延迟的实时语音对话、视频对话、多模态交互能力

---

## 🚀 项目概述
本项目集成阿里巴巴通义实验室最新推出的 **Qwen-2.5-Omni-Realtime** 多模态大模型，通过 Fastrtc 实时通信框架实现：
- 语音到文本（STT）实时对话
- 视频流实时分析与响应
- 多模态上下文理解（文本+语音+图像）

## ⚙️ 安装与配置

### 1. 安装依赖
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip install -r requirements.txt
```

### 2. 配置模型API

- 打开`.env`文件，将阿里云百炼API key填入（sk开头）

```python
# .env
Qwen_API_KEY=<your api_key>
```

### 3. 启动服务

- 配置UI模式

```bash
export MODE=UI
```

- 运行代码

```python
# 示例
python test-hello.py
```

## 📝 使用示例

### 实时语音对话

- 运行`test-voice.py`：

```bash
python test-voice.py
```

- 效果：
https://github.com/fengnian123/qwen-2.5-omni-realtime-chat/blob/main/onmi-demo/Audio%20Video%20Chat.mov


### 实时视频对话

- 运行`test-vedio.py`：

```bash
python test-vedio.py
```

- 效果：

  

### 说出特定词语后激活聊天

- 运行`test-hello.py`：

```bash
python test-hello.py
```

- 修改激活词语：在`test-hello.py`文件中修改**stop_words**变量，默认为"computer"（目前仅支持英文）

```python
# test-hello.py
stream = Stream(
    ReplyOnStopWords(
        response,  # type: ignore
        stop_words=["computer"],
        input_sample_rate=16000,
    ),
    mode="send",
    modality="audio",
    additional_inputs=[chatbot, state],
    additional_outputs=[chatbot, state],
    additional_outputs_handler=lambda *a: (a[2], a[3]),
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
)
```

- 效果：

  

### html代码实时语音编写

- 运行`test-code/test-code.py`：

```bash
python test-code/test-code.py
```

- 效果：



### 实时语音转录

- 运行`Whisper-Transcription/app.py`：

```bash
python Whisper-Transcription/app.py
```

- 效果：
