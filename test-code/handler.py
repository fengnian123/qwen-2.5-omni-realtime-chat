import base64
import os
import re
from pathlib import Path
from http import HTTPStatus
import dashscope
from dashscope.audio.asr import Recognition
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    get_stt_model,
    audio_to_bytes,
)
load_dotenv()

model = get_stt_model()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("Qwen_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
path = Path(__file__).parent / "assets"

spinner_html = open(path / "spinner.html").read()


system_prompt = "You are an AI coding assistant. Your task is to write single-file HTML applications based on a user's request. Only return the necessary code. Include all necessary imports and styles. You may also be asked to edit your original response."
user_prompt = "Please write a single-file HTML application to fulfill the following request.\nThe message:{user_message}\nCurrent code you have written:{code}"


def extract_html_content(text):
    """
    Extract content including HTML tags.
    """
    match = re.search(r"<!DOCTYPE html>.*?</html>", text, re.DOTALL)
    return match.group(0) if match else None


def display_in_sandbox(code):
    encoded_html = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    data_uri = f"data:text/html;charset=utf-8;base64,{encoded_html}"
    return f'<iframe src="{data_uri}" width="100%" height="600px"></iframe>'


def generate(user_message: tuple[int, np.ndarray], history: list[dict], code: str):
    yield AdditionalOutputs(history, spinner_html)
    text = model.stt(user_message)
    text = "Can you help me write a tic-tac-toe game?"
    print("STT in handler", text)
    # 构造用户提示
    user_msg_formatted = user_prompt.format(user_message=text, code=code)
    history.append({"role": "user", "content": user_msg_formatted})

    # 调用阿里云百炼大模型生成回复
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
            messages=history,
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            # extra_body={"enable_thinking": False},
        )
    print(completion.model_dump_json())

    # 提取模型输出并处理
    output = completion.choices[0].message.content
    html_code = extract_html_content(output)
    history.append({"role": "assistant", "content": output})
    yield AdditionalOutputs(history, html_code)

CodeHandler = ReplyOnPause(generate)  # type: ignore