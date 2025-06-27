import base64
import json
import os
from pathlib import Path
import secrets
import websocket
import gradio as gr
import numpy as np
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnStopWords,
    Stream,
    get_stt_model,
    get_twilio_turn_credentials,
)
from gradio.utils import get_space
from pydantic import BaseModel
from websockets.asyncio.client import connect

load_dotenv()

curr_dir = Path(__file__).parent
model = get_stt_model()


# 获取 API 密钥
API_KEY = os.getenv("Qwen_API_KEY")
print(API_KEY)

def msg_id() -> str:
    return f"event_{secrets.token_hex(10)}"

# 定义 URL 和语音选项
API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen-omni-turbo-realtime-latest"
VOICES = ["Chelsie", "Serena", "Ethan", "Cherry"]
# 设置请求头（注意格式转换）
headers = {"Authorization": "Bearer " + API_KEY}
global_msg_id = msg_id()
# 定义全局变量
client = None
connection = None
try:
    # 将 headers 转换为字符串列表格式
    header_list = [f"{key}: {value}" for key, value in headers.items()]

    # 创建同步 WebSocket 连接
    connection = websocket.create_connection(API_URL, header=header_list)
    print("Connected to Qwen")

    # 构造并发送 JSON 数据
    payload = json.dumps({
        "event_id": global_msg_id,
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "turn_detection": None,
            "voice": "Chelsie",
            "input_audio_format": "pcm16",
        }
    })
    connection.send(payload)
    print("Connected successfully")

except Exception as e:
    print("Connection failed:", e)
    os._exit(1)
def response(
    audio: tuple[int, np.ndarray],
    gradio_chatbot: list[dict] | None = None,
    conversation_state: list[dict] | None = None,
):  
    msg_id  = f"event_{secrets.token_hex(10)}"
    gradio_chatbot = gradio_chatbot or []
    conversation_state = conversation_state or []
    _, array = audio
    array = array.squeeze()
    audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
    connection.send(
        json.dumps(
            {
                "event_id": msg_id,
                "type": "input_audio_buffer.append",
                "audio": audio_message,
            }
        )
    )
    connection.send(
        json.dumps(
            {
                "event_id": msg_id,
                "type": "input_audio_buffer.commit",
            }
        )
    )
    connection.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "instructions": "Hello",
                    "modalities": ["text", "audio"]
                },
                "event_id": msg_id
            }
        )
    )
    user_text = ""
    response_text = ""
    while True:
        data = connection.recv()
        event = json.loads(data)
        # print(event)
        if "type" not in event:
            continue
        if event["type"] == "response.audio_transcript.delta":
            response_text += event["delta"]
        if event["type"] == "response.done":
            connection.send(
                json.dumps(
                    {
                        "event_id": msg_id,
                        "type": "input_audio_buffer.clear",
                    }
                )
            )
            break
    print(f"user_text: {user_text}")
    print(f"response_text: {response_text}")
    sample_rate, array = audio
    gradio_chatbot.append(
        {"role": "user", "content": gr.Audio((sample_rate, array.squeeze()))}
    )
    yield AdditionalOutputs(gradio_chatbot, conversation_state)

    conversation_state.append({"role": "user", "content": user_text})

    response = {"role": "assistant", "content": response_text}

    conversation_state.append(response)
    gradio_chatbot.append(response)

    yield AdditionalOutputs(gradio_chatbot, conversation_state)


title = gr.Markdown(
        "## 说出 **computer** 唤醒词后激活 Qwen-omni",
        elem_id="activation-instruction"
    )
chatbot = gr.Chatbot(type="messages", value=[])
state = gr.State(value=[])
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

app = FastAPI()
stream.mount(app)


class Message(BaseModel):
    role: str
    content: str


class InputData(BaseModel):
    webrtc_id: str
    chatbot: list[Message]
    state: list[Message]


@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (curr_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


@app.post("/input_hook")
async def _(data: InputData):
    body = data.model_dump()
    stream.set_input(data.webrtc_id, body["chatbot"], body["state"])


def audio_to_base64(file_path):
    audio_format = "wav"
    with open(file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    return f"data:audio/{audio_format};base64,{encoded_audio}"


@app.get("/outputs")
async def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            chatbot = output.args[0]
            state = output.args[1]
            data = {
                "message": state[-1],
                "audio": audio_to_base64(chatbot[-1]["content"].value["path"])
                if chatbot[-1]["role"] == "user"
                else None,
            }
            yield f"event: output\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)

