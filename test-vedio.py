import asyncio
import base64
import json
import os
import secrets
import signal
import time
from io import BytesIO
from fastapi.responses import HTMLResponse, StreamingResponse

from fastapi import FastAPI
from pathlib import Path
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastrtc import (
    AsyncAudioVideoStreamHandler,
    AdditionalOutputs,
    VideoEmitType,
    Stream,
    WebRTC,
    get_cloudflare_turn_credentials_async,
    wait_for_item,
)
from gradio.utils import get_space
from PIL import Image
from websockets.asyncio.client import connect

load_dotenv()

cur_dir = Path(__file__).parent
API_KEY = os.getenv("Qwen_API_KEY")
print(API_KEY)
API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen-omni-turbo-realtime-latest"
VOICES = ["Chelsie", "Serena", "Ethan", "Cherry"]
headers = {"Authorization": "Bearer " + API_KEY}

def encode_audio(data: np.ndarray) -> dict:
    """Encode Audio data to send to the server"""
    return {
        "mime_type": "audio/pcm",
        "data": base64.b64encode(data.tobytes()).decode("UTF-8"),
    }


def encode_image(data: np.ndarray) -> dict:
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(data)
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return {"mime_type": "image/jpeg", "data": base64_str}


class QwenOmniHandler(AsyncAudioVideoStreamHandler):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            "mono",
            output_sample_rate=24000,
            input_sample_rate=16000,
        )
        self.video_queue = asyncio.Queue()
        self.last_frame_time = 0
        self.quit = asyncio.Event()
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.last_valid_frame = None  # 新增：保存最近一帧

    def copy(self):
        return QwenOmniHandler()
    
    @staticmethod
    def msg_id() -> str:
        return f"event_{secrets.token_hex(10)}"

    async def start_up(self):
        try:
            print("starting up")
            # await self.wait_for_args()
            print(API_URL)
            print(headers)
            async with connect(API_URL, additional_headers=headers) as conn:
                print("Connected to Qwen")
                self.client = conn
                print("id" + self.msg_id())
                await conn.send(
                    json.dumps(
                        {
                            "event_id": self.msg_id(),
                            "type": "session.update",
                            "session": {
                                "modalities": ["text", "audio"],
                                "voice": "Chelsie",
                                "input_audio_format": "pcm16",
                            },
                        }
                    )
                )
                print("Connected successfully")
                self.connection = conn
                async for data in self.connection:
                    event = json.loads(data)
                    if "type" not in event:
                        continue
                    if event["type"] == "input_audio_buffer.speech_started":
                        print("1.speech started")
                        self.clear_queue()
                    if event["type"] == "conversation.item.input_audio_transcription.completed":
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user", "content": event["transcript"]})
                        )
                        print("2." + event["transcript"])
                    if event["type"] == "response.audio_transcript.done":
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event["transcript"]})
                        )
                        print("3." + event["transcript"])
                    if event["type"] == "response.audio.delta":
                        await self.output_queue.put(
                            (self.output_sample_rate, np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16).reshape(1, -1))
                        )
                        print("4.")
                    
        except Exception as e:
            print(f"Connection error: {e}")
            await self.shutdown()
        # while not self.quit.is_set():
        #         turn = self.session.receive()
        #         try:
        #             async for response in turn:
        #                 if data := response.data:
        #                     audio = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        #                 self.audio_queue.put_nowait(audio)
        #         except websockets.exceptions.ConnectionClosedOK:
        #             print("connection closed")
        #             break

    async def video_receive(self, frame: np.ndarray):
        self.video_queue.put_nowait(frame)
        if self.connection:
            # send image every 1 second
            if time.time() - self.last_frame_time > 1:
                print("sending image")
                self.last_frame_time = time.time()
                image_send =  encode_image(frame)
                await self.connection.send(
                        json.dumps(
                            {
                                "event_id": self.msg_id(),
                                "type": "input_image_buffer.append",
                                "image": image_send["data"],
                            }
                        )
                    )
                print("sending image success")
                # if self.latest_args[0] is not None:
                #     image_send =  encode_image(self.latest_args[0])
                #     await self.connection.send(
                #         json.dumps(
                #             {
                #                 "event_id": self.msg_id(),
                #                 "type": "input_image_buffer.append",
                #                 "image": image_send,
                #             }
                #         )
                #     )

    async def video_emit(self) -> VideoEmitType:
        return await self.video_queue.get()

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.send(
            json.dumps(
                {
                    "event_id": self.msg_id(),
                    "type": "input_audio_buffer.append",
                    "audio": audio_message,
                }
            )
        )
        

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None

        # 清空队列
        while not self.output_queue.empty():
            self.output_queue.get_nowait()

def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot



voice = gr.Dropdown(choices=VOICES, value=VOICES[0], type="value", label="Voice")
latest_message = gr.Textbox(type="text", visible=False)
# image = gr.Image(label="Image", type="numpy", sources=["upload", "clipboard"])
# 可选：暂时禁用 TURN 配置进行测试
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # 取消注释可禁用 TURN 测试

css = """
#video-source {max-width: 600px !important; max-height: 600 !important;}
"""

with gr.Blocks(css=css) as demo:
    # with gr.Row() as api_key_row:
    #     api_key = gr.Textbox(label="API Key", type="password", placeholder="Enter your API Key", value=os.getenv("GOOGLE_API_KEY"))
    with gr.Row() as row:
        with gr.Column():
            webrtc = WebRTC(
                label="Video Chat",
                modality="audio-video",
                mode="send-receive",
                elem_id="video-source",
                rtc_configuration=rtc_config,
                icon="https://avatars.githubusercontent.com/u/109945100?s=200&v=4",
                pulse_color="rgb(35, 157, 225)",
                icon_button_color="rgb(35, 157, 225)",
            )
        with gr.Column():
            image_input = gr.Image(label="Image", type="numpy", sources=["upload", "clipboard"])

        webrtc.stream(
            QwenOmniHandler(),
            inputs=[webrtc,image_input],
            outputs=[webrtc],
            time_limit=90,
            concurrency_limit=2,
        )


if __name__ == "__main__":
    demo.launch()