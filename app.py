"""Chat UI for huihui-ai/Huihui-Spark-X2.5-4B-abliterated, served on a Petabyte GPU.

The model ships a custom architecture (Spark2_5ForCausalLM via auto_map) so it needs
trust_remote_code. Its chat template takes `enable_thinking`, which defaults to TRUE and
emits reasoning blocks into the reply — off by default here, toggleable in the UI.
"""
import os
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "huihui-ai/Huihui-Spark-X2.5-4B-abliterated")
CUDA = torch.cuda.is_available()

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.bfloat16 if CUDA else torch.float32,
    device_map="auto" if CUDA else None,
).eval()


def chat(message, history, think, max_new_tokens, temperature):
    msgs = [*(history or []), {"role": "user", "content": message}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=bool(think)
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=ids, streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0, temperature=float(temperature), top_p=0.9,
            pad_token_id=tok.eos_token_id,
        ),
        daemon=True,
    ).start()
    out = ""
    for chunk in streamer:
        out += chunk
        yield out


gr.ChatInterface(
    chat,
    title="Huihui-Spark-X2.5-4B-abliterated",
    description=f"Running on a Petabyte GPU · {'CUDA' if CUDA else 'CPU'} · {MODEL_ID}",
    additional_inputs=[
        gr.Checkbox(value=False, label="Show reasoning"),
        gr.Slider(64, 4096, value=1024, step=64, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
    ],
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
