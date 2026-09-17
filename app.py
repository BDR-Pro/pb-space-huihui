"""Chat UI for huihui-ai/Huihui-Spark-X2.5-4B-abliterated, served on a Petabyte GPU.

The model ships a custom architecture (Spark2_5ForCausalLM via auto_map) so it needs
trust_remote_code. Its chat template takes `enable_thinking`, which defaults to TRUE and
emits reasoning blocks into the reply — off by default here, toggleable in the UI.

The model itself has no internet: it is a plain generate loop. `tools.py` gives it search, URL
fetch and shell, and the loop below runs whatever it asks for and feeds the result back.

ACCESS: a Petabyte space is served on a PUBLIC url whose only protection is the opaque VM id.
Set SPACE_PASSWORD (env, or a .space_password file next to this one) and the whole UI goes behind
a login — and only then is the shell tool enabled. With no password the space stays open and
shell stays off, because an open shell on a public url is a gift to whoever finds it.
"""
import os
from threading import Thread

# The Petabyte gateway terminates TLS and forwards X-Forwarded-Proto, but uvicorn only BELIEVES
# that header from an address in forwarded_allow_ips, which defaults to 127.0.0.1 — and the
# request arrives from the docker bridge, not loopback. Without this Gradio keeps thinking it is
# on http and emits http:// URLs for its own JS and CSS, which the browser blocks as mixed
# content on an https page: the space serves 200s, accepts the login, and renders BLANK.
# Set before gradio imports uvicorn, since uvicorn reads it when the Config is built.
os.environ.setdefault("FORWARDED_ALLOW_IPS", "*")

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import tools

MODEL_ID = os.environ.get("MODEL_ID", "huihui-ai/Huihui-Spark-X2.5-4B-abliterated")
CUDA = torch.cuda.is_available()
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "4"))


def _password():
    """Env first; a file lets an ALREADY-RUNNING space gain a password without re-booking it
    (container env is fixed at creation, the repo checkout is not)."""
    pw = os.environ.get("SPACE_PASSWORD", "").strip()
    if pw:
        return pw
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".space_password")) as f:
            return f.read().strip()
    except OSError:
        return ""


PASSWORD = _password()
USERNAME = os.environ.get("SPACE_USERNAME", "petabyte").strip() or "petabyte"
SHELL_OK = bool(PASSWORD)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.bfloat16 if CUDA else torch.float32,
    device_map="auto" if CUDA else None,
).eval()


def _stream(msgs, think, max_new_tokens, temperature):
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=bool(think)
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        input_ids=ids, streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0, temperature=float(temperature), top_p=0.9,
        pad_token_id=tok.eos_token_id,
    )
    # Stop the moment a tool call closes, instead of letting the model narrate what it imagines
    # the result was. Older transformers lack stop_strings; the parser copes either way.
    try:
        kwargs.update(stop_strings=["</tool>"], tokenizer=tok)
    except Exception:  # noqa: BLE001
        pass
    Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    for chunk in streamer:
        yield chunk


def chat(message, history, think, max_new_tokens, temperature, use_tools):
    msgs = list(history or [])
    if use_tools:
        msgs = [{"role": "system", "content": tools.system_prompt(SHELL_OK)}] + msgs
    msgs.append({"role": "user", "content": message})

    shown = ""
    for _ in range(MAX_TOOL_ROUNDS):
        out = ""
        for chunk in _stream(msgs, think, max_new_tokens, temperature):
            out += chunk
            yield shown + out

        call = tools.parse_tool_call(out) if use_tools else None
        if not call:
            return                                  # ordinary answer, already streamed
        name, arg = call

        prose = out.split("<tool>")[0].strip()
        shown += (prose + "\n\n") if prose else ""
        shown += f"> 🔧 **{name}** · `{arg.splitlines()[0][:120]}`\n"
        yield shown + "\n_running…_"

        result = tools.run(name, arg, shell_ok=SHELL_OK)
        shown += "\n```\n" + result + "\n```\n\n"
        yield shown

        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})

    yield shown + f"\n_(stopped after {MAX_TOOL_ROUNDS} tool calls)_"


_access = ("🔒 password-protected · shell **on**" if SHELL_OK
           else "🌐 public link · shell **off** (set SPACE_PASSWORD to enable it)")

demo = gr.ChatInterface(
    chat,
    title="Huihui-Spark-X2.5-4B-abliterated",
    description=(f"Running on a Petabyte GPU · {'CUDA' if CUDA else 'CPU'} · {MODEL_ID}<br>"
                 f"Tools: search · fetch · shell — {_access}"),
    additional_inputs=[
        gr.Checkbox(value=False, label="Show reasoning"),
        gr.Slider(64, 4096, value=1024, step=64, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Checkbox(value=True, label="Use tools (search / fetch / shell)"),
    ],
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    auth=(USERNAME, PASSWORD) if PASSWORD else None,
)
