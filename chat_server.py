import asyncio
import json
import os
import socket
import sys
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx


def get_local_ip():
    """Returns the primary local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# Add anemll to path to import their logic
sys.path.append(os.path.abspath("anemll"))
from tests import chat as anemll_chat

app = FastAPI(title="ANE Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model in memory
MODEL_STATE = {
    "embed_model": None,
    "ffn_models": None,
    "lmhead_model": None,
    "tokenizer": None,
    "metadata": None,
    "causal_mask": None,
    "cache_state": None,
    "loaded_dir": None,
    "stop_token_ids": set(),
}


class StaticArgs:
    def __init__(self, model_dir):
        self.d = model_dir
        self.meta = os.path.join(model_dir, "meta.yaml")
        # Ensure fallback paths exist since args are required
        self.embed = os.path.join(
            model_dir,
            os.path.basename(model_dir).replace("05b", "25") + "_embeddings.mlmodelc",
        )
        self.ffn = os.path.join(
            model_dir,
            os.path.basename(model_dir).replace("05b", "25")
            + "_FFN_PF_chunk_01of01.mlmodelc",
        )
        self.lmhead = os.path.join(
            model_dir,
            os.path.basename(model_dir).replace("05b", "25") + "_lm_head.mlmodelc",
        )
        self.tokenizer = model_dir
        self.eval = True
        self.cpu = False
        self.pf = None
        self.context_length = None
        self.split_rotate = False
        self.mem_report = False


# Fallback path search if exact name changes
def find_model_components(model_dir):
    p = Path(model_dir)
    meta_path = p / "meta.yaml"
    if not meta_path.exists():
        return None
    import yaml

    with open(meta_path, "r") as f:
        meta_yaml = yaml.safe_load(f)
    if meta_yaml:
        return meta_yaml
    return None


def load_model_into_memory(model_dir="/tmp/qwen05b"):
    import traceback

    if MODEL_STATE["loaded_dir"] == model_dir:
        return True  # Already loaded

    print(f"Loading weights from {model_dir} into Neural Engine...")
    args = StaticArgs(model_dir)
    yaml_meta = find_model_components(model_dir)

    if yaml_meta:
        model_info = yaml_meta.get("model_info", {})
        params = model_info.get("parameters", {})
        args.embed = str(
            Path(model_dir) / params.get("embeddings", "qwen25_embeddings.mlmodelc")
        )
        args.lmhead = str(
            Path(model_dir) / params.get("lm_head", "qwen25_lm_head.mlmodelc")
        )
        args.ffn = str(
            Path(model_dir) / params.get("ffn", "qwen25_FFN_PF_chunk_01of01.mlmodelc")
        )

    try:
        # Load core models using anemll functions
        embed_model, ffn_models, lmhead_model, metadata = anemll_chat.load_models(
            args, {}
        )
        tokenizer = anemll_chat.initialize_tokenizer(model_dir, eval_mode=True)

        causal_mask = anemll_chat.initialize_causal_mask(
            metadata["context_length"], eval_mode=True
        )
        cache_state = anemll_chat.create_unified_state(
            ffn_models, metadata["context_length"], eval_mode=True, metadata=metadata
        )

        # Save to global
        MODEL_STATE["embed_model"] = embed_model
        MODEL_STATE["ffn_models"] = ffn_models
        MODEL_STATE["lmhead_model"] = lmhead_model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["metadata"] = metadata
        MODEL_STATE["causal_mask"] = causal_mask
        MODEL_STATE["cache_state"] = cache_state
        MODEL_STATE["loaded_dir"] = model_dir

        # Stop IDs
        MODEL_STATE["stop_token_ids"] = anemll_chat.build_stop_token_ids(tokenizer)

        print("Model loaded natively to ANE successfully.", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"Error loading model {model_dir}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False


@app.get("/api/models")
def get_models():
    # Detect available converted models in /tmp and TMPDIR
    models = []
    
    search_dirs = [Path("ane_studio/models"), Path("/tmp")]
    if os.environ.get("TMPDIR"):
        search_dirs.append(Path(os.environ.get("TMPDIR")))
        
    for tmp_path in search_dirs:
        if not tmp_path.exists():
            continue
        for subdir in tmp_path.iterdir():
            try:
                if subdir.is_dir() and (subdir / "meta.yaml").exists():
                    # Avoid duplicates if TMPDIR and /tmp overlap or have same models
                    if not any(m["name"] == subdir.name for m in models):
                        models.append({"id": str(subdir), "name": subdir.name})
            except (PermissionError, OSError):
                continue
                
    return {"models": models}


@app.get("/api/version")
def ollama_version():
    """Ollama-compatible version endpoint"""
    return {"version": "0.3.0"}


@app.get("/api/tags")
def ollama_tags():
    # Provide an Ollama compatible endpoint for Zed editor
    models = get_models()["models"]
    return {
        "models": [
            {
                "name": m["name"],
                "model": m["name"],
                "details": {
                    "family": "llama",
                    "parameter_size": "Unknown",
                    "quantization_level": "ANE_Native",
                },
            }
            for m in models
        ]
    }


@app.post("/api/show")
async def ollama_show(request: Request):
    """Ollama-compatible endpoint to show model details"""
    data = await request.json()
    model_name = data.get("name", "")

    # Find the model
    models = get_models()["models"]
    for m in models:
        if m["name"] == model_name:
            return {
                "license": "unknown",
                "modelfile": f"FROM {model_name}",
                "parameters": "ANE_Native",
                "template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant:",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "Unknown",
                    "quantization_level": "ANE_Native",
                },
                "model_info": {
                    "general.architecture": "transformer",
                    "general.file_type": 0,
                },
            }

    return {"error": f"model '{model_name}' not found"}


@app.get("/api/ps")
def ollama_ps():
    """Ollama-compatible endpoint to show running models"""
    if MODEL_STATE["loaded_dir"]:
        model_name = Path(MODEL_STATE["loaded_dir"]).name
        return {
            "models": [
                {
                    "name": model_name,
                    "model": model_name,
                    "size": 0,
                    "digest": "ane-native",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "Unknown",
                        "quantization_level": "ANE_Native",
                    },
                    "expires_at": "0001-01-01T00:00:00Z",
                    "size_vram": 0,
                }
            ]
        }
    return {"models": []}


@app.post("/api/load")
async def load_model_api(request: Request):
    data = await request.json()
    model_dir = data.get("model_dir", "/tmp/qwen05b")
    success = load_model_into_memory(model_dir)
    return {"status": "success" if success else "error"}


async def generate_chat_sse(prompt_text: str):
    """Async generator to stream tokens via Server-Sent Events"""
    tokenizer = MODEL_STATE["tokenizer"]
    metadata = MODEL_STATE["metadata"]
    embed_model = MODEL_STATE["embed_model"]
    ffn_models = MODEL_STATE["ffn_models"]
    lmhead_model = MODEL_STATE["lmhead_model"]
    causal_mask = MODEL_STATE["causal_mask"]
    state = MODEL_STATE["cache_state"]
    stop_ids = MODEL_STATE["stop_token_ids"]

    # Format prompt
    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(torch.int32)
    except:
        formatted_prompt = f"[INST] {prompt_text} [/INST]"
        input_ids = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(torch.int32)

    context_pos = input_ids.size(1)
    context_length = metadata.get("context_length")
    batch_size = metadata.get("batch_size", 64)
    sliding_window = metadata.get("sliding_window", None)
    update_mask_prefill = metadata.get("update_mask_prefill", False)
    single_token_mode = not update_mask_prefill

    # Run Prefill
    anemll_chat.run_prefill(
        embed_model,
        ffn_models,
        input_ids,
        context_pos,
        context_length,
        batch_size,
        state,
        causal_mask,
        sliding_window,
        single_token_mode=single_token_mode,
        use_update_mask=update_mask_prefill,
    )

    pos = context_pos
    tokens_generated = 0
    buffer_bytes = []

    while pos < context_length - 1:
        # Prevent event loop blocking with ANE calls using asyncio.sleep
        await asyncio.sleep(0)

        next_token = anemll_chat.generate_next_token(
            embed_model,
            ffn_models,
            lmhead_model,
            input_ids,
            pos,
            context_length,
            metadata,
            state,
            causal_mask,
        )

        if pos < input_ids.size(1):
            input_ids[0, pos] = next_token
        else:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.int32)], dim=1
            )

        if next_token in stop_ids:
            break

        # Decode directly to avoid building a complex decode mapping buffer
        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
        if token_str:
            data = json.dumps({"token": token_str})
            yield f"data: {data}\n\n"

        pos += 1
        tokens_generated += 1
        if tokens_generated > 1000:
            break

    yield f"data: [DONE]\n\n"


async def inject_mcp_tools(messages: list):
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11437/mcp/list", timeout=2.0)
            if resp.status_code == 200:
                mcp_data = resp.json()
                tool_lines = []
                for server, tools in mcp_data.items():
                    if isinstance(tools, list):
                        for t in tools:
                            tool_lines.append(f"""
<tool>
  <name>{server}:{t.get('name')}</name>
  <description>{t.get('description')}</description>
  <parameters>{json.dumps(t.get('inputSchema', {}), indent=2)}</parameters>
</tool>""")

                if tool_lines:
                    system_instructions = "You have access to the following tools. If you need to use a tool, respond ONLY with a JSON object in this format: { \"tool_call\": { \"name\": \"SERVER:TOOL_NAME\", \"arguments\": {} } }. WAIT for the user to provide the tool result before continuing.\n\n<tools>" + "".join(tool_lines) + "\n</tools>"
                    
                    # Check if there's already a system prompt
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] += "\n\n" + system_instructions
                    else:
                        messages.insert(0, {"role": "system", "content": system_instructions})
    except Exception as e:
        print(f"Warning: Failed to fetch MCP tools: {e}")

async def generate_ollama_stream(model_name: str, messages: list):
    """Async generator to stream tokens via newline-delimited JSON (Ollama format for Zed)"""
    await inject_mcp_tools(messages)
    
    tokenizer = MODEL_STATE["tokenizer"]
    metadata = MODEL_STATE["metadata"]
    embed_model = MODEL_STATE["embed_model"]
    ffn_models = MODEL_STATE["ffn_models"]
    lmhead_model = MODEL_STATE["lmhead_model"]
    causal_mask = MODEL_STATE["causal_mask"]
    state = MODEL_STATE["cache_state"]
    stop_ids = MODEL_STATE["stop_token_ids"]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(torch.int32)
    except:
        formatted_prompt = ""
        for m in messages:
            formatted_prompt += f"{m['role']}: {m['content']}\n"
        formatted_prompt += "assistant: "
        input_ids = tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(torch.int32)

    context_pos = input_ids.size(1)
    context_length = metadata.get("context_length")
    batch_size = metadata.get("batch_size", 64)
    sliding_window = metadata.get("sliding_window", None)
    update_mask_prefill = metadata.get("update_mask_prefill", False)
    single_token_mode = not update_mask_prefill

    anemll_chat.run_prefill(
        embed_model,
        ffn_models,
        input_ids,
        context_pos,
        context_length,
        batch_size,
        state,
        causal_mask,
        sliding_window,
        single_token_mode=single_token_mode,
        use_update_mask=update_mask_prefill,
    )

    pos = context_pos
    tokens_generated = 0

    while pos < context_length - 1:
        await asyncio.sleep(0)

        next_token = anemll_chat.generate_next_token(
            embed_model,
            ffn_models,
            lmhead_model,
            input_ids,
            pos,
            context_length,
            metadata,
            state,
            causal_mask,
        )

        if pos < input_ids.size(1):
            input_ids[0, pos] = next_token
        else:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.int32)], dim=1
            )

        if next_token in stop_ids:
            break

        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
        if token_str:
            resp = {
                "model": model_name,
                "message": {"role": "assistant", "content": token_str},
                "done": False,
            }
            yield json.dumps(resp) + "\n"

        pos += 1
        tokens_generated += 1
        if tokens_generated > 1000:
            break

    final_resp = {
        "model": model_name,
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }
    yield json.dumps(final_resp) + "\n"


async def generate_ollama_sync(model_name: str, messages: list):
    """Async function to generate full response synchronously for non-streaming Ollama requests"""
    if not MODEL_STATE["loaded_dir"]:
        load_model_into_memory()

    await inject_mcp_tools(messages)

    tokenizer = MODEL_STATE["tokenizer"]
    metadata = MODEL_STATE["metadata"]
    embed_model = MODEL_STATE["embed_model"]
    ffn_models = MODEL_STATE["ffn_models"]
    lmhead_model = MODEL_STATE["lmhead_model"]
    causal_mask = MODEL_STATE["causal_mask"]
    state = MODEL_STATE["cache_state"]
    stop_ids = MODEL_STATE["stop_token_ids"]

    if tokenizer is None:
        return {
            "model": model_name,
            "message": {"role": "assistant", "content": f"Error: Failed to load model {model_name}"},
            "done": True,
        }

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(torch.int32)
    except:
        formatted_prompt = ""
        for m in messages:
            formatted_prompt += f"{m['role']}: {m['content']}\n"
        formatted_prompt += "assistant: "
        
        try:
            input_ids = tokenizer(
                formatted_prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(torch.int32)
        except TypeError:
            # Fallback if tokenizer is custom and lacks __call__ but has encode
            input_ids = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.int32)

    context_pos = input_ids.size(1)
    context_length = metadata.get("context_length")
    batch_size = metadata.get("batch_size", 64)
    sliding_window = metadata.get("sliding_window", None)
    update_mask_prefill = metadata.get("update_mask_prefill", False)
    single_token_mode = not update_mask_prefill

    anemll_chat.run_prefill(
        embed_model,
        ffn_models,
        input_ids,
        context_pos,
        context_length,
        batch_size,
        state,
        causal_mask,
        sliding_window,
        single_token_mode=single_token_mode,
        use_update_mask=update_mask_prefill,
    )

    pos = context_pos
    tokens_generated = 0
    full_content = ""

    while pos < context_length - 1:
        await asyncio.sleep(0)

        next_token = anemll_chat.generate_next_token(
            embed_model,
            ffn_models,
            lmhead_model,
            input_ids,
            pos,
            context_length,
            metadata,
            state,
            causal_mask,
        )

        if pos < input_ids.size(1):
            input_ids[0, pos] = next_token
        else:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], dtype=torch.int32)], dim=1
            )

        if next_token in stop_ids:
            break

        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
        if token_str:
            full_content += token_str

        pos += 1
        tokens_generated += 1
        if tokens_generated > 1000:
            break

    return {
        "model": model_name,
        "message": {"role": "assistant", "content": full_content},
        "done": True,
    }


# ----------------- MCP & Utility Bridge Proxies ------------------
@app.get("/api/mcp/list")
async def proxy_mcp_list():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11437/mcp/list", timeout=5.0)
            return resp.json()
    except Exception as e:
        return {"error": f"Utility bridge unreachable: {e}"}

@app.post("/api/mcp/call")
async def proxy_mcp_call(request: Request):
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://localhost:11437/mcp/call", json=data, timeout=30.0)
            return resp.json()
    except Exception as e:
        return {"error": f"Utility bridge unreachable: {e}"}

@app.post("/api/upload")
async def proxy_upload(file: UploadFile = File(...)):
    try:
        async with httpx.AsyncClient() as client:
            file_data = await file.read()
            files = {'file': (file.filename, file_data, file.content_type)}
            resp = await client.post("http://localhost:11437/api/upload", files=files, timeout=60.0)
            return resp.json()
    except Exception as e:
        return {"error": f"Utility bridge unreachable: {e}"}
# -----------------------------------------------------------------

@app.post("/api/chat")
@app.post("/api/chat/stream")
@app.post("/api/generate")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        print(f"DEBUG INBOUND REQUEST: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"DEBUG REQUEST PARSE ERROR: {e}")
        data = {}

    # Check if a model change is requested (e.g. from Zed ollama spec)
    model = data.get("model")
    model_loaded = False
    
    if model:
        # Check both /tmp and TMPDIR
        search_dirs = [Path("ane_studio/models"), Path("/tmp")]
        if os.environ.get("TMPDIR"):
            search_dirs.append(Path(os.environ.get("TMPDIR")))
            
        for tmp_path in search_dirs:
            if not tmp_path.exists():
                continue
            for subdir in tmp_path.iterdir():
                if subdir.is_dir() and subdir.name == model:
                    print(f"DEBUG: Found matching model directory for {model} in {tmp_path}. Attempting load...", flush=True)
                    model_loaded = load_model_into_memory(str(subdir))
                    break
            if model_loaded:
                break

    if not MODEL_STATE["loaded_dir"] and not model_loaded:
        print(f"DEBUG: Model still not loaded. Defaulting load_model_into_memory().", flush=True)
        # Check if the requested model directory exists explicitly even if not matched in iterdir
        if model and Path(f"ane_studio/models/{model}").exists():
            load_model_into_memory(f"ane_studio/models/{model}")
        elif model and Path(f"/tmp/{model}").exists():
            load_model_into_memory(f"/tmp/{model}")
        elif model and os.environ.get("TMPDIR") and Path(f"{os.environ.get('TMPDIR')}/{model}").exists():
            load_model_into_memory(f"{os.environ.get('TMPDIR')}/{model}")
        else:
            load_model_into_memory()

    if not MODEL_STATE["loaded_dir"] and not model_loaded:
        # If model failed to load in both passes
        print(f"DEBUG: Aborting chat generation, no model in memory.", flush=True)

    # Ollama API Check
    if "messages" in data and "prompt" not in data:
        messages = data.get("messages", [])
        stream = data.get("stream", True)
        if stream:
            return StreamingResponse(
                generate_ollama_stream(model or "qwen05b", messages),
                media_type="application/x-ndjson",
            )
        else:
            return await generate_ollama_sync(model or "qwen05b", messages)

    # Native UI check
    prompt = data.get("prompt", "")
    return StreamingResponse(generate_chat_sse(prompt), media_type="text/event-stream")


# Serve the Single Page App or API info
@app.get("/")
def get_root(request: Request):
    # Check if client wants JSON (API client like Zed, Ollama, etc.)
    accept_header = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()

    # Return JSON for API clients, HTML for browsers
    if (
        "application/json" in accept_header
        or "ollama" in user_agent
        or "python" in user_agent
        or "curl" in user_agent
        or "wget" in user_agent
    ):
        return {
            "name": "ANE Chat Server",
            "version": "0.3.0",
            "ollama_compatible": True,
            "endpoints": {
                "models": "/api/tags",
                "chat": "/api/chat",
                "generate": "/api/generate",
                "version": "/api/version",
            },
        }

    html_path = os.path.join(os.path.dirname(__file__), "ane_studio", "ui", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    # Autoload model synchronously
    load_model_into_memory()
    local_ip = get_local_ip()
    port = 11436
    print(f"\n🚀 ANE Chat Server starting...")
    print(f"🔗 Local:  http://127.0.0.1:{port}")
    print(f"🌐 Network: http://{local_ip}:{port}")
    print(f"💡 For network access (Zed, mobile, etc.), use the Network URL above.\n")
    uvicorn.run("chat_server:app", host="0.0.0.0", port=port, reload=False)
