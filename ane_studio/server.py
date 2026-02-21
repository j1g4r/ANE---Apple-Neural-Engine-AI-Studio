"""
ANE Studio — Backend API Server
Full-featured FastAPI server for Apple Neural Engine model management and inference.
"""

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import psutil
import torch
import uvicorn
import yaml
import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


def get_local_ip():
    """Returns the primary local IP address of the machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just triggers OS routing logic
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── anemll import ───────────────────────────────────────────────────────────
ANEMLL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anemll")
sys.path.insert(0, ANEMLL_PATH)
from tests import chat as anemll_chat

# ─── App Configuration ───────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(APP_DIR, "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="ANE Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {
        "id": "qwen-0.5b",
        "name": "Qwen 2.5 0.5B Instruct",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "size_gb": 1.2,
        "gated": False,
        "description": "Ultra-fast lightweight model. Great for testing.",
        "family": "qwen",
    },
    {
        "id": "qwen-1.5b",
        "name": "Qwen 2.5 1.5B Instruct",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "size_gb": 3.5,
        "gated": False,
        "description": "Good balance of speed and capability.",
        "family": "qwen",
    },
    {
        "id": "deepseek-r1-qwen-1.5b",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "params": "1.5B",
        "size_gb": 3.5,
        "gated": False,
        "description": "DeepSeek's distilled model with strong reasoning.",
        "family": "qwen",
    },
    {
        "id": "qwen-3b",
        "name": "Qwen 2.5 3B Instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params": "3B",
        "size_gb": 6.5,
        "gated": False,
        "description": "Recommended. Strong reasoning with fast speed.",
        "family": "qwen",
    },
    {
        "id": "qwen-coder-3b",
        "name": "Qwen 2.5 Coder 3B Instruct",
        "hf_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "params": "3B",
        "size_gb": 6.5,
        "gated": False,
        "description": "Specialized for coding tasks.",
        "family": "qwen",
    },
    {
        "id": "qwen-7b",
        "name": "Qwen 2.5 7B Instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "params": "7B",
        "size_gb": 14,
        "gated": False,
        "description": "Highest quality. Requires significant RAM.",
        "family": "qwen",
    },
    {
        "id": "deepseek-r1-qwen-7b",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "params": "7B",
        "size_gb": 14,
        "gated": False,
        "description": "Advanced reasoning version of Qwen 7B.",
        "family": "qwen",
    },
    {
        "id": "gemma-1b",
        "name": "Gemma 3 1B IT",
        "hf_id": "google/gemma-3-1b-it",
        "params": "1B",
        "size_gb": 2.5,
        "gated": True,
        "description": "Ultra-compact Google model.",
        "family": "gemma",
    },
    {
        "id": "gemma-4b",
        "name": "Gemma 3 4B IT",
        "hf_id": "google/gemma-3-4b-it",
        "params": "4B",
        "size_gb": 8,
        "gated": True,
        "description": "Google's efficient model. Requires HuggingFace login.",
        "family": "gemma",
    },
    {
        "id": "llama-1b",
        "name": "LLaMA 3.2 1B Instruct",
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "params": "1B",
        "size_gb": 2.5,
        "gated": True,
        "description": "Meta's compact model. Requires HuggingFace login.",
        "family": "llama",
    },
    {
        "id": "smollm2-1.7b",
        "name": "SmolLM2 1.7B Instruct",
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "params": "1.7B",
        "size_gb": 3.8,
        "gated": False,
        "description": "High performance compact model from HuggingFace.",
        "family": "llama",
    },
    {
        "id": "llama-3b",
        "name": "LLaMA 3.2 3B Instruct",
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "params": "3B",
        "size_gb": 6.5,
        "gated": True,
        "description": "Meta's mid-range model. Requires HuggingFace login.",
        "family": "llama",
    },
    {
        "id": "smollm2-360m",
        "name": "SmolLM2 360M Instruct",
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "size_gb": 0.8,
        "gated": False,
        "description": "Extremely fast, fits on any device.",
        "family": "llama",
    },
]

# ─── Global State ─────────────────────────────────────────────────────────────
ENGINE_STATE = {
    "embed_model": None,
    "ffn_models": None,
    "lmhead_model": None,
    "tokenizer": None,
    "metadata": None,
    "causal_mask": None,
    "cache_state": None,
    "loaded_model_id": None,
    "loaded_dir": None,
    "stop_token_ids": set(),
}

# Track download progress per model
# model_id -> { "status": "downloading"|"compiling"|"done"|"error", "progress": 0-100, "message": "", "full_log": "" }
DOWNLOAD_PROGRESS = {}
PROGRESS_LOCK = threading.Lock()


# API Server config
API_CONFIG = {
    "port": 11436,
    "host": "0.0.0.0",
    "running": True,
    "hf_token": "",  # Default provided by user
}

# ─── Global State ─────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def get_model_dir(model_id: str) -> str:
    return os.path.join(MODELS_DIR, model_id)


def is_model_installed(model_id: str) -> bool:
    model_dir = get_model_dir(model_id)
    return os.path.isdir(model_dir) and os.path.exists(
        os.path.join(model_dir, "meta.yaml")
    )


def get_installed_models():
    installed_ids = set()
    installed = []
    
    # 1. Check registry models
    for entry in MODEL_REGISTRY:
        if is_model_installed(entry["id"]):
            model_dir = get_model_dir(entry["id"])
            try:
                size_on_disk = sum(f.stat().st_size for f in Path(model_dir).rglob("*") if f.is_file())
                installed.append({
                    **entry,
                    "installed": True,
                    "path": model_dir,
                    "size_on_disk_gb": round(size_on_disk / (1024**3), 2),
                })
                installed_ids.add(entry["id"])
            except Exception: continue

    # 2. Scan for others (like chat_server.py did)
    if os.path.exists(MODELS_DIR):
        for subdir in Path(MODELS_DIR).iterdir():
            if subdir.is_dir() and (subdir / "meta.yaml").exists() and subdir.name not in installed_ids:
                try:
                    size_on_disk = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                    installed.append({
                        "id": subdir.name,
                        "name": subdir.name.replace("-", " ").title(),
                        "hf_id": f"local/{subdir.name}",
                        "params": "Unknown",
                        "size_gb": round(size_on_disk / (1024**3), 2),
                        "gated": False,
                        "description": "Local/Custom model found on disk.",
                        "family": "llama",
                        "installed": True,
                        "path": str(subdir),
                        "size_on_disk_gb": round(size_on_disk / (1024**3), 2),
                    })
                except Exception: continue
                
    return installed


def download_and_compile_model(model_id: str):
    """Background task: Download from HuggingFace and compile for ANE."""
    registry_entry = next((m for m in MODEL_REGISTRY if m["id"] == model_id), None)
    if not registry_entry:
        with PROGRESS_LOCK:
            DOWNLOAD_PROGRESS[model_id] = {
                "status": "error",
                "progress": 0,
                "message": "Unknown model",
                "full_log": "",
            }
        return

    with PROGRESS_LOCK:
        DOWNLOAD_PROGRESS[model_id] = {
            "status": "downloading",
            "progress": 5,
            "message": "Initializing...",
            "full_log": "Starting background process...\n",
        }

    model_dir = get_model_dir(model_id)
    # The workspace root is one level up from the ane_studio app directory
    workspace_root = os.path.dirname(APP_DIR)

    hf_id = registry_entry["hf_id"]
    convert_script = os.path.join(ANEMLL_PATH, "tests", "conv", "test_hf_model.sh")

    try:
        env = os.environ.copy()
        env["TMPDIR"] = os.path.join(APP_DIR, "tmp_cache")
        # Pass the token through environment if needed by transformers,
        # but the script also reads ~/.cache/huggingface/token
        if API_CONFIG["hf_token"]:
            env["HUGGING_FACE_HUB_TOKEN"] = API_CONFIG["hf_token"]
            # Also try to write it to the cache file so the script's python block finds it
            try:
                hf_cache = os.path.expanduser("~/.cache/huggingface")
                os.makedirs(hf_cache, exist_ok=True)
                with open(os.path.join(hf_cache, "token"), "w") as f:
                    f.write(API_CONFIG["hf_token"])
            except Exception as e:
                print(f"Warning: Could not write HF token to cache: {e}")

        os.makedirs(env["TMPDIR"], exist_ok=True)
        # Enable high-speed downloads
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Check disk space before starting
        disk = shutil.disk_usage(APP_DIR)
        if disk.free < 5 * (1024**3):  # 5GB threshold
            print(
                f"Warning: Low disk space detected ({disk.free / (1024**3):.1f} GB). Large models may fail."
            )

        # 1. cd to workspace root so relative paths in anemll scripts work
        activate_script = os.path.join(ANEMLL_PATH, "env-anemll", "bin", "activate")
        cmd = f"cd {workspace_root} && source {activate_script} && bash {convert_script} {hf_id} {model_dir}"

        # Use Popen to stream logs in real-time
        process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )

        full_log = ""
        current_status = "downloading"

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                full_log += line
                # Simple heuristical status updates based on log content
                msg = line.strip()

                # Ensure progress is initialized before accessing
                prog = DOWNLOAD_PROGRESS.get(model_id, {}).get("progress", 0)

                if "Downloading" in line:
                    current_status = "downloading"
                    prog = max(prog, 20)
                elif "Converting" in line or "Compiling" in line:
                    current_status = "compiling"
                    prog = max(prog, 60)
                elif "Tracing" in line:
                    prog = min(prog + 1, 95)

                with PROGRESS_LOCK:
                    DOWNLOAD_PROGRESS[model_id].update(
                        {
                            "status": current_status,
                            "progress": prog,
                            "message": msg[:100] + ("..." if len(msg) > 100 else ""),
                            "full_log": full_log,
                        }
                    )

        returncode = process.poll()
        if returncode != 0:
            with PROGRESS_LOCK:
                DOWNLOAD_PROGRESS[model_id].update(
                    {
                        "status": "error",
                        "progress": 0,
                        "message": f"Process exited with code {returncode}",
                        "full_log": full_log,
                    }
                )
            return

        # Verify installation
        if is_model_installed(model_id):
            with PROGRESS_LOCK:
                DOWNLOAD_PROGRESS[model_id].update(
                    {
                        "status": "done",
                        "progress": 100,
                        "message": "Model ready!",
                        "full_log": full_log
                        + "\n[SUCCESS] Model installed successfully.",
                    }
                )
        else:
            with PROGRESS_LOCK:
                DOWNLOAD_PROGRESS[model_id].update(
                    {
                        "status": "error",
                        "progress": 0,
                        "message": "Files missing after compilation",
                        "full_log": full_log
                        + "\n[ERROR] Verfication failed: meta.yaml not found.",
                    }
                )

    except Exception as e:
        with PROGRESS_LOCK:
            DOWNLOAD_PROGRESS[model_id] = {
                "status": "error",
                "progress": 0,
                "message": str(e),
                "full_log": str(e),
            }
    finally:
        # Cleanup tmp_cache to recover disk space
        try:
            tmp_dir = os.path.join(APP_DIR, "tmp_cache")
            if os.path.exists(tmp_dir):
                # We don't want to delete the directory itself as it's the TMPDIR env var for others
                for filename in os.listdir(tmp_dir):
                    file_path = os.path.join(tmp_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                print("Cleaned up tmp_cache.")
        except Exception as e:
            print(f"Cleanup error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (INTO ANE HARDWARE)
# ═══════════════════════════════════════════════════════════════════════════════


class ModelArgs:
    """Mimics the argparse namespace expected by anemll_chat.load_models()."""

    def __init__(self, model_dir, embed, ffn, lmhead):
        self.d = model_dir
        self.meta = os.path.join(model_dir, "meta.yaml")
        self.embed = embed
        self.ffn = ffn
        self.lmhead = lmhead
        self.tokenizer = model_dir
        self.eval = True
        self.cpu = False
        self.pf = None
        self.context_length = None
        self.split_rotate = False
        self.mem_report = False


ENGINE_LOCK = threading.Lock()

def load_model_into_engine(model_id: str) -> bool:
    """Synchronous model loader (should be called via to_thread)."""
    with ENGINE_LOCK:
        if ENGINE_STATE["loaded_model_id"] == model_id:
            return True

        model_dir = get_model_dir(model_id)
        if not is_model_installed(model_id):
            print(f"Model {model_id} is not installed.")
            return False

        print(f"🔄 Loading {model_id} into Neural Engine...")

        # Parse meta.yaml
        meta_path = os.path.join(model_dir, "meta.yaml")
        with open(meta_path, "r") as f:
            meta_yaml = yaml.safe_load(f)

        model_info = meta_yaml.get("model_info", {})
        params = model_info.get("parameters", {})

        embed_file = params.get("embeddings", "")
        lmhead_file = params.get("lm_head", "")
        ffn_file = params.get("ffn", "")

        args = ModelArgs(
            model_dir,
            os.path.join(model_dir, embed_file),
            os.path.join(model_dir, ffn_file),
            os.path.join(model_dir, lmhead_file),
        )

        try:
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

            ENGINE_STATE["embed_model"] = embed_model
            ENGINE_STATE["ffn_models"] = ffn_models
            ENGINE_STATE["lmhead_model"] = lmhead_model
            ENGINE_STATE["tokenizer"] = tokenizer
            ENGINE_STATE["metadata"] = metadata
            ENGINE_STATE["causal_mask"] = causal_mask
            ENGINE_STATE["cache_state"] = cache_state
            ENGINE_STATE["loaded_model_id"] = model_id
            ENGINE_STATE["loaded_dir"] = model_dir
            ENGINE_STATE["stop_token_ids"] = anemll_chat.build_stop_token_ids(tokenizer)

            print(f"✅ {model_id} loaded to ANE successfully.", flush=True)
            return True
        except Exception as e:
            import traceback
            print(f"Error loading model {model_id}: {e}")
            traceback.print_exc()
            return False


def unload_engine():
    """Release the current model from memory."""
    ENGINE_STATE["embed_model"] = None
    ENGINE_STATE["ffn_models"] = None
    ENGINE_STATE["lmhead_model"] = None
    ENGINE_STATE["tokenizer"] = None
    ENGINE_STATE["metadata"] = None
    ENGINE_STATE["causal_mask"] = None
    ENGINE_STATE["cache_state"] = None
    ENGINE_STATE["loaded_model_id"] = None
    ENGINE_STATE["loaded_dir"] = None
    ENGINE_STATE["stop_token_ids"] = set()


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


async def generate_sse_stream(prompt: str):
    """Generate tokens as Server-Sent Events (for the built-in chat UI)."""
    try:
        print(f"DEBUG: Starting SSE stream for prompt: {prompt[:50]}...")
        tokenizer = ENGINE_STATE["tokenizer"]
        metadata = ENGINE_STATE["metadata"]
        embed_model = ENGINE_STATE["embed_model"]
        ffn_models = ENGINE_STATE["ffn_models"]
        lmhead_model = ENGINE_STATE["lmhead_model"]
        causal_mask = ENGINE_STATE["causal_mask"]
        state = ENGINE_STATE["cache_state"]
        stop_ids = ENGINE_STATE["stop_token_ids"]

        if not tokenizer:
            print("ERROR: Tokenizer is missing in ENGINE_STATE")
            yield f"data: {json.dumps({'token': 'Error: Neural Engine is not ready. Please select and load a model in the Models tab first.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        messages = [{"role": "user", "content": prompt}]
        
        print("DEBUG: Applying chat template...")
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(torch.int32)
        except Exception as te:
            print(f"DEBUG: Chat template failed ({te}), falling back to raw format.")
            formatted = f"[INST] {prompt} [/INST]"
            input_ids = tokenizer(
                formatted, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(torch.int32)

        print(f"DEBUG: Input tokens: {input_ids.size(1)}")
        context_pos = input_ids.size(1)
        context_length = metadata.get("context_length")
        batch_size = metadata.get("batch_size", 64)
        sliding_window = metadata.get("sliding_window", None)
        update_mask = metadata.get("update_mask_prefill", False)

        # 1. Non-blocking Prefill
        await asyncio.to_thread(
            anemll_chat.run_prefill,
            embed_model,
            ffn_models,
            input_ids,
            context_pos,
            context_length,
            batch_size,
            state,
            causal_mask,
            sliding_window,
            single_token_mode=not update_mask,
            use_update_mask=update_mask,
        )

        pos = context_pos
        generated = 0
        while pos < context_length - 1:
            # 2. Non-blocking token generation
            next_token = await asyncio.to_thread(
                anemll_chat.generate_next_token,
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
                yield f"data: {json.dumps({'token': token_str})}\n\n"
            pos += 1
            generated += 1
            if generated > 2048:
                break
        yield "data: [DONE]\n\n"
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR in SSE stream: {e}")
        traceback.print_exc()
        yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


async def generate_ollama_stream(model_name: str, messages: list):
    """Generate tokens as newline-delimited JSON (Ollama format for Zed)."""
    try:
        tokenizer = ENGINE_STATE["tokenizer"]
        metadata = ENGINE_STATE["metadata"]
        embed_model = ENGINE_STATE["embed_model"]
        ffn_models = ENGINE_STATE["ffn_models"]
        lmhead_model = ENGINE_STATE["lmhead_model"]
        causal_mask = ENGINE_STATE["causal_mask"]
        state = ENGINE_STATE["cache_state"]
        stop_ids = ENGINE_STATE["stop_token_ids"]

        if not tokenizer:
            print("ERROR: Tokenizer is missing in generate_ollama_stream")
            yield json.dumps({"error": "Neural Engine is not ready."}) + "\n"
            return

        print(f"DEBUG: Ollama stream starting with {len(messages)} messages...")
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(torch.int32)
        except Exception as te:
            print(f"DEBUG: Ollama chat template failed ({te}), using format fallback.")
            formatted = ""
            for m in messages:
                formatted += f"{m['role']}: {m['content']}\n"
            formatted += "assistant: "
            input_ids = tokenizer(
                formatted, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(torch.int32)

        if input_ids.size(1) == 0:
            print("ERROR: input_ids is empty for Ollama request")
            yield json.dumps({"error": "Empty input sequence generated."}) + "\n"
            return

        print(f"DEBUG: Ollama input tokens: {input_ids.size(1)}")
        context_pos = input_ids.size(1)
        context_length = metadata.get("context_length")
        batch_size = metadata.get("batch_size", 64)
        sliding_window = metadata.get("sliding_window", None)
        update_mask = metadata.get("update_mask_prefill", False)

        # 1. Non-blocking Prefill (Ollama)
        await asyncio.to_thread(
            anemll_chat.run_prefill,
            embed_model,
            ffn_models,
            input_ids,
            context_pos,
            context_length,
            batch_size,
            state,
            causal_mask,
            sliding_window,
            single_token_mode=not update_mask,
            use_update_mask=update_mask,
        )

        pos = context_pos
        generated = 0
        while pos < context_length - 1:
            # 2. Non-blocking token generation (Ollama)
            next_token = await asyncio.to_thread(
                anemll_chat.generate_next_token,
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
                yield (
                    json.dumps(
                        {
                            "model": model_name,
                            "message": {"role": "assistant", "content": token_str},
                            "done": False,
                        }
                    )
                    + "\n"
                )
            pos += 1
            generated += 1
            if generated > 2048:
                break
        yield (
            json.dumps(
                {
                    "model": model_name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
            )
            + "\n"
        )
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR in Ollama stream: {e}")
        traceback.print_exc()
        yield json.dumps({"error": str(e), "model": model_name, "done": True}) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Model Registry ──────────────────────────────────────────────────────────


@app.get("/api/models/registry")
def api_model_registry():
    """Get full model catalog with install status."""
    result = []
    for entry in MODEL_REGISTRY:
        result.append(
            {
                **entry,
                "installed": is_model_installed(entry["id"]),
                "loaded": ENGINE_STATE["loaded_model_id"] == entry["id"],
                "downloading": entry["id"] in DOWNLOAD_PROGRESS
                and DOWNLOAD_PROGRESS[entry["id"]]["status"]
                in ("downloading", "compiling"),
            }
        )
    return {"models": result}


@app.get("/api/models/installed")
def api_installed_models():
    """Get only installed models."""
    return {"models": get_installed_models()}


@app.get("/api/models/progress/{model_id}")
def api_download_progress(model_id: str):
    """Get download/compile progress for a model."""
    if model_id in DOWNLOAD_PROGRESS:
        return DOWNLOAD_PROGRESS[model_id]
    return {"status": "idle", "progress": 0, "message": ""}


@app.post("/api/models/download")
async def api_download_model(request: Request):
    """Trigger model download and ANE compilation."""
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)

    if is_model_installed(model_id):
        return {"status": "already_installed"}

    if model_id in DOWNLOAD_PROGRESS and DOWNLOAD_PROGRESS[model_id]["status"] in (
        "downloading",
        "compiling",
    ):
        return {"status": "already_downloading"}

    # Run in background thread
    thread = threading.Thread(
        target=download_and_compile_model, args=(model_id,), daemon=True
    )
    thread.start()
    return {"status": "started"}


@app.post("/api/models/delete")
async def api_delete_model(request: Request):
    """Delete a compiled model."""
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)

    # Unload if currently loaded
    if ENGINE_STATE["loaded_model_id"] == model_id:
        unload_engine()

    model_dir = get_model_dir(model_id)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        DOWNLOAD_PROGRESS.pop(model_id, None)
        return {"status": "deleted"}
    return {"status": "not_found"}


@app.post("/api/models/load")
async def api_load_model(request: Request):
    """Load a model into the Neural Engine."""
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)

    success = await asyncio.to_thread(load_model_into_engine, model_id)
    return {"status": "loaded" if success else "error", "model_id": model_id}


@app.post("/api/models/unload")
async def api_unload_model():
    """Unload the current model from memory."""
    unload_engine()
    return {"status": "unloaded"}


# ─── Ollama-Compatible Endpoints ─────────────────────────────────────────────


@app.get("/api/version")
def ollama_version():
    """Ollama-compatible version endpoint."""
    return {"version": "0.3.0"}


@app.get("/api/ps")
def ollama_ps():
    """Ollama-compatible endpoint to show running models."""
    if ENGINE_STATE["loaded_model_id"]:
        model_id = ENGINE_STATE["loaded_model_id"]
        model_info = next(
            (m for m in get_installed_models() if m["id"] == model_id),
            {"id": model_id, "family": "llama", "params": "Unknown"},
        )
        return {
            "models": [
                {
                    "name": model_info["id"],
                    "model": model_info["id"],
                    "size": 0,
                    "digest": "ane-native",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": model_info.get("family", "llama"),
                        "families": [model_info.get("family", "llama")],
                        "parameter_size": str(model_info.get("params", "Unknown")),
                        "quantization_level": "ANE_FP16",
                    },
                    "expires_at": "0001-01-01T00:00:00Z",
                    "size_vram": 0,
                }
            ]
        }
    return {"models": []}


@app.post("/api/show")
async def ollama_show(request: Request):
    """Ollama-compatible endpoint to show model details."""
    try:
        data = await request.json()
        model_name = data.get("name", "")
    except Exception:
        model_name = ""

    # Find the model
    models = get_installed_models()
    model_info = next((m for m in models if m["id"] == model_name), None)

    if model_info:
        return {
            "license": "unknown",
            "modelfile": f"FROM {model_name}",
            "parameters": model_info.get("params", "Unknown"),
            "template": "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant:",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": model_info.get("family", "llama"),
                "families": [model_info.get("family", "llama")],
                "parameter_size": str(model_info.get("params", "Unknown")),
                "quantization_level": "ANE_FP16",
            },
            "model_info": {
                "general.architecture": "transformer",
                "general.file_type": 0,
            },
        }

    return {"error": f"model '{model_name}' not found"}


@app.get("/api/tags")
def ollama_tags():
    """Ollama-compatible model listing for Zed Editor."""
    installed = get_installed_models()
    return {
        "models": [
            {
                "name": m["id"],
                "model": m["id"],
                "details": {
                    "family": m["family"],
                    "parameter_size": m["params"],
                    "quantization_level": "ANE_FP16",
                },
            }
            for m in installed
        ]
    }


# ─── Chat ─────────────────────────────────────────────────────────────────────


@app.post("/api/chat")
@app.post("/api/chat/stream")
@app.post("/api/generate")
async def api_chat(request: Request):
    """Unified chat endpoint supporting both SSE and Ollama NDJSON."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    # Auto-load model if needed (NON-BLOCKING)
    model_name = data.get("model")
    if model_name:
        if ENGINE_STATE["loaded_model_id"] != model_name and is_model_installed(model_name):
            await asyncio.to_thread(load_model_into_engine, model_name)
    
    # Final fallback if still nothing loaded
    if not ENGINE_STATE["loaded_model_id"]:
        installed = get_installed_models()
        if installed:
            await asyncio.to_thread(load_model_into_engine, installed[0]["id"])
        else:
            return JSONResponse(
                {"error": "No models installed. Please download a model in the Models tab first."},
                status_code=400,
            )

    messages = data.get("messages")
    prompt = data.get("prompt")

    if messages is not None:
        print(f"DEBUG: Processing Ollama chat request with {len(messages)} messages.")
        return StreamingResponse(
            generate_ollama_stream(ENGINE_STATE["loaded_model_id"], messages),
            media_type="application/x-ndjson",
        )

    prompt = prompt or ""
    print(f"DEBUG: Processing native chat request with prompt.")
    return StreamingResponse(
        generate_sse_stream(prompt), media_type="text/event-stream"
    )


# ─── API Server Config ───────────────────────────────────────────────────────


@app.get("/api/server/status")
def api_server_status():
    local_ip = get_local_ip()
    return {
        "running": API_CONFIG["running"],
        "port": API_CONFIG["port"],
        "host": API_CONFIG["host"],
        "network_ip": local_ip,
        "loaded_model": ENGINE_STATE["loaded_model_id"],
        "ollama_url": f"http://{local_ip}:{API_CONFIG['port']}",
    }


@app.post("/api/server/config")
async def api_server_config(request: Request):
    data = await request.json()
    if "port" in data:
        API_CONFIG["port"] = int(data["port"])
    if "host" in data:
        API_CONFIG["host"] = data["host"]
    return API_CONFIG


# ─── System Info ──────────────────────────────────────────────────────────────


@app.get("/api/system/info")
def api_system_info():
    import platform

    mem = psutil.virtual_memory()
    disk = shutil.disk_usage(MODELS_DIR)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_used_gb": round(mem.used / (1024**3), 1),
        "ram_available_gb": round(mem.available / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "models_dir": MODELS_DIR,
        "loaded_model": ENGINE_STATE["loaded_model_id"],
        "network_ip": get_local_ip(),
    }


# ─── Static UI ────────────────────────────────────────────────────────────────


@app.get("/")
def serve_ui(request: Request):
    """Serve UI for browsers, JSON for API clients."""
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
            "name": "ANE Studio Server",
            "version": "0.3.0",
            "ollama_compatible": True,
            "endpoints": {
                "models": "/api/tags",
                "chat": "/api/chat",
                "generate": "/api/generate",
                "version": "/api/version",
                "running": "/api/ps",
                "model_details": "/api/show",
            },
        }

    html_path = os.path.join(APP_DIR, "ui", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


# ─── Entry Point ──────────────────────────────────────────────────────────────


def run_server(port: int = 11436, host: str = "0.0.0.0"):
    """Start the Uvicorn server (callable from app.py or CLI)."""
    API_CONFIG["port"] = port
    API_CONFIG["host"] = host
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
