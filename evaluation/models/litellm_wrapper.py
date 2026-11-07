# evaluation/models/litellm_wrapper.py
"""Unified LiteLLM wrapper for both local and API models"""

import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import base64
import logging
import subprocess
import time
import requests
import yaml
from typing import List, Dict, Optional, Union, Any
from PIL import Image
import io
import litellm
from litellm import completion
from pydantic import ValidationError
# litellm._turn_on_debug() # for debugging

from .base_model import BaseModelWrapper
from .media_processing import load_image_from_path, load_video_frames_from_path
from .response_formats import get_response_format, get_evaluation_format
from .prompt_formatters import get_prompt_formatter

logger = logging.getLogger(__name__)

# Model pricing information (per 1M tokens)
MODEL_PRICING = {
    # OpenAI models (with provider prefix)
    "openai/gpt-4o": {"input": 2.50, "output": 10.00, "image": 0.0001},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "image": 0.0001},
    "openai/gpt-4.1": {"input": 2.50, "output": 10.00, "image": 0.0001},
    "openai/gpt-4.1-mini": {"input": 0.15, "output": 0.60, "image": 0.0001},
    
    # Anthropic models (with provider prefix)
    "anthropic/claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "image": 0.0},
    "anthropic/claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25, "image": 0.0},
    
    # Google models (with provider prefix)
    "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00, "image": 0.0},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30, "image": 0.0},
    
    # Local models (free) - Small models for testing
    "vllm/HuggingFaceTB/SmolVLM2-256M-Video-Instruct": {"input": 0.0, "output": 0.0, "image": 0.0},
    "vllm/HuggingFaceTB/SmolVLM2-500M-Video-Instruct": {"input": 0.0, "output": 0.0, "image": 0.0},
    "vllm/Qwen/Qwen2.5-VL-3B-Instruct": {"input": 0.0, "output": 0.0, "image": 0.0},
    "vllm/Qwen/Qwen2.5-VL-7B-Instruct": {"input": 0.0, "output": 0.0, "image": 0.0},
    "vllm/llava-hf/llava-1.5-7b-hf": {"input": 0.0, "output": 0.0, "image": 0.0},
}

class LiteLLMWrapper(BaseModelWrapper):
    """Unified wrapper for both local and API models using LiteLLM"""

    # Class variables to track server processes per port
    _servers = {}  # port -> {"process": ..., "model": ..., "log_handle": ..., "starting": False}
    _active_models = {}  # Track active model instances by model_id
    _model_instances = {}  # Cache model instances to avoid reloading
    _query_cache = {}  # Cache all queries for debugging (successful and failed)

    # Legacy class variables for backwards compatibility
    _server_process = None
    _server_model = None
    _server_log_handle = None
    _server_starting = False

    @classmethod
    def _load_model_config(cls):
        """Load model vision/server config from YAML (cached at class level)."""
        if hasattr(cls, '_model_config_cache'):
            return cls._model_config_cache
        config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
        with open(config_path) as f:
            cls._model_config_cache = yaml.safe_load(f)
        return cls._model_config_cache

    @classmethod
    def _get_model_vision_config(cls, model_name: str) -> dict:
        """Get vision config for a model from the YAML, falling back to defaults."""
        config = cls._load_model_config()
        models = config.get("models", {})
        defaults = config.get("defaults", {})
        return models.get(model_name, defaults)


    @classmethod
    def get_or_create_model(cls,
                           model_id: str,
                           api_base: Optional[str] = None,
                           api_key: Optional[str] = None,
                           max_tokens: int = 512,
                           temperature: float = 0.1,
                           use_structured_output: bool = True,
                           auto_start_server: bool = True,
                           port: int = 8002,
                           allowed_media_path: Optional[str] = None,
                           max_model_len: Optional[int] = None,
                           enforce_eager: bool = True,
                           weights_path: Optional[str] = None,
                           **kwargs):
        """Get existing model instance or create new one with caching"""
        # Create a cache key based on model configuration
        cache_key = f"{model_id}_{api_base}_{api_key}_{max_tokens}_{temperature}_{use_structured_output}_{port}_{weights_path}"

        # Return cached instance if it exists
        if cache_key in cls._model_instances:
            logger.info(f"Reusing cached model instance for {model_id}")
            return cls._model_instances[cache_key]

        # Create new instance
        instance = cls(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            use_structured_output=use_structured_output,
            auto_start_server=auto_start_server,
            port=port,
            allowed_media_path=allowed_media_path,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            weights_path=weights_path,
            **kwargs
        )

        # Cache the instance
        cls._model_instances[cache_key] = instance
        cls._active_models[model_id] = instance

        return instance

    def __init__(self,
                 model_id: str,
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 use_structured_output: bool = True,
                 auto_start_server: bool = True,
                 gpu_memory_utilization: float = 0.8,
                 port: int = 8002,
                 allowed_media_path: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 weights_path: Optional[str] = None,
                 **kwargs):
        super().__init__(model_id, **kwargs)

        self.model_id = model_id
        self.weights_path = weights_path
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_structured_output = use_structured_output
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.max_model_len = max_model_len  # Context length for vLLM server (needs to fit images/videos)
        # Resolve symlinks for allowed media path
        self.allowed_media_path = os.path.realpath(allowed_media_path) if allowed_media_path else None

        # Determine if this is a local model first
        self.is_local_model = self._is_local_model()

        # Validate API key for remote models
        if not self.is_local_model:
            self._validate_api_key()

        # Setup LiteLLM client
        self._setup_client()

        # Initialize cost tracking
        self.total_cost = 0.0
        self.total_calls = 0

        # Auto-start server for local models if needed
        if self.is_local_model and auto_start_server:
            self._ensure_server_running()

        # Cache video_only flag from model config
        actual_model = model_id.replace("vllm/", "")
        self.video_only = self._get_model_vision_config(actual_model).get("video_only", False)

        logger.info(f"Initialized LiteLLM wrapper for {model_id} (local: {self.is_local_model})")


    def _setup_client(self):
        """Setup LiteLLM client with appropriate configuration"""
        # Use LiteLLM's native completion function
        self.client = completion
        logger.info("Using LiteLLM completion with native structured output support")
    
    def _validate_api_key(self):
        """Check that the required API key env var is set for remote models."""
        key_map = {
            "openai/": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
            "gemini/": ("GOOGLE_API_KEY", "https://aistudio.google.com/apikey"),
            "anthropic/": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/"),
        }
        for prefix, (env_var, url) in key_map.items():
            if self.model_id.startswith(prefix):
                key = self.api_key or os.environ.get(env_var, "")
                if not key or key.startswith("your-"):
                    raise ValueError(
                        f"Missing API key for {self.model_id}. "
                        f"Set {env_var} or pass --api-key. Get one at {url}"
                    )
                return
        # Unknown provider — let litellm handle it
        logger.warning(f"No API key validation for provider in {self.model_id}")

    def _is_local_model(self) -> bool:
        """Determine if this is a local model (vLLM) or API model"""
        # Common local model patterns
        local_patterns = [
            "vllm/",
        ]
        
        return any(pattern in self.model_id.lower() for pattern in local_patterns)
    
    def _check_server_running(self, port=8002):
        """Check if vLLM server is running and healthy"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                # Also check if models endpoint is accessible
                models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
                return models_response.status_code == 200
            return False
        except requests.exceptions.RequestException:
            return False
    
    def _check_server_health(self, port=8002):
        """Check detailed server health and return status info"""
        try:
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            
            if health_response.status_code == 200 and models_response.status_code == 200:
                models_data = models_response.json()
                return {
                    "healthy": True,
                    "models": [model["id"] for model in models_data.get("data", [])],
                    "server_model": models_data.get("data", [{}])[0].get("id") if models_data.get("data") else None
                }
            else:
                return {"healthy": False, "error": f"Health: {health_response.status_code}, Models: {models_response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"healthy": False, "error": str(e)}
    
    def _start_server(self, model_id, port=8002, allowed_media_path=None, gpu_memory_utilization=0.8, max_model_len=None, weights_path=None):
        """Start vLLM server for the given model.

        `weights_path`, when set, overrides the path passed to `vllm serve`
        with a local checkpoint directory while keeping `model_id` as the
        served-model-name and the key for `_get_model_vision_config` lookups.
        Used to evaluate fine-tuned merged-weights checkpoints under their
        base model's identity.
        """
        # Extract the actual model name from the vllm/ prefix
        if model_id.startswith("vllm/"):
            actual_model = model_id[5:]  # Remove "vllm/" prefix
        else:
            actual_model = model_id

        # `actual_model` stays the base model name (used for config lookups
        # and --served-model-name); `serve_target` is the positional arg to
        # `vllm serve` (a HF id by default, or a local checkpoint dir when
        # weights_path is provided).
        if weights_path:
            print(f"Starting vLLM server with weights: {weights_path} "
                  f"(served as {actual_model})")
            serve_target = weights_path
        else:
            print(f"Starting vLLM server with model: {actual_model}")
            serve_target = actual_model

        # Auto-detect whether to pass --max-model-len.
        # We pass it to cap memory for models with huge native contexts (e.g.
        # Qwen 128k), but omit it for models whose native limit is already
        # smaller (e.g. LLaVA-NeXT-Video 10240) so vLLM can auto-derive it.
        if max_model_len is not None:
            model_cfg = self._get_model_vision_config(actual_model)
            native_hint = model_cfg.get("max_position_embeddings")
            if native_hint is None:
                print(f"No max_position_embeddings in model_config.yaml for {actual_model}. "
                      f"Omitting --max-model-len to let vLLM auto-detect.")
                max_model_len = None
            elif native_hint <= max_model_len:
                print(f"Model native context ({native_hint}) <= requested max_model_len "
                      f"({max_model_len}). Omitting --max-model-len to let vLLM auto-detect.")
                max_model_len = None
                self.max_model_len = native_hint  # Conservative estimate for frame budget
            else:
                print(f"Model native context ({native_hint}) > requested max_model_len "
                      f"({max_model_len}). Passing --max-model-len to cap memory.")

        cmd = [
            "vllm", "serve", serve_target,
            "--served-model-name", actual_model,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--host", "0.0.0.0",
            "--no-enable-log-requests",  # Reduce logging overhead
            "--trust-remote-code",  # Allow custom code in models
            "--enforce-eager",  # Disable CUDA graphs (fixes hanging on older GPUs without breaking EngineCore)
            "--mm-processor-cache-gb", "0",
            "--no-enable-prefix-caching",  # Disable prefix caching to prevent encoder cache accumulation with videos
        ]

        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Per-model extra server args from model_config.yaml
        model_cfg = self._get_model_vision_config(actual_model)
        max_num_batched_tokens = model_cfg.get("max_num_batched_tokens")
        if max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

        # For thinking models: either enable reasoning parser to separate <think>
        # content, or disable thinking entirely via chat template kwargs.
        if model_cfg.get("disable_thinking", False):
            import json as _json
            cmd.extend(["--default-chat-template-kwargs",
                         _json.dumps({"enable_thinking": False})])
        else:
            reasoning_parser = model_cfg.get("reasoning_parser")
            if reasoning_parser:
                import json as _json
                cmd.extend(["--structured-outputs-config",
                             _json.dumps({"reasoning_parser": reasoning_parser})])

        # Pre-allocate multimodal buffers only when this server will actually
        # receive images/videos (indicated by allowed_media_path being set).
        # Skipping this for text-only servers (e.g. the LLM judge) avoids
        # wasting GPU memory on unused video frame buffers.
        if allowed_media_path:
            import vllm
            vllm_version = tuple(int(x) for x in vllm.__version__.split(".")[:2])
            if vllm_version >= (0, 15):
                num_frames = self._compute_max_frames()
                # For models where we don't control fps (no use_max_pixels),
                # vLLM's processor picks its own frame count (typically 8).
                # The encoder cache must be large enough to fit that.
                if not model_cfg.get("use_max_pixels", False):
                    vllm_default = model_cfg.get("default_num_frames", 8)
                    num_frames = max(num_frames, vllm_default)
                max_res = model_cfg.get("max_video_resolution", 512)
                if model_cfg.get("video_only", False):
                    # Video-only models (e.g. LLaVA-NeXT-Video): no image support
                    mm_limit = f'{{"video": {{"count": 1, "num_frames": {num_frames}, "width": {max_res}, "height": {max_res}}}}}'
                else:
                    mm_limit = f'{{"image": 1, "video": {{"count": 1, "num_frames": {num_frames}, "width": {max_res}, "height": {max_res}}}}}'
            else:
                mm_limit = '{"image": 1, "video": 1}'
            cmd.extend(["--limit-mm-per-prompt", mm_limit])

        # Add allowed media path if provided
        if allowed_media_path:
            cmd.extend(["--allowed-local-media-path", allowed_media_path])

        print(f"Server command: {' '.join(cmd)}")

        # Create log file for server output (absolute path so it works from any cwd, e.g. sbatch)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        log_file = os.path.join(log_dir, f"vllm_server_{port}_{job_id}.log")
        print(f"Server output will be logged to: {log_file}")

        # Verify port is free before starting
        if self._check_server_running(port):
            print(f"WARNING: Port {port} still has a server responding before launch! Waiting for it to die...")
            for wait_i in range(15):
                time.sleep(2)
                if not self._check_server_running(port):
                    print(f"Port {port} is now free after {(wait_i+1)*2}s")
                    break
            else:
                print(f"ERROR: Port {port} still occupied after 30s. Cannot start new server.")
                return None
        
        # Allow vLLM to use memory more flexibly and avoid fragmentation issues
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        # Auto-restart EngineCore when it dies (known vLLM 0.15.x bug
        # where EngineCore_DP0 subprocess dies silently after ~50-100 requests).
        env["VLLM_KEEP_ALIVE_ON_ENGINE_DEATH"] = "1"

        # Start server process with output capture
        log_handle = open(log_file, "w")
        LiteLLMWrapper._server_log_handle = log_handle
        process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)

        # Fix C: Immediately check if the process died (e.g., vllm not on PATH)
        time.sleep(2)
        exit_code = process.poll()
        if exit_code is not None:
            log_handle.close()
            print(f"ERROR: vllm server process exited immediately with code {exit_code}")
            # Print log contents to help diagnose
            try:
                with open(log_file, "r") as f:
                    log_contents = f.read()
                if log_contents:
                    print(f"Server log:\n{log_contents[:2000]}")
                else:
                    print("Server log is EMPTY - vllm likely not found on PATH")
            except Exception:
                pass
            return None

        print(f"Server process started (PID: {process.pid})")

        # Wait for server to be ready
        print("Waiting for server to start...", flush=True)
        for i in range(600):  # Wait up for model loading including weights (10 min)
            # Check if process is still alive
            if process.poll() is not None:
                log_handle.close()
                print(f"\nERROR: Server process died during startup with exit code {process.returncode}")
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                    print(f"Last 30 lines of server log:")
                    for line in lines[-30:]:
                        print(f"  {line.rstrip()}")
                except Exception:
                    pass
                return None

            # Detect EngineCore crash early: the APIServer process may hang
            # instead of exiting when the EngineCore dies during init.
            try:
                log_handle.flush()
                with open(log_file, "r") as lf:
                    if "EngineDeadError" in lf.read():
                        log_handle.close()
                        print(f"\nERROR: vLLM EngineCore crashed during startup (EngineDeadError)")
                        process.terminate()
                        return None
            except Exception:
                pass

            if self._check_server_running(port):
                # Fix B: Verify the CORRECT model is being served, not a stale server
                health = self._check_server_health(port)
                server_model = health.get("server_model", "")
                if server_model == actual_model:
                    print(f"\nServer is ready (serving: {server_model})")
                    # Send a warmup request to ensure model is loaded
                    self._warmup_model(port)
                    return process
                else:
                    # A different model is responding - old server still dying
                    if i % 5 == 0:
                        print(f"\n  Port responding but wrong model (got: {server_model}, want: {actual_model}), waiting...", flush=True)

            time.sleep(2)
            if i % 5 == 0:  # Print every 10 seconds
                print(f" {i*2}s", end="", flush=True)

        # Server failed to start - print log contents
        print(f"\nERROR: Server failed to start within 4 minutes")
        log_handle.close()

        # Print last 50 lines of log file to help diagnose
        print(f"\nLast 50 lines of server log ({log_file}):")
        print("=" * 80)
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
        print("=" * 80)

        process.terminate()
        return None
    
    def _warmup_model(self, port=8002):
        """Send a warmup request to ensure the model is fully loaded"""
        try:
            print("Sending warmup request to load model...")
            warmup_payload = {
                "model": self.model_id.replace("vllm/", ""),  # Remove vllm prefix
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                json=warmup_payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                print("✓ Model warmup completed")
            else:
                print(f"⚠ Warmup failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠ Warmup failed: {e}")
    
    def _get_server_state(self) -> dict:
        """Get server state for this instance's port"""
        if self.port not in LiteLLMWrapper._servers:
            LiteLLMWrapper._servers[self.port] = {
                "process": None,
                "model": None,
                "log_handle": None,
                "starting": False
            }
        return LiteLLMWrapper._servers[self.port]

    def _ensure_server_running(self):
        """Ensure vLLM server is running for this model on this port"""
        server_state = self._get_server_state()
        logger.info(f"[_ensure_server_running] Checking server for model: {self.model_id} on port {self.port}")
        logger.info(f"[_ensure_server_running] Current server model on port {self.port}: {server_state['model']}")

        # If server is currently starting on this port, wait for it
        if server_state["starting"]:
            logger.info(f"Another process is starting the server on port {self.port}, waiting...")
            max_wait = 240  # Wait up to 4 minutes
            for i in range(max_wait):
                if not server_state["starting"]:
                    logger.info("Server startup by other process completed")
                    break
                time.sleep(1)
                if i % 10 == 0:
                    logger.info(f"Still waiting for server startup... ({i}s)")
            else:
                logger.warning("Timeout waiting for server startup by other process")

        # If we already know this model's server is running, check it's still healthy
        if server_state["model"] == self.model_id:
            logger.info(f"Server state shows server should be running for {self.model_id}, verifying...")
            if self._check_server_running(self.port):
                logger.info(f"Server confirmed running and healthy for {self.model_id} on port {self.port}")
                return
            else:
                logger.warning(f"Server was supposed to be running but health check failed, will restart")
                server_state["model"] = None

        # Check if server is already running on this port
        if self._check_server_running(self.port):
            logger.info(f"Detected a server running on port {self.port}, checking model...")
            # Query the server to see what model it's actually serving
            health = self._check_server_health(self.port)
            if health.get("healthy"):
                server_model = health.get("server_model")
                # Extract just the model name from our model_id (remove vllm/ prefix if present)
                requested_model = self.model_id.replace("vllm/", "")

                if server_model == requested_model:
                    logger.info(f"Server already running with correct model: {server_model}")
                    server_state["model"] = self.model_id
                    return
                else:
                    logger.warning(f"Server running with WRONG model (running: {server_model}, requested: {requested_model}), restarting...")
                    self._stop_server(self.port)
            else:
                logger.warning(f"Server health check failed: {health.get('error')}")
                self._stop_server(self.port)
        else:
            logger.info(f"No server detected on port {self.port}")

        # Set flag to prevent concurrent starts
        server_state["starting"] = True
        try:
            # Start server for this model
            logger.info(f"Starting server for model: {self.model_id} on port {self.port}")
            logger.info(f"Allowed media path: {self.allowed_media_path}")
            logger.info(f"Max model length: {self.max_model_len}")
            process = self._start_server(
                self.model_id,
                port=self.port,
                allowed_media_path=self.allowed_media_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                weights_path=self.weights_path,
            )
            if process:
                server_state["process"] = process
                server_state["model"] = self.model_id
                logger.info(f"Server started successfully for model: {self.model_id} on port {self.port}")
            else:
                raise RuntimeError(f"Failed to start server for model: {self.model_id} on port {self.port}")
        finally:
            # Always clear the flag
            server_state["starting"] = False

    @staticmethod
    def _pids_on_visible_gpus():
        """PIDs of compute processes on the GPUs in CUDA_VISIBLE_DEVICES
        (all GPUs nvidia-smi can see if unset). None if nvidia-smi fails."""
        try:
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

        index_to_uuid = {}
        for line in gpus.strip().splitlines():
            index, uuid = [x.strip() for x in line.split(",")]
            index_to_uuid[index] = uuid

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            uuids = set(index_to_uuid.values())
        else:
            # Entries are GPU indices or UUIDs.
            uuids = {index_to_uuid.get(d.strip(), d.strip())
                     for d in visible.split(",") if d.strip()}

        pids = set()
        for line in apps.strip().splitlines():
            pid, uuid = [x.strip() for x in line.split(",")]
            if uuid in uuids:
                pids.add(pid)
        return pids

    @classmethod
    def _stop_server(cls, port=8002):
        """Stop the vLLM server by killing its entire process tree, then verify."""
        print("Stopping vLLM server...")

        current_pid = str(os.getpid())

        # Step 1: Kill the stored server process tree (covers EngineCore & workers)
        server_state = cls._servers.get(port, {})
        server_proc = server_state.get("process")
        if server_proc and server_proc.poll() is None:
            server_pid = str(server_proc.pid)
            # pkill -P kills children; then kill the parent
            subprocess.run(["pkill", "-9", "-P", server_pid], capture_output=True)
            subprocess.run(["kill", "-9", server_pid], capture_output=True)
            server_proc.wait()
            print(f"Killed server process tree (root PID {server_pid})")

        # Step 2: Kill any orphaned vLLM processes (e.g. from a previous crash)
        # that hold memory on the GPU(s) this run uses. Restricting to our
        # GPUs keeps us from killing vLLM jobs on other GPUs of a shared node.
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints|VLLM::EngineCore"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            vllm_pids = [p for p in result.stdout.strip().split('\n') if p != current_pid]
            gpu_pids = cls._pids_on_visible_gpus()
            if gpu_pids is not None:
                vllm_pids = [p for p in vllm_pids if p in gpu_pids]
            if vllm_pids:
                print(f"Found {len(vllm_pids)} orphaned vLLM process(es): {', '.join(vllm_pids)}")
                for pid in vllm_pids:
                    subprocess.run(["kill", "-9", pid], capture_output=True)

        # Step 3: Wait for port to be freed (up to 15s)
        for wait_i in range(15):
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            pids_on_port = [p for p in result.stdout.strip().split('\n') if p and p != current_pid]
            if not pids_on_port:
                print(f"Port {port} is now free (after {wait_i}s)")
                break
            time.sleep(1)
        else:
            print(f"WARNING: Port {port} still in use after 15s (PIDs: {', '.join(pids_on_port)})")
            for pid in pids_on_port:
                subprocess.run(["kill", "-9", pid], capture_output=True)
            time.sleep(2)

        print("Server cleanup complete for port", port)

        # Clean up server state for this port
        if port in cls._servers:
            if server_state.get("log_handle"):
                server_state["log_handle"].close()
            cls._servers[port] = {
                "process": None,
                "model": None,
                "log_handle": None,
                "starting": False
            }

        # Also clean legacy class variables for backwards compatibility
        cls._server_process = None
        cls._server_model = None
        cls._server_starting = False
        if cls._server_log_handle:
            cls._server_log_handle.close()
            cls._server_log_handle = None

        time.sleep(1)
        print(f"Server cleanup complete for port {port}")
    
    @classmethod
    def _save_query_to_cache(cls, query_id: str, model_id: str, prompt: str, 
                            response: str = None, error: str = None, 
                            query_type: str = "generation"):
        """Save query details to the query cache for debugging"""
        import datetime
        cls._query_cache[query_id] = {
            "timestamp": datetime.datetime.now(),
            "model_id": model_id,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "full_prompt": prompt,
            "response": response,
            "error": error,
            "query_type": query_type,
            "success": error is None
        }
    
    @classmethod
    def get_query_cache(cls):
        """Get the query cache for debugging"""
        return cls._query_cache

    @classmethod
    def clear_query_cache(cls):
        """Clear the query cache"""
        cls._query_cache.clear()
        logger.info("Query cache cleared")
    
    @classmethod
    def get_active_models(cls):
        """Get list of currently active model instances"""
        return list(cls._active_models.keys())
    
    @classmethod
    def shutdown_all_models(cls):
        """Shutdown all model instances and stop every server this process
        started. _stop_server() defaults to port 8002 (the primary model's
        port) — calling it with no args here only ever cleaned that up, so
        a judge server on a different port (e.g. 8003) was silently left
        running after exit, orphaned, piling up across repeated invocations
        in the same session (each new invocation has fresh in-memory state
        and doesn't know about the previous one's server)."""
        for port in list(cls._servers.keys()):
            cls._stop_server(port)
        cls._active_models.clear()
        cls._model_instances.clear()
        logger.info("All models shutdown")
    
    @classmethod
    def get_server_status(cls):
        """Get current server status and health"""
        if not cls._server_process:
            return {"running": False, "model": None, "health": None}
        
        # Check if process is still alive
        if cls._server_process.poll() is not None:
            return {"running": False, "model": cls._server_model, "health": "Process terminated"}
        
        # Check server health
        health_info = cls._check_server_health()
        return {
            "running": True,
            "model": cls._server_model,
            "health": health_info
        }
    
    @classmethod
    def _check_server_health(cls, port=8002):
        """Check detailed server health and return status info"""
        try:
            health_response = requests.get(f"http://localhost:{port}/health", timeout=2)
            models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            
            if health_response.status_code == 200 and models_response.status_code == 200:
                models_data = models_response.json()
                return {
                    "healthy": True,
                    "models": [model["id"] for model in models_data.get("data", [])],
                    "server_model": models_data.get("data", [{}])[0].get("id") if models_data.get("data") else None
                }
            else:
                return {"healthy": False, "error": f"Health: {health_response.status_code}, Models: {models_response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"healthy": False, "error": str(e)}
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Get pricing information for the current model"""
        return MODEL_PRICING.get(self.model_id, {
            "input": 0.0,  # Unknown model, assume free
            "output": 0.0,
            "image": 0.0
        })
    
    def _estimate_cost(self, 
                      input_tokens: int, 
                      output_tokens: int, 
                      num_images: int = 0) -> float:
        """Estimate cost for a model call"""
        pricing = self._get_model_pricing()
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        image_cost = num_images * pricing["image"]
        
        return input_cost + output_cost + image_cost
    
    def _print_cost_estimate(self, 
                           input_tokens: int, 
                           output_tokens: int, 
                           num_images: int = 0,
                           call_type: str = "generation"):
        """Print cost estimate for a model call"""
        if self.is_local_model:
            print(f"{call_type.capitalize()} cost: FREE (local model)")
            return
        
        cost = self._estimate_cost(input_tokens, output_tokens, num_images)
        pricing = self._get_model_pricing()
        
        print(f"{call_type.capitalize()} cost estimate:")
        print(f"   Input tokens: {input_tokens:,} (${input_tokens/1_000_000 * pricing['input']:.6f})")
        print(f"   Output tokens: {output_tokens:,} (${output_tokens/1_000_000 * pricing['output']:.6f})")
        if num_images > 0:
            print(f"   Images/videos: {num_images} (${num_images * pricing['image']:.6f})")
        print(f"   Total: ${cost:.6f}")
        
        # Update running totals
        self.total_cost += cost
        self.total_calls += 1
        print(f"   Running total: ${self.total_cost:.6f} ({self.total_calls} calls)")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for this model instance"""
        return {
            "model_id": self.model_id,
            "is_local": self.is_local_model,
            "total_cost": self.total_cost,
            "total_calls": self.total_calls,
            "average_cost_per_call": self.total_cost / self.total_calls if self.total_calls > 0 else 0.0,
            "pricing": self._get_model_pricing()
        }
    
    def _prepare_messages(self,
                         prompt: str,
                         media_path: Optional[str] = None,
                         system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """Prepare messages for the model with media if provided"""
        # Reset per-request video fps override
        self._current_video_fps = None

        # Some models (e.g. Molmo2, LLaVA-NeXT-Video) don't support system
        # messages — their chat template requires strict user/assistant
        # alternation.  For these, prepend the system prompt to the user text.
        actual_model = self.model_id.replace("vllm/", "")
        model_cfg = self._get_model_vision_config(actual_model)
        if system_prompt and model_cfg.get("no_system_prompt", False):
            prompt = f"{system_prompt}\n\n{prompt}"
            system_prompt = None

        # Build the system message prefix (if any)
        system_messages = []
        if system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})

        if media_path:
            media_path = str(media_path)
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Media file not found: {media_path}")

            # Compute dynamic video fps if this is a video file
            if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self._current_video_fps = self._compute_video_fps(media_path)

            # Both local vLLM and API models use OpenAI multimodal format
            # vLLM's OpenAI-compatible API expects image_url content type, not custom <img> tags
            return system_messages + self._prepare_openai_format_messages(prompt, media_path)
        else:
            # Text-only prompt
            return system_messages + [{"role": "user", "content": prompt}]
    
    def _process_media_for_prompt_formatter(self, media_path: str) -> Union[str, List[str]]:
        """Process media for use with prompt formatters"""
        if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # For images, use local file path for vLLM, base64 for API
            if self.is_local_model:
                return media_path
            else:
                return load_image_from_path(media_path)
        elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # For videos, pass video file path directly - the model handles frame extraction
            return media_path
        else:
            raise ValueError(f"Unsupported media type: {media_path}")
    
    def _format_prompt_with_media(self, prompt: str, media_data: Union[str, List[str]]) -> str:
        """Format prompt using model-specific formatter"""
        # Determine model family
        model_family = self._get_model_family()
        formatter = get_prompt_formatter(model_family)
        
        return formatter(prompt, media_data)
    
    def _get_model_family(self) -> str:
        """Determine model family from model_id"""
        model_id_lower = self.model_id.lower()
        if "qwen" in model_id_lower:
            return "qwen"
        elif "llava" in model_id_lower:
            return "llava"
        else:
            return "generic"
    
    def _prepare_openai_format_messages(self, prompt: str, media_path: str) -> List[Dict[str, Any]]:
        """Prepare messages in OpenAI format for both local vLLM and API models"""
        messages = []

        if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # Image processing
            if self.is_local_model:
                # For local vLLM, use file:// URL with realpath to resolve symlinks
                real_path = os.path.realpath(media_path)
                image_url = f"file://{real_path}"
                logger.debug(f"Using local file URL for image: {image_url}")
            else:
                # For API models, use base64 encoding
                image_url = self._process_image_for_litellm(media_path)
                if not image_url:
                    raise ValueError(f"Failed to process image: {media_path}")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
        elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video processing
            if self.is_local_model:
                # For local vLLM, use file:// URL with realpath to resolve symlinks
                real_path = os.path.realpath(media_path)
                video_url = f"file://{real_path}"
                logger.debug(f"Using local file URL for video: {video_url}")
            else:
                # For API models, use base64 encoding
                video_url = self._process_video_for_litellm(media_path)
                if not video_url:
                    raise ValueError(f"Failed to process video: {media_path}")

            # Gemini expects "file" content type for videos; OpenAI needs
            # frames as image_url entries; local vLLM uses video_url.
            if self.model_id.startswith("gemini/"):
                # Detect MIME type from extension
                ext = os.path.splitext(media_path)[1].lower()
                mime = {"mp4": "video/mp4", "avi": "video/x-msvideo",
                        "mov": "video/quicktime", "mkv": "video/x-matroska"}.get(ext.lstrip("."), "video/mp4")
                video_content = [{"type": "file", "file": {"file_id": video_url, "format": mime}}]
            elif self.model_id.startswith("openai/") or self.model_id.startswith("anthropic/"):
                # OpenAI/Anthropic don't support video_url — send as sampled frames
                frame_urls = load_video_frames_from_path(media_path, max_frames=8)
                video_content = [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in frame_urls
                ]
            else:
                video_content = [{"type": "video_url", "video_url": {"url": video_url}}]

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *video_content
                ]
            })
        else:
            raise ValueError(f"Unsupported media type: {media_path}")

        return messages
    
    def _process_image_for_litellm(self, image_path: str) -> Optional[str]:
        """Process image for LiteLLM (use local file path)"""
        # For local models (vLLM), use local file path directly
        if self.is_local_model:
            logger.info(f"Using local file path for image: {image_path}")
            return image_path
        
        # For API models, use base64 encoding
        logger.info(f"Converting image to base64: {image_path}")
        image_data = load_image_from_path(image_path)
        return image_data
    
    def _process_video_for_litellm(self, video_path: str) -> str:
        """Process video for LiteLLM (use local file path)"""
        # For local models (vLLM), use local file path directly
        if self.is_local_model:
            logger.info(f"Using local file path for video: {video_path}")
            return video_path
        
        # Check if it's already a URL
        if video_path.startswith(('http://', 'https://', 'file://')):
            return video_path
        
        # For API models, use base64 encoding
        logger.info(f"Converting video to base64: {video_path}")
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode()
            
            # Determine MIME type based on file extension
            if video_path.lower().endswith('.mp4'):
                mime_type = 'video/mp4'
            elif video_path.lower().endswith('.avi'):
                mime_type = 'video/avi'
            elif video_path.lower().endswith('.mov'):
                mime_type = 'video/quicktime'
            elif video_path.lower().endswith('.mkv'):
                mime_type = 'video/x-matroska'
            else:
                mime_type = 'video/mp4'  # Default
            
            return f"data:{mime_type};base64,{video_base64}"
    
    # def _estimate_tokens_per_frame(self) -> int:
    #     """Estimate visual tokens per video frame from the model's config.

    #     Reads patch_size, spatial_merge_size, and image resolution from the
    #     HuggingFace model config to compute a per-frame token estimate.
    #     Falls back to a conservative default if config can't be read.
    #     """
    #     if hasattr(self, '_cached_tokens_per_frame'):
    #         return self._cached_tokens_per_frame

    #     default_tokens = 1280  # conservative fallback
    #     try:
    #         from transformers import AutoConfig, AutoProcessor
    #         actual_model = self.model_id.replace("vllm/", "")
    #         cfg = AutoConfig.from_pretrained(actual_model, trust_remote_code=True)
    #         vision_cfg = getattr(cfg, 'vision_config', None)

    #         patch_size = getattr(vision_cfg, 'patch_size', None) if vision_cfg else None
    #         image_size = getattr(vision_cfg, 'image_size', None) if vision_cfg else None
    #         spatial_merge = getattr(cfg, 'spatial_merge_size', None)
    #         if spatial_merge is None and vision_cfg:
    #             spatial_merge = getattr(vision_cfg, 'spatial_merge_size', None)

    #         if patch_size:
    #             # Try to get max_pixels from processor for dynamic-resolution models
    #             try:
    #                 proc = AutoProcessor.from_pretrained(actual_model, trust_remote_code=True)
    #                 max_pixels = getattr(proc.image_processor, 'max_pixels', None)
    #             except Exception:
    #                 max_pixels = None

    #             if image_size:
    #                 # Fixed resolution models (InternVL, LLaVA)
    #                 patches = (image_size // patch_size) ** 2
    #             elif max_pixels:
    #                 # Dynamic resolution models (Qwen-VL): use max_pixels as upper bound
    #                 patches = max_pixels // (patch_size ** 2)
    #             else:
    #                 patches = (1080 // patch_size) ** 2  # assume 1080p

    #             tokens = patches
    #             if spatial_merge and spatial_merge > 1:
    #                 tokens = patches // (spatial_merge ** 2)

    #             # Add safety margin (20%)
    #             tokens = int(tokens * 1.2)
    #             logger.info(f"Estimated {tokens} tokens/frame for {actual_model} "
    #                         f"(patch={patch_size}, merge={spatial_merge}, image_size={image_size})")
    #             self._cached_tokens_per_frame = tokens
    #             return tokens

    #     except Exception as e:
    #         logger.warning(f"Could not estimate tokens/frame from config: {e}")

    #     self._cached_tokens_per_frame = default_tokens
    #     return default_tokens

    def _estimate_tokens_per_frame(self) -> int:
        """
        Estimate visual tokens per video frame from model_config.yaml.
        Standardizes on the active spatial budget (512x512).
        """
        if hasattr(self, '_cached_tokens_per_frame'):
            return self._cached_tokens_per_frame

        # --- Constants ---
        min_token_floor = 64

        # --- Look up model config from YAML ---
        actual_model = self.model_id.replace("vllm/", "")
        model_cfg = self._get_model_vision_config(actual_model)

        # Use the model's max_video_resolution (default 512) so the estimate
        # matches the --limit-mm-per-prompt allocation.
        target_side = model_cfg.get("max_video_resolution", 512)
        active_pixels = target_side * target_side

        architecture_type = model_cfg.get("architecture_type", "standard")
        patch_size = model_cfg.get("patch_size", 14)
        merge_factor = model_cfg.get("spatial_merge_size", 2)

        # Resampler models output a fixed number of tokens per frame
        if architecture_type == "resampler":
            resampler_tokens = model_cfg.get("resampler_output_tokens", 96)
            self._cached_tokens_per_frame = resampler_tokens
            logger.info(f"Estimate for {self.model_id}: {resampler_tokens} tokens/frame (resampler)")
            return self._cached_tokens_per_frame

        # --- Standard / visual_tokenizer calculation ---
        # Area per token = (patch_size * merge_factor)^2
        effective_area = (patch_size * merge_factor) ** 2
        base_tokens = active_pixels // effective_area

        # Add a 15% safety margin for newlines and special tokens
        final_estimate = int(base_tokens * 1.15)

        self._cached_tokens_per_frame = max(final_estimate, min_token_floor)

        logger.info(f"Estimate for {self.model_id}: {self._cached_tokens_per_frame} tokens/frame "
                    f"(patch={patch_size}, merge={merge_factor})")

        return self._cached_tokens_per_frame

    def _compute_max_frames(self) -> int:
        """Max video frames that fit within the context budget.

        Reserves a fixed overhead for text prompt + output tokens, then
        divides the remaining token budget by the per-frame cost.
        """
        TEXT_OVERHEAD = 1500  # tokens reserved for prompt + output + special tokens
        if self.max_model_len is None:
            return 64
        tokens_per_frame = self._estimate_tokens_per_frame()
        return max(1, (self.max_model_len - TEXT_OVERHEAD) // tokens_per_frame)

    def _compute_video_fps(self, video_path: str) -> float:
        """Dynamically compute FPS to maximize frames within context budget.

        Returns the highest FPS that fits within max_model_len, capped at
        the video's native FPS (no point sampling faster than the source).
        Falls back to 2.0 if budget can't be determined.
        """
        DEFAULT_FPS = 2.0

        if not self.is_local_model or self.max_model_len is None:
            return DEFAULT_FPS

        try:
            import decord
            vr = decord.VideoReader(video_path)
            total_video_frames = len(vr)
            native_fps = vr.get_avg_fps()
            if native_fps <= 0 or total_video_frames <= 0:
                return DEFAULT_FPS
            duration_seconds = total_video_frames / native_fps
            if duration_seconds <= 0:
                return DEFAULT_FPS

            max_frames = self._compute_max_frames()

            # FPS to fill the budget, capped at native FPS
            target_fps = min(max_frames / duration_seconds, native_fps)
            target_fps = max(0.1, target_fps)  # floor

            logger.info(
                f"Video {os.path.basename(video_path)}: "
                f"duration={duration_seconds:.1f}s, "
                f"max_frames={max_frames} (at {self._estimate_tokens_per_frame()} tok/frame), "
                f"target_fps={target_fps:.2f}, "
                f"actual_frames={int(duration_seconds * target_fps)}"
            )
            return target_fps
        except Exception as e:
            logger.warning(f"Failed to compute video FPS for {video_path}: {e}")
            return DEFAULT_FPS

    def generate_single(self,
                       prompt: str,
                       media_path: Optional[str] = None,
                       system_prompt: Optional[str] = None,
                       **kwargs) -> str:
        """Generate response for a single prompt with image or video path"""
        try:
            messages = self._prepare_messages(prompt, media_path, system_prompt=system_prompt)
            
            # Estimate tokens and images for cost calculation
            estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            estimated_output_tokens = kwargs.get("max_tokens", self.max_tokens)
            num_images = 0
            
            if media_path and os.path.exists(media_path):
                if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    num_images = 1
                elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # OpenAI/Anthropic get 8 frames; others send as single video
                    if self.model_id.startswith(("openai/", "anthropic/")):
                        num_images = 8
                    else:
                        num_images = 1

            # Print cost estimate
            self._print_cost_estimate(
                int(estimated_input_tokens),
                int(estimated_output_tokens),
                num_images,
                "generation"
            )
            
            # Prepare completion parameters
            # For local vLLM servers via HTTP, use openai/ prefix to route
            # through HTTP API (not vllm/ which spawns a new in-process instance)
            api_model_id = self.model_id
            if self.is_local_model and self.model_id.startswith("vllm/"):
                actual_model = self.model_id[5:]  # Remove "vllm/" prefix
                api_model_id = f"openai/{actual_model}"

            completion_params = {
                "model": api_model_id,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }

            # Only set api_base and api_key for local models
            if self.is_local_model:
                completion_params["api_base"] = self.api_base or f"http://localhost:{self.port}/v1"
                completion_params["api_key"] = self.api_key or "EMPTY"
            else:
                # For API models, let LiteLLM handle the routing
                if self.api_key:
                    completion_params["api_key"] = self.api_key

            # Add per-request mm_processor_kwargs for local vLLM only.
            # max_pixels is Qwen-VL specific (smart_resize); fps only needed for video.
            if self.is_local_model:
                actual_model = self.model_id.replace("vllm/", "")
                model_cfg = self._get_model_vision_config(actual_model)
                mm_kwargs = {}
                if model_cfg.get("use_max_pixels", False):
                    # fps and max_pixels are Qwen-VL specific (smart_resize + fps sampling).
                    # Other models (InternVL, llava, MiniCPM, Molmo2) don't support these
                    # kwargs and InternVL in particular returns a 400 if fps is sent.
                    max_res = model_cfg.get("max_video_resolution", 512)
                    mm_kwargs["max_pixels"] = max_res * max_res
                    if self._current_video_fps is not None:
                        mm_kwargs["fps"] = self._current_video_fps
                if mm_kwargs:
                    completion_params["extra_body"] = {"mm_processor_kwargs": mm_kwargs}

            # Generate response
            response = self.client(**completion_params)

            if response is None:
                raise ValueError(
                    f"LiteLLM returned None for model {completion_params.get('model')}. "
                    "This usually means the vLLM server returned a 400/500 error. "
                    "Check the vLLM server log for details."
                )

            # Log actual cost from API usage when available
            if not self.is_local_model and hasattr(response, 'usage') and response.usage:
                u = response.usage
                pricing = self._get_model_pricing()
                actual_input = getattr(u, 'prompt_tokens', 0) or 0
                actual_output = getattr(u, 'completion_tokens', 0) or 0
                actual_cost = self._estimate_cost(actual_input, actual_output, 0)
                print(f"   Actual usage: {actual_input:,} in / {actual_output:,} out = ${actual_cost:.6f}")

            # Extract text from response
            if hasattr(response, 'choices') and response.choices:
                response_content = response.choices[0].message.content

                # Strip any residual <think>...</think> blocks from content
                if response_content and '<think>' in response_content:
                    response_content = re.sub(
                        r'<think>.*?</think>\s*', '', response_content, flags=re.DOTALL
                    ).strip()

                # Save successful query to cache
                import datetime
                query_id = f"success_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._save_query_to_cache(
                    query_id=query_id,
                    model_id=self.model_id,
                    prompt=prompt,
                    response=response_content,
                    query_type="generation"
                )
                
                return response_content
            else:
                return str(response)
                
        except (FileNotFoundError, ValueError) as e:
            # Media processing errors - don't fall back to text-only
            logger.error(f"Media processing failed: {e}")
            raise e
        except Exception as e:
            # Other errors (API, network, etc.)
            logger.error(f"Error generating response: {e}")
            raise e
    
    def generate_batch(self, 
                      prompts: List[str], 
                      media_paths: Optional[List[str]] = None,
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """Generate responses for a batch of prompts with image or video paths"""
        if media_paths is None:
            media_paths = [None] * len(prompts)
        
        if len(prompts) != len(media_paths):
            raise ValueError("Number of prompts and media paths must match")
        
        responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_media_paths = media_paths[i:i + batch_size]
            
            batch_responses = []
            for prompt, media_path in zip(batch_prompts, batch_media_paths):
                response = self.generate_single(prompt, media_path, **kwargs)
                batch_responses.append(response)
            
            responses.extend(batch_responses)
        
        return responses
    
    
    def generate_structured(self,
                          prompt: str,
                          response_model: Any,
                          media_path: Optional[str] = None,
                          system_prompt: Optional[str] = None,
                          **kwargs) -> Any:
        """Generate structured response using LiteLLM's native response_format support"""
        # Add JSON example to the prompt so models know the expected structure
        if response_model:
            example = {name: f"<{field.description or name}>" for name, field in response_model.model_fields.items()}
            json_prompt = f"{prompt}\n\nRespond in JSON format, e.g. {json.dumps(example)}."
        else:
            json_prompt = f"{prompt}\n\nPlease respond in JSON format."

        messages = self._prepare_messages(json_prompt, media_path, system_prompt=system_prompt)

        # Estimate tokens and images for cost calculation
        estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        estimated_output_tokens = kwargs.get("max_tokens", self.max_tokens)
        num_images = 0

        if media_path and os.path.exists(media_path):
            if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                num_images = 1
            elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                num_images = 1  # Now using video_url format

        # Print cost estimate
        self._print_cost_estimate(
            int(estimated_input_tokens),
            int(estimated_output_tokens),
            num_images,
            "structured generation"
        )

        # Prepare completion parameters
        # For local vLLM servers via HTTP, use openai/ prefix to use HTTP API
        # (not vllm/ which spawns new process - see FRAMEWORK_SUMMARY.md)
        api_model_id = self.model_id
        if self.is_local_model and self.model_id.startswith("vllm/"):
            # Use openai/ prefix + actual model name for HTTP API
            actual_model = self.model_id[5:]  # Remove "vllm/" prefix
            api_model_id = f"openai/{actual_model}"

        completion_params = {
            "model": api_model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": 600,
            "num_retries": 0
        }

        # For structured outputs with response_model
        if response_model:
            if self.is_local_model:
                # vLLM: use json_object to avoid guided-decoding bugs
                completion_params["response_format"] = {"type": "json_object"}
            else:
                # API models: use json_schema with strict mode. OpenAI's
                # strict mode requires additionalProperties: false on every
                # object in the schema, including nested $defs (e.g.
                # ClaimRecallEvaluation's nested ClaimMatch), not just the root.
                schema = response_model.model_json_schema()
                schema["additionalProperties"] = False
                for nested_def in schema.get("$defs", {}).values():
                    if nested_def.get("type") == "object":
                        nested_def["additionalProperties"] = False
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True
                    }
                }

        # Only set api_base and api_key for local models
        if self.is_local_model:
            completion_params["api_base"] = self.api_base or f"http://localhost:{self.port}/v1"
            completion_params["api_key"] = self.api_key or "EMPTY"
        else:
            # For API models, let LiteLLM handle the routing
            if self.api_key:
                completion_params["api_key"] = self.api_key

        # Add per-request mm_processor_kwargs for local vLLM only (must match generate_single).
        # fps and max_pixels are Qwen-VL specific; other models (InternVL, llava, MiniCPM,
        # Molmo2) don't support them and InternVL returns a 400 if fps is sent.
        if self.is_local_model:
            actual_model = self.model_id.replace("vllm/", "")
            model_cfg = self._get_model_vision_config(actual_model)
            mm_kwargs = {}
            if model_cfg.get("use_max_pixels", False):
                max_res = model_cfg.get("max_video_resolution", 512)
                mm_kwargs["max_pixels"] = max_res * max_res
                if self._current_video_fps is not None:
                    mm_kwargs["fps"] = self._current_video_fps
            extra_body = completion_params.get("extra_body", {})
            if mm_kwargs:
                extra_body["mm_processor_kwargs"] = mm_kwargs
            # Only set guided decoding backend for json_schema mode (not json_object).
            # json_object mode doesn't need guided decoding and outlines leaks state,
            # crashing vLLM's EngineCore after ~40-80 requests.
            rf = completion_params.get("response_format", {})
            if rf.get("type") == "json_schema":
                extra_body["guided_decoding_backend"] = "outlines"
            if extra_body:
                completion_params["extra_body"] = extra_body

        # Generate structured response using LiteLLM's native support
        response = self.client(**completion_params)

        if response is None:
            raise ValueError(
                f"LiteLLM returned None for model {completion_params.get('model')}. "
                "This usually means the vLLM server returned a 400/500 error "
                "(e.g. chat template doesn't support system messages). "
                "Check the vLLM server log for details."
            )

        # Log actual cost from API usage when available
        if not self.is_local_model and hasattr(response, 'usage') and response.usage:
            u = response.usage
            actual_input = getattr(u, 'prompt_tokens', 0) or 0
            actual_output = getattr(u, 'completion_tokens', 0) or 0
            actual_cost = self._estimate_cost(actual_input, actual_output, 0)
            print(f"   Actual usage: {actual_input:,} in / {actual_output:,} out = ${actual_cost:.6f}")

        # Extract and parse the structured response
        if hasattr(response, 'choices') and response.choices:
            msg = response.choices[0].message
            content = msg.content

            # Capture reasoning_content from thinking models (e.g. Qwen3.5).
            # vLLM's reasoning parser separates <think> blocks into this field.
            model_reasoning = getattr(msg, 'reasoning_content', None)

            # Strip any residual <think>...</think> blocks from content
            if content and '<think>' in content:
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()

            # Handle empty or None content
            if not content:
                raise ValueError(f"Model returned empty response. Raw response: {response}")

            # Parse JSON string
            if isinstance(content, str):
                try:
                    json_data = json.loads(content)
                except json.JSONDecodeError as e:
                    # Handle truncated JSON (model hit max_tokens mid-string, usually inside "reasoning")
                    # Try to rescue the "answer" field which typically appears before "reasoning"
                    if "Unterminated string" in str(e):
                        m = re.search(r'"answer"\s*:\s*"([^"]*)"', content)
                        if m:
                            answer_val = m.group(1)
                            logger.warning(
                                f"JSON truncated at max_tokens; rescued answer='{answer_val[:80]}' "
                                f"from partial content ({len(content)} chars)"
                            )
                            json_data = {"answer": answer_val, "reasoning": "[truncated]"}
                        else:
                            raise ValueError(
                                f"Failed to parse model response as JSON: {e}. Content: {content[:500]}"
                            )
                    else:
                        raise ValueError(
                            f"Failed to parse model response as JSON: {e}. Content: {content[:500]}"
                        )
            else:
                json_data = content

            # Use thinking model's reasoning_content as the reasoning field
            # if the model didn't already provide one in the JSON.
            if model_reasoning and isinstance(json_data, dict):
                if not json_data.get("reasoning"):
                    json_data["reasoning"] = model_reasoning

            # Parse into Pydantic model if provided. Finetuned models sometimes
            # emit valid JSON but with alternate top-level keys (e.g.
            # {"main_interactions": "..."} or {"person red": "..."}) under the
            # soft `json_object` constraint we use for local vLLM. Catch the
            # validation error and try to coerce.
            if response_model:
                try:
                    return response_model(**json_data)
                except ValidationError:
                    if (response_model.__name__ == "OpenQuestionResponse"
                            and isinstance(json_data, dict)):
                        # Strategy 1: well-known alias keys carry the answer.
                        ALT_ANSWER_KEYS = (
                            "main_interactions", "main_answer", "interactions",
                            "main_response", "response", "description",
                            "output", "result", "text",
                        )
                        for k in ALT_ANSWER_KEYS:
                            v = json_data.get(k)
                            if isinstance(v, list):
                                # Models sometimes emit list-of-strings (one
                                # interaction per item). Join into a sentence.
                                v = " ".join(str(x) for x in v if x).strip()
                            if isinstance(v, str) and v.strip():
                                logger.warning(
                                    f"OpenQuestionResponse missing 'answer'; "
                                    f"using alias key '{k}'"
                                )
                                return response_model(
                                    answer=v,
                                    reasoning=json_data.get("reasoning"),
                                )
                        # Strategy 2: concat non-reasoning string values
                        # (handles "person red"/"person green" descriptive dumps).
                        non_reasoning = {
                            k: v for k, v in json_data.items()
                            if isinstance(v, str) and v.strip() and k != "reasoning"
                        }
                        if non_reasoning:
                            combined = " ".join(non_reasoning.values())
                            logger.warning(
                                f"OpenQuestionResponse missing 'answer'; concat "
                                f"{len(non_reasoning)} value(s) from "
                                f"{list(non_reasoning.keys())[:3]}"
                            )
                            return response_model(
                                answer=combined,
                                reasoning=json_data.get("reasoning"),
                            )
                    raise
            return json_data
        else:
            raise ValueError(f"No choices in response. Raw response: {response}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration"""
        return {
            "model_id": self.model_id,
            "type": "LiteLLMWrapper",
            "is_local_model": self.is_local_model,
            "use_structured_output": self.use_structured_output,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "config": self.config,
            "cost_summary": self.get_cost_summary()
        }
    
    @classmethod
    def print_status(cls):
        """Print current status of all models and server"""
        print("=== LiteLLM Wrapper Status ===")
        print(f"Active models: {cls.get_active_models()}")
        
        server_status = cls.get_server_status()
        print(f"Server running: {server_status['running']}")
        if server_status['running']:
            print(f"Server model: {server_status['model']}")
            if server_status['health']:
                health = server_status['health']
                print(f"Server healthy: {health['healthy']}")
                if health['healthy']:
                    print(f"Available models: {health['models']}")
                else:
                    print(f"Health error: {health['error']}")
        print("=" * 30)
