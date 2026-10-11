"""Standalone, offline Llama inference; does not depend on PoisonedRAG modules.

Usage:
    python local_llama.py --config local_llama_config.json --prompt "Hello"
    model = LocalLlama("local_llama_config.json")
    answer = model.query(prompt)

Inputs are raw strings, matching the public PoisonedRAG Llama interface. No chat
template or additional system prompt is inserted. Only generated tokens are
decoded, avoiding the original implementation's character-slicing ambiguity.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed


class LocalLlama:
    def __init__(self, config):
        """Accept a config filename or an already parsed config dictionary."""
        if isinstance(config, (str, Path)):
            config_file = Path(config).expanduser().resolve()
            config = json.loads(config_file.read_text(encoding="utf-8"))
            base_dir = config_file.parent
        else:
            base_dir = Path.cwd()
        config = copy.deepcopy(config)
        info = config["model_info"]
        params = config["params"]
        path = Path(info["local_path"]).expanduser()
        self.model_path = (path if path.is_absolute() else base_dir / path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {self.model_path}")
        self.name = info.get("name", self.model_path.name)
        self.device = torch.device(params.get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the server's PyTorch/CUDA installation.")
        dtype_name = params.get("dtype", "float16")
        if dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        self.seed = int(params.get("seed", 100))
        max_output_tokens = int(params.get("max_output_tokens", 150))
        temperature = float(params.get("temperature", 0.1))
        do_sample = params.get("do_sample")
        if do_sample is not None and type(do_sample) is not bool:
            raise ValueError("do_sample must be true, false, or null (inherit checkpoint)")
        if max_output_tokens <= 0 or temperature < 0:
            raise ValueError("max_output_tokens must be positive; temperature cannot be negative")
        # Seed once, not once per query, so a run follows a reproducible RNG sequence.
        set_seed(self.seed)
        self._lock = threading.Lock()
        self.tokenizer = LlamaTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True)
        load_options = {"local_files_only": True, "torch_dtype": getattr(torch, dtype_name)}
        if params.get("attn_implementation"):
            load_options["attn_implementation"] = params["attn_implementation"]
        self.model = LlamaForCausalLM.from_pretrained(str(self.model_path), **load_options)
        self.model.to(self.device)
        self.model.eval()
        # The public interface does not explicitly set do_sample. Preserve the
        # checkpoint's behavior unless the user explicitly chooses true/false.
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.max_new_tokens = max_output_tokens
        self.generation_config.temperature = temperature
        if do_sample is not None:
            self.generation_config.do_sample = do_sample
        if self.generation_config.do_sample and temperature <= 0:
            raise ValueError("Sampling requires a positive temperature")
        self.context_limit = int(self.model.config.max_position_embeddings)

    def query(self, prompt):
        """Return only the generated answer, with no automatic prompt wrapping."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            input_length = input_ids.shape[-1]
            if input_length + self.generation_config.max_new_tokens > self.context_limit:
                raise ValueError(
                    f"Input ({input_length}) + output budget "
                    f"({self.generation_config.max_new_tokens}) exceeds context limit "
                    f"({self.context_limit}); input was not truncated.")
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                generation_config=self.generation_config)
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            return self.tokenizer.decode(sequences[0, input_length:], skip_special_tokens=True)

    def metadata(self):
        """Expose effective settings for recording an experiment's configuration."""
        return {
            "model": self.name,
            "local_path": str(self.model_path),
            "device": str(self.device),
            "dtype": str(self.model.dtype),
            "seed": self.seed,
            "prompt_format": "raw",
            "output_extraction": "generated_tokens_only",
            "context_limit": self.context_limit,
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "generation_config": self.generation_config.to_dict(),
        }


def main():
    parser = argparse.ArgumentParser(description="Standalone offline Llama-2 inference")
    parser.add_argument("--config", default=str(Path(__file__).with_name("local_llama_config.json")))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Raw prompt text")
    source.add_argument("--prompt_file", help="UTF-8 file containing the raw prompt")
    parser.add_argument("--show_config", action="store_true", help="Print effective settings to stderr")
    args = parser.parse_args()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model = LocalLlama(args.config)
    if args.show_config:
        print(json.dumps(model.metadata(), ensure_ascii=False, indent=2), file=sys.stderr)
    print(model.query(prompt))


if __name__ == "__main__":
    main()
