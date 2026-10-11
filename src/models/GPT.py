import os

from openai import OpenAI

from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        params = config.get("params", {})
        self.request_seed = int(params["seed"])
        self.max_output_tokens = int(params["max_output_tokens"])
        self.top_p = params.get("top_p")
        self.frequency_penalty = params.get("frequency_penalty")
        self.presence_penalty = params.get("presence_penalty")
        self.response_format = params.get("response_format")
        api_key_env = config.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {api_key_env!r} is required by the model config")
        else:
            key_info = config.get("api_key_info", {})
            api_keys = key_info.get("api_keys", [])
            api_pos = int(key_info.get("api_key_use", -1))
            if not 0 <= api_pos < len(api_keys):
                raise ValueError("Model config must provide api_key_env or a valid api_key_info entry")
            api_key = api_keys[api_pos]

        options = {"api_key": api_key}
        if config.get("base_url"):
            options["base_url"] = config["base_url"]
        if params.get("timeout") is not None:
            options["timeout"] = float(params["timeout"])
        if params.get("max_retries") is not None:
            options["max_retries"] = int(params["max_retries"])
        self.client = OpenAI(**options)

    def query_with_metadata(self, msg, system_prompt=None):
        request = {
            "model": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "n": 1,
            "seed": self.request_seed,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ],
        }
        for key, value in (
            ("top_p", self.top_p),
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
            ("response_format", self.response_format),
        ):
            if value is not None:
                request[key] = value
        completion = self.client.chat.completions.create(**request)
        choice = completion.choices[0]
        message = choice.message
        usage = getattr(completion, "usage", None)
        if usage is not None:
            usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
        return {
            "text": message.content,
            "requested_model": self.name,
            "response_model": getattr(completion, "model", None),
            "finish_reason": choice.finish_reason,
            "usage": usage,
            "request_id": getattr(completion, "id", None),
            "system_fingerprint": getattr(completion, "system_fingerprint", None),
            "refusal": getattr(message, "refusal", None),
        }

    def query(self, msg):
        try:
            return self.query_with_metadata(msg)["text"] or ""
        except Exception as exc:
            print(exc)
            return ""
