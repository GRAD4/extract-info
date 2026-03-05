"""
Thin LLM client that supports two providers:

  - ollama  (local, free, default provider)
  - bedrock (AWS, requires credentials)

Set LLM_PROVIDER in your .env to switch between them.
"""
import json
import os

import requests


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        aws_region: str | None = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")

        if self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        elif self.provider == "bedrock":
            import boto3

            self.model = model or os.getenv(
                "BEDROCK_MODEL_ID",
                "anthropic.claude-3-haiku-20240307-v1:0",
            )
            region = aws_region or os.getenv("AWS_REGION", "us-east-1")
            self.client = boto3.client("bedrock-runtime", region_name=region)

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. Choose 'ollama' or 'bedrock'."
            )

    def generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama(prompt)
        return self._bedrock(prompt)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ollama(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ── AWS Bedrock ───────────────────────────────────────────────────────────

    def _bedrock(self, prompt: str) -> str:
        is_claude = "anthropic" in self.model
        if is_claude:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
        else:
            # Llama / other models
            payload = {"prompt": prompt, "temperature": 0.2, "max_gen_len": 512}

        resp = self.client.invoke_model(
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(resp["body"].read())

        if is_claude:
            return body.get("content", [{}])[0].get("text", "")
        return body.get("generation", body.get("output", ""))
