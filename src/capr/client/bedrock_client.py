import sys
import time

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, ".")   # allow import from project root
from config import API_PAUSE, AWS_REGION, MAX_RETRIES


class BedrockClient:
    def __init__(self, region: str = AWS_REGION):
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def converse(
        self,
        model_id:      str,
        messages:      list[dict],
        system_prompt: str   = "",
        temperature:   float = 0.0,
        max_tokens:    int   = 512,
    ) -> tuple[str, int, float]:

        payload: dict = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            t_start = time.time()
            try:
                response = self._client.converse(**payload)
                latency  = time.time() - t_start

                text   = response["output"]["message"]["content"][0]["text"]
                usage  = response.get("usage", {})
                tokens = usage.get("inputTokens", 0) + usage.get("outputTokens", 0)

                return text, tokens, latency

            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in ("ThrottlingException", "ServiceUnavailableException"):
                    last_error = exc
                    wait = API_PAUSE * (2 ** attempt)
                    print(f"      [Bedrock] throttled – retrying in {wait:.1f}s "
                          f"(attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Bedrock error [{code}]: {exc}") from exc

        raise RuntimeError(
            f"Bedrock call failed after {MAX_RETRIES} retries: {last_error}"
        )
