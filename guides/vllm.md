# vLLM

vLLM allows users to locally host open-source, for more info about what models are supported see [this](https://docs.vllm.ai/en/stable/models/supported_models.html). Once an vLLM instance is locally running in your setup (default `http://localhost:8000`), you can use the `aisuite` API for chat completions as shown below. By default, no API Key is needed for these locally hosted models.

## Create a Chat Completion

Sample code:
```python
import aisuite as ai

client = ai.Client(
    provider_configs={
        "vllm": {
            "api_url": "http://localhost:8000",
            "timeout": 300,
        }
    }
)

messages = [
    {
        "role": "system", 
        "content": "You are a calculus expert."
    },
    {
        "role": "user", 
        "content": "integrate x^2 dx from 0 to 1"
    },
]

model = "vllm:Qwen/Qwen2.5-0.5B-Instruct"

response = client.chat.completions.create(
    model=model, 
    messages=messages, 
    temperature=0.75,
)

print(response.choices[0].message.content)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
