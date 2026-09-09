import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def test():
    load_dotenv()
    print("Testing basic ChatNVIDIA call with DeepSeek...", flush=True)

    api_key = os.getenv("NVIDIA_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-flash-0731")

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=50,
    )

    try:
        print("Sending message...", flush=True)
        response = llm.invoke([HumanMessage(content="Reply with just the word YES.")])
        print("Response received:", response.content, flush=True)
    except Exception as e:
        print("Call failed:", e, flush=True)


if __name__ == "__main__":
    test()
