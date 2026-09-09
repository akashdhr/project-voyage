import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class TestSchema(BaseModel):
    airline: str
    price: float


def test():
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

    model_kwargs = {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 1024,
        }
    }

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
        model_kwargs=model_kwargs,
    )

    print("Testing basic generation...")
    try:
        res = llm.invoke([HumanMessage(content="Hello!")])
        print("Basic output:", res.content)
    except Exception as e:
        print("Basic failed:", e)

    print("Testing structured output...")
    try:
        structured_llm = llm.with_structured_output(TestSchema)
        res = structured_llm.invoke(
            [HumanMessage(content="The British Airways flight costs 500.")]
        )
        print("Structured output:", res)
    except Exception as e:
        print("Structured failed:", e)


if __name__ == "__main__":
    test()
