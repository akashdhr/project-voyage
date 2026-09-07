import os
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Returns a configured LangChain ChatOpenAI instance pointing to NVIDIA's NIM API.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set.")
        
    # Disabling thinking for speed tests
    model_kwargs = {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    }
    
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0, 
        max_tokens=8192,
        model_kwargs=model_kwargs
    )
    return llm
