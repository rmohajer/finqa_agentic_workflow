#!/usr/bin/env python
"""
Test script comparing LLM providers:
- NVIDIA Nemotron 3 Ultra
- Groq (openai/gpt-oss-20b)

Measures response time on a complex financial reasoning prompt.
Model names are configurable at the top of this file.
"""
import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq

load_dotenv()

# ============================================================
# CONFIGURATION - Model names and parameters
# ============================================================
NVIDIA_MODEL =  "nvidia/nemotron-3.5-bolt-e4b" # "nvidia/nemotron-3-ultra-550b-a55b"
GROQ_MODEL = "openai/gpt-oss-20b"

# Common LLM parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# NVIDIA-specific parameters
NVIDIA_CHAT_TEMPLATE_KWARGS = {'reasoning_effort': None}

# Complex financial reasoning prompt (similar to FinQA tasks)
COMPLEX_PROMPT = """You are a financial analyst. Given the following financial data, answer the question.

Financial Report Excerpt:
- Revenue Q1 2024: $12.5M (up 15% YoY)
- Revenue Q2 2024: $14.2M (up 18% YoY) 
- Revenue Q3 2024: $13.8M (up 12% YoY)
- Revenue Q4 2024: $16.1M (up 22% YoY)
- Gross Margin: 68% (consistent across quarters)
- Operating Expenses Q1: $4.2M, Q2: $4.5M, Q3: $4.3M, Q4: $4.8M
- Net Income Q1: $1.8M, Q2: $2.3M, Q3: $2.1M, Q4: $3.1M
- Total Assets: $45.2M
- Total Liabilities: $18.7M
- Shareholders' Equity: $26.5M
- Cash and Equivalents: $8.4M
- Long-term Debt: $12.3M

Question: Calculate the full-year 2024 revenue growth rate YoY, the average quarterly operating margin, and the debt-to-equity ratio. Then explain whether the company's financial health is improving or deteriorating based on these metrics. Provide step-by-step reasoning."""


def time_llm_call(llm, prompt, provider_name):
    """Time a synchronous LLM call"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} (sync)")
    print(f"{'='*60}")
    
    start = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        elapsed = time.perf_counter() - start
        print(f"✅ Completed in {elapsed:.2f}s")
        print(f"Response length: {len(response.content)} chars")
        print(f"First 200 chars: {response.content[:200]}...")
        return elapsed, response.content
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"❌ Failed after {elapsed:.2f}s: {e}")
        return elapsed, None


async def time_llm_call_async(llm, prompt, provider_name):
    """Time an asynchronous LLM call"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} (async)")
    print(f"{'='*60}")
    
    start = time.perf_counter()
    try:
        response = await llm.ainvoke(prompt)
        elapsed = time.perf_counter() - start
        print(f"✅ Completed in {elapsed:.2f}s")
        print(f"Response length: {len(response.content)} chars")
        print(f"First 200 chars: {response.content[:200]}...")
        return elapsed, response.content
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"❌ Failed after {elapsed:.2f}s: {e}")
        return elapsed, None


def create_nvidia_llm(api_key, model_name=None, temperature=None, max_tokens=None, chat_template_kwargs=None):
    """Create NVIDIA ChatNVIDIA instance with configurable parameters"""
    return ChatNVIDIA(
        model=model_name or NVIDIA_MODEL,
        api_key=api_key,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        chat_template_kwargs=chat_template_kwargs or NVIDIA_CHAT_TEMPLATE_KWARGS,
    )


def create_groq_llm(api_key, model_name=None, temperature=None, max_tokens=None):
    """Create Groq ChatGroq instance with configurable parameters"""
    return ChatGroq(
        model=model_name or GROQ_MODEL,
        api_key=api_key,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
    )


def test_nvidia_sync(model_name=None, temperature=None, max_tokens=None):
    """Test NVIDIA model synchronous"""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("⚠️  NVIDIA_API_KEY not set, skipping NVIDIA sync test")
        return None
        
    llm = create_nvidia_llm(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return time_llm_call(llm, COMPLEX_PROMPT, f"NVIDIA {model_name or NVIDIA_MODEL}")


def test_groq_sync(model_name=None, temperature=None, max_tokens=None):
    """Test Groq model synchronous"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  GROQ_API_KEY not set, skipping Groq sync test")
        return None
    
    llm = create_groq_llm(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return time_llm_call(llm, COMPLEX_PROMPT, f"Groq {model_name or GROQ_MODEL}")


async def test_nvidia_async(model_name=None, temperature=None, max_tokens=None):
    """Test NVIDIA model asynchronous"""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("⚠️  NVIDIA_API_KEY not set, skipping NVIDIA async test")
        return None
    
    llm = create_nvidia_llm(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await time_llm_call_async(llm, COMPLEX_PROMPT, f"NVIDIA {model_name or NVIDIA_MODEL}")


async def test_groq_async(model_name=None, temperature=None, max_tokens=None):
    """Test Groq model asynchronous"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  GROQ_API_KEY not set, skipping Groq async test")
        return None
    
    llm = create_groq_llm(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await time_llm_call_async(llm, COMPLEX_PROMPT, f"Groq {model_name or GROQ_MODEL}")


def print_summary(results):
    """Print comparison summary"""
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    sync_results = {k: v for k, v in results.items() if not k.endswith('_async')}
    async_results = {k: v for k, v in results.items() if k.endswith('_async')}
    
    if sync_results:
        print("\n🔄 Synchronous:")
        for name, (elapsed, _) in sync_results.items():
            if elapsed:
                print(f"  {name}: {elapsed:.2f}s")
    
    if async_results:
        print("\n⚡ Asynchronous:")
        for name, (elapsed, _) in async_results.items():
            if elapsed:
                print(f"  {name}: {elapsed:.2f}s")
    
    # Find fastest
    all_results = {k: v[0] for k, v in results.items() if v and v[0]}
    if all_results:
        fastest = min(all_results, key=all_results.get)
        print(f"\n🏆 Fastest: {fastest} ({all_results[fastest]:.2f}s)")


def main():
    print("\n🔬 LLM Provider Benchmark: Complex Financial Reasoning")
    print("Prompt: Multi-metric calculation + analysis (FinQA-style)")
    print("Set NVIDIA_API_KEY and/or GROQ_API_KEY in .env\n")
    print(f"NVIDIA Model: {NVIDIA_MODEL}")
    print(f"Groq Model:   {GROQ_MODEL}")
    
    results = {}
    
    # Synchronous tests
    print("\n" + "🔄 RUNNING SYNCHRONOUS TESTS")
    nvidia_sync = test_nvidia_sync()
    if nvidia_sync:
        results[f'NVIDIA {NVIDIA_MODEL}'] = nvidia_sync
    
    groq_sync = test_groq_sync()
    if groq_sync:
        results[f'Groq {GROQ_MODEL}'] = groq_sync
    
    # Asynchronous tests
    print("\n" + "⚡ RUNNING ASYNCHRONOUS TESTS")
    nvidia_async = asyncio.run(test_nvidia_async())
    if nvidia_async:
        results[f'NVIDIA {NVIDIA_MODEL}_async'] = nvidia_async
    
    groq_async = asyncio.run(test_groq_async())
    if groq_async:
        results[f'Groq {GROQ_MODEL}_async'] = groq_async
    
    # Summary
    print_summary(results)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()