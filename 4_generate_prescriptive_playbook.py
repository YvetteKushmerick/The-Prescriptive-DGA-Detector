# Filename: 4_generate_prescriptive_playbook.py
import json
import asyncio
import aiohttp
import os
import sys


API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

async def generate_playbook(xai_findings, api_key):
    """
    Creates a simple, step-by-step incident response playbook using the Gemini API.
    """
    prompt = f"""
As a SOC Manager, create a simple, step-by-step incident response playbook for a Tier 1 analyst.
Base it only on the alert details and the model explanation. Output a numbered list of 3–4 concise steps.

**Alert Details & AI Explanation:**
{xai_findings}
"""

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    timeout = aiohttp.ClientTimeout(total=30)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(API_URL, params=params, json=payload, headers=headers) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    return f"Error: Non-JSON response (status {resp.status}): {text[:300]}"

                if resp.status != 200:
                    return f"Error {resp.status}: {json.dumps(data)}"

                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    return "Error: Unexpected response shape: " + json.dumps(data)

    except aiohttp.ClientConnectorError as e:
        return f"An error occurred: Could not connect to the API endpoint. {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# Data representing an alert from a DGA detection model
findings = """- **Alert:** Potential DGA domain detected in DNS logs.
- **Domain:** `kq3v9z7j1x5f8g2h.info`
- **Source IP:** `10.1.1.50` (Workstation-1337)
- **AI Model Explanation (from SHAP):** 99.8% confidence due to very high entropy and long length.
"""

async def main():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("---")
        print("🚨 Error: GOOGLE_API_KEY environment variable not set.")
        print("Linux/macOS:  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
        print('Windows PS:   $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"')
        print("---")
        sys.exit(1)

    # Optional: show masked tail of key for sanity
    print(f"Using GOOGLE_API_KEY ending with ...{api_key[-6:]}")

    print("---")
    print("Context: Generating a prescriptive playbook from alert findings.")
    print("Input being sent to Gemini:")
    print(findings)
    print("--------------------------------------------------")
    print("\n--- AI-Generated Playbook ---")

    playbook = await generate_playbook(findings, api_key)
    print(playbook)

def smart_async_run(coro):
    # Works in plain scripts and in environments with a running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

if __name__ == "__main__":
    smart_async_run(main())


