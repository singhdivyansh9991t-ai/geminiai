import os
import platform
from importlib.metadata import version
from pathlib import Path

try:
    import httpx
    from dotenv import load_dotenv
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    print("Required packages are missing. Install them with: pip install -r requirements.txt")
    raise SystemExit(1)

MODEL = "gemini-3.8-flash"


def print_api_error(error, api_key):
    status_code = getattr(error, "code", getattr(error, "status_code", "unknown"))
    message = getattr(error, "message", None) or "(No API error message was returned.)"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")

    print(f"Gemini API error: {status_code}")
    print(f"Message: {message}")
    print(f"Exception type: {type(error).__name__}")
    print(f"Model: {MODEL}")
    print(f"google-genai: {version('google-genai')}")
    print(f"Python: {platform.python_version()}")


def main():
    env_file = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_file)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to the .env file in this project folder.")
        return

    try:
        client = genai.Client(api_key=api_key)
        response = client.interactions.create(
            model=MODEL,
            input="Explain artificial intelligence in one simple paragraph.",
        )
    except genai_errors.APIError as error:
        print_api_error(error, api_key)
        return
    except (httpx.TimeoutException, httpx.NetworkError):
        print("Could not reach the Gemini API. Check your internet connection and try again.")
        return
    except Exception as error:
        if hasattr(error, "code") or hasattr(error, "status_code") or hasattr(error, "message"):
            print_api_error(error, api_key)
            return
        print(f"Unexpected error while calling Gemini ({type(error).__name__}).")
        return

    if response.output_text:
        print(response.output_text)
    else:
        print("Gemini returned no text for this request.")


if __name__ == "__main__":
    main()