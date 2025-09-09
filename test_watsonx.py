from backend.query_llm import call_watsonx
import os

res = call_watsonx(
    "Test Watsonx integration!",
    api_key=os.getenv("WATSONX_KEY"),
    url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECT")
)
print("Watsonx Response:", res)
