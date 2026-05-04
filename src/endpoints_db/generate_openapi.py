"""Generate openapi.json and API.md from the FastAPI app schema."""
import json
from app.main import app
from openapi_markdown.generator import to_markdown

schema = app.openapi()
with open("openapi.json", "w") as f:
    json.dump(schema, f, indent=2)
print("openapi.json written")

to_markdown("openapi.json", "API.md")
print("API.md written")
