# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preset transforms for the Shopify product catalogue dataset (OpenAI multimodal)."""

import json
from typing import Any

from inference_endpoint.dataset_manager.transforms import RowProcessor, Transform


def _product_metadata_schema() -> dict[str, Any]:
    """JSON schema for ProductMetadata output (category, brand, is_secondhand).

    Matches ProductMetadata.model_json_schema() from mlperf_inf_mm_q3vl.schema at https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/schema.py#L822.
    """
    return {
        "type": "object",
        "title": "ProductMetadata",
        "description": "Json format for the expected responses from the VLM.",
        "properties": {
            "category": {
                "type": "string",
                "title": "Category",
                "description": (
                    "The complete category of the product, e.g.,\n"
                    '"Clothing & Accessories > Clothing > Shirts > Polo Shirts".\n'
                    'Each categorical level is separated by " > ".'
                ),
            },
            "brand": {
                "type": "string",
                "title": "Brand",
                "description": 'The brand of the product, e.g., "giorgio armani".',
            },
            "is_secondhand": {
                "type": "boolean",
                "title": "Is Secondhand",
                "description": "True if the product is second-hand, False otherwise.",
            },
        },
        "required": ["category", "brand", "is_secondhand"],
        "additionalProperties": False,
    }


class ShopifyMultimodalFormatter(RowProcessor):
    """Transform Shopify product rows to OpenAI multimodal format (prompt + system).

    Outputs system (str) and prompt (list of content parts: text + image_url)
    compatible with OpenAIMsgspecAdapter.
    """

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert product row to prompt/system for OpenAI vision API."""
        product_title = row.get("product_title", "")
        product_description = row.get("product_description", "")
        categories_str = row.get("potential_product_categories", "[]")
        image_base64 = row.get("product_image_base64", "")
        image_format = row.get("product_image_format", "png")

        schema_str = json.dumps(_product_metadata_schema(), indent=2)

        system = f"""Please analyze the product from the user prompt
and provide the following fields in a valid JSON object:
- category
- brand
- is_secondhand

You must choose only one, which is the most appropriate, correct, and specific
category out of the list of possible product categories.

The description of the product sometimes contains various types of source code
(e.g., JavaScript, CSS, HTML, etc.), where useful product information is embedded
somewhere inside the source code. For this task, you should extract the useful
product information from the source code and leverage it, and discard the
programmatic parts of the source code.

Your response should only contain a valid JSON object and nothing more, e.g.,
you should not fence the JSON object inside a ```json code block.
The JSON object should match the following JSON schema:
```json
{schema_str}
```
"""

        # Build multimodal prompt: text part + optional image part
        text_content = f"""The title of the product is the following:
```text
{product_title}
```

The description of the product is the following:
```text
{product_description}
```

The following are the possible product categories:
```json
{categories_str}
```
"""

        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": text_content},
        ]

        if image_base64:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}",
                    },
                }
            )

        row["system"] = system
        row["prompt"] = content_parts
        return row


def q3vl() -> list[Transform]:
    """Preset for Qwen3-VL / OpenAI multimodal adapter (vLLM vision, etc.)."""
    return [
        ShopifyMultimodalFormatter(),
    ]
