# OpenAI API

The OpenAI OpenAPI specs are available at https://github.com/openai/openai-openapi which describes the
REST API used to communicate between clients and servers.

We include the OpenAI OpenAPI specs as well as the generated pydantic definitions as part of the source.
The SHA used for development and testing is 498c71ddf6f1c45b983f972ccabca795da211a3e and the complete URL
for the specification is
https://raw.githubusercontent.com/openai/openai-openapi/498c71ddf6f1c45b983f972ccabca795da211a3e/openapi.yaml

The pydantic definitions can be generated from the yaml specifications (as described below, with the SHA).
Since it takes about 10 seconds to generate them, and do not change unless the underlying definitions
in the yaml file change, we include the generated definitions as well.

```

wget https://raw.githubusercontent.com/openai/openai-openapi/498c71ddf6f1c45b983f972ccabca795da211a3e/openapi.yaml
uv pip install datamodel-code-generator
uv run datamodel-codegen --input-file-type openapi --input openapi.yaml --output openai_types_gen.py --output-model-type pydantic_v2.BaseModel
sed -i 's/min_items/min_length/g' openai_types_gen.py
```

The above commands will generate `openai_types_gen.py` which contains the python definitions for the
request and response structures.
