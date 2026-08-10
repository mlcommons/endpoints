# Token accounting — reference-implementation spec

How ISL, OSL, and TPOT are defined, where they come from, and what every new model or
workload must declare before it can report them.

## 1. Governing principle

**The benchmark defines its own token numbers.**

ISL and OSL are _tokens under the run's pinned reference tokenizer and pinned chat template,
applied to the request the client sent and the response the client received_ — not "what the
server says it processed." No integer asserted by the endpoint is ever recorded as a metric.

This is a deliberate trade. It costs absolute fidelity to the SUT's internal count; it buys
reproducibility across SUTs and removes the incentive to inflate a self-reported throughput
numerator. Divergence between the client number and the server's `usage.*` is a **measured,
published quantity** (§7), not an error to be chased away.

Two consequences follow, and both are load-bearing:

- Where the client sends token IDs, ISL is exact by construction and no server flag can perturb
  it.
- Where the client sends chat messages, ISL is a client-side reconstruction whose fidelity is
  bounded by how precisely the template, its kwargs, and the tokenizer are pinned. That
  reconstruction is defined by rule **I-2** below and is identical for every model — what varies
  is only the pinned data.

## 2. The chain

```
 CLIENT                                                    │  WIRE  │            SERVER
─────────────────────────────────────────────────────────  │        │  ─────────────────────────
 ① dataset file  (parquet / jsonl / HF)
      │  Dataset.load()  +  predefined preset transforms
      ▼
 ② row dict in memory   { prompt | messages | input_tokens , tools , system , … }
      │  adapter.dataset_transforms()      ←── Harmonize() lives HERE (client tokenizer)
      ▼
 ③ adapter.encode_query()  ─────────────────► ④ HTTP POST ────────► ⑤ pre-inference
      │                                                                (template? tokenize?)
      │                                                                        │
      │                                                                  [ inference ]
      │                                                                        │
 ⑦ accumulator + adapter.decode_*  ◄────────── ⑥ HTTP response ◄────────────────┘
      │        QueryResult{response_output, metadata}  /  StreamChunk
      ▼
   EventRecord(ISSUED,   PromptData)          ──┬──► events.jsonl
   EventRecord(COMPLETE, TextModelOutput)     ──┘         │
                                                          ▼
                                            MetricsAggregator subprocess
                                                          │
                                                          ▼
                                              ⑧ ISL / OSL / TPOT
```

Stage ⑤ is the fork the whole design turns on: either the server tokenizes (chat completions) or
it does not (token-ID prompts). Everything else follows.

## 3. The spec sheet

Every reference implementation fills this form. Copy it verbatim into the workload's
`examples/<NN>_<Name>/README.md` and fill every row; `n/a` is an acceptable value, a blank is not.

Rows marked **(derived)** are not free-form — they are looked up from earlier rows using the
tables in §4. If a workload cannot be expressed by the derived rules, that is a signal the rule
set needs extending; record the reason in `Deviations` and open an issue rather than
hand-rolling a count.

| #                               | Field                                            | Value                                                                                                           |
| ------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Identity**                    |                                                  |                                                                                                                 |
| 1                               | Workload / model                                 |                                                                                                                 |
| 2                               | `api_type` + route                               |                                                                                                                 |
| 3                               | Server + launch flags affecting token accounting |                                                                                                                 |
| 4                               | Streaming                                        | on / off                                                                                                        |
| **Reference pin** (client side) |                                                  |                                                                                                                 |
| 5                               | Reference tokenizer                              | HF `repo@revision`, or `harmony:<encoding>`, or `none`                                                          |
| 6                               | Chat template source + pin                       | tokenizer's `chat_template` / explicit file + sha256 / GGUF-embedded / `n/a`                                    |
| 7                               | `chat_template_kwargs` sent                      | exact dict, or `none`                                                                                           |
| 8                               | Reasoning-delimiter constant                     | tokens to re-add when the server strips reasoning into a separate field (integer, default `0`)                  |
| 9                               | Count EOS in OSL                                 | yes / no                                                                                                        |
| **Input** (drives ISL)          |                                                  |                                                                                                                 |
| 10                              | Input form on the wire                           | `token_ids` \| `chat_messages` \| `raw_text` \| `content_parts` \| `none`                                       |
| 11                              | Who tokenizes the input                          | client / server                                                                                                 |
| 12                              | Input record                                     | the exact row fields that constitute the input (what `PromptData` must carry)                                   |
| 13                              | Input tokens invisible to client                 | media tokens, server-injected request fields, generation stubs — with measured size when known                  |
| **Output** (drives OSL / TPOT)  |                                                  |                                                                                                                 |
| 14                              | Response fields returned                         | `output_ids+text` \| `text` \| `content` \| `content+reasoning` \| `content+reasoning+tool_calls` \| `non-text` |
| 15                              | Server-side output parsing                       | none / reasoning parser / tool parser / both — **and whether it is config-locked**                              |
| 16                              | Output token IDs available                       | yes (field name) / no                                                                                           |
| 17                              | Special tokens preserved in text                 | yes / no (`skip_special_tokens`)                                                                                |
| 18                              | First-chunk marker for TPOT                      | how the first streamed chunk is identified                                                                      |
| **Derivation**                  |                                                  |                                                                                                                 |
| 19                              | **ISL rule** (derived)                           | from row 10 via §4.1                                                                                            |
| 20                              | **OSL rule** (derived)                           | from row 14 via §4.2                                                                                            |
| 21                              | **TPOT rule** (derived)                          | from row 4 via §4.3                                                                                             |
| 22                              | Known bias vs server `usage`                     | Δ% ISL / Δ% OSL from `token_accounting.json`, or `unmeasured`                                                   |
| 23                              | Deviations & rationale                           | free text — anything the rules above do not express                                                             |

Rows 5–9 are the only per-model _data_. Rows 19–21 are per-model _nothing_. That split is the
point: adding a model is a data exercise, not a code exercise.

## 4. Derivation rules

These are shared by every workload. They are the complete definition of the metrics.

Common bindings: `tok` is the row-5 reference tokenizer. `render(msgs, **kw)` means
`tok.apply_chat_template(msgs, tokenize=True, return_dict=False, **kw)` — `return_dict` **must**
be passed explicitly, because it defaults to `True` on `transformers==5.5.0` and the return is
then a `BatchEncoding` whose `len()` is the key count, not the token count.

### 4.1 ISL

| Row 10 (input form) | Rule    | Formula                                                                                                                  | Exactness                                                 |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `token_ids`         | **I-1** | `ISL = len(input_token_ids)`                                                                                             | Exact. The client sent them; nothing to reconstruct.      |
| `chat_messages`     | **I-2** | `ISL = len(render(messages, tools=tools, add_generation_prompt=True, **chat_template_kwargs))`                           | Bounded by rows 6–7. Residual reported in rows 13 and 22. |
| `raw_text`          | **I-3** | `ISL = len(tok(text, add_special_tokens=S).input_ids)` where `S` matches the server's completions tokenization setting   | Near-exact; only the special-token convention can differ. |
| `content_parts`     | **I-4** | Text parts via I-3; media tokens declared **unavailable** in row 13 and reported as such — never silently folded into 0. | Text-exact, media-blind.                                  |
| `none`              | **I-0** | Not reported.                                                                                                            | —                                                         |

### 4.2 OSL

Applied to the completed response. The governing idea: **generated text is counted as text; only
structural syntax goes through the template.**

| Row 14 (response fields)       | Rule    | Formula                                                                                      |
| ------------------------------ | ------- | -------------------------------------------------------------------------------------------- |
| `output_ids+text`              | **O-1** | `OSL = len(output_token_ids)`, gated on `tok.decode(ids) == text`; mismatch is a hard error. |
| `text` / `content`             | **O-2** | `OSL = len(tok(text, add_special_tokens=False).input_ids) + eos` (row 9)                     |
| `content+reasoning`            | **O-3** | `OSL = O-2(content) + len(tok(reasoning)) + reasoning_delimiters` (row 8)                    |
| `content+reasoning+tool_calls` | **O-4** | `OSL = O-3 + struct_delta`                                                                   |
| `non-text`                     | **O-0** | Not reported.                                                                                |

```
struct_delta = len(render([U, A(content="", tool_calls=TC)], add_generation_prompt=False))
             − len(render([U, A(content="")],              add_generation_prompt=False))

U  = {"role": "user", "content": ""}
A  = assistant message
TC = tool calls, with function.arguments normalized from the OpenAI wire JSON *string*
     to a dict (Hermes/Qwen-style templates iterate it as a mapping and diverge otherwise)
```

Subtracting the empty-assistant render removes the assistant frame — which the model does not
generate — while keeping the tool-call syntax, which it does.

**Reasoning must not be routed through `apply_chat_template`.** Input-side templates strip
`reasoning_content` from assistant history by design; the render succeeds, drops the reasoning,
and returns a plausible number. There is no exception to catch, so no fallback fires. Reasoning
is raw generated text and is counted as raw generated text (O-3). Row 8 carries only the
delimiter tokens (`<think>` / `</think>`, channel markers) that the server's parser consumed.

### 4.3 TPOT

| Row 4     | Rule    | Formula                                                             |
| --------- | ------- | ------------------------------------------------------------------- |
| streaming | **T-1** | `TPOT = (complete_ns − recv_first_ns) / (OSL − first_chunk_tokens)` |
| off       | **T-2** | Not reported.                                                       |

`first_chunk_tokens` is counted with the same OSL rule applied to the first chunk's payload.
The denominator is derived _from_ OSL rather than by independently re-tokenizing the tail:
`tokenize(tail) ≠ tokenize(whole) − tokenize(head)`, so an independent tail count is both a
second tokenization pass and a silent inconsistency with the reported OSL.

## 5. Filled sheets — current workloads

Values reflect `main` as of this document. `⚠` marks a cell that does not match the rule the row
above it derives, i.e. a gap in the current implementation.

### 5.1 GPT-OSS-120B / SGLang

| #   | Field                  | Value                                                                                                                                                                                                                                               |
| --- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `openai/gpt-oss-120b`                                                                                                                                                                                                                               |
| 2   | `api_type` + route     | `sglang` → `POST /generate`                                                                                                                                                                                                                         |
| 3   | Server + flags         | SGLang; no `separate_reasoning`, no tool parser                                                                                                                                                                                                     |
| 4   | Streaming              | on                                                                                                                                                                                                                                                  |
| 5   | Reference tokenizer    | `harmony:HARMONY_GPT_OSS` (o200k_harmony) for input; `openai/gpt-oss-120b` HF for output                                                                                                                                                            |
| 6   | Chat template          | `n/a` — Harmony conversation rendered client-side by `Harmonize()`                                                                                                                                                                                  |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                                                                                |
| 8   | Reasoning delimiters   | 0 — channel markers stay inline in `text`                                                                                                                                                                                                           |
| 9   | Count EOS              | no (current); see Deviations                                                                                                                                                                                                                        |
| 10  | Input form             | `token_ids`                                                                                                                                                                                                                                         |
| 11  | Who tokenizes input    | client                                                                                                                                                                                                                                              |
| 12  | Input record           | `input_tokens` (parquet col; accuracy sets go `prompt` → `Harmonize()` → `input_tokens`)                                                                                                                                                            |
| 13  | Invisible input tokens | SGLang generation stub, ~3 tok                                                                                                                                                                                                                      |
| 14  | Response fields        | `output_ids+text`                                                                                                                                                                                                                                   |
| 15  | Server output parsing  | none — structurally locked by the `/generate` route                                                                                                                                                                                                 |
| 16  | Output token IDs       | **yes** — `output_ids` (delta per chunk when streaming)                                                                                                                                                                                             |
| 17  | Special tokens in text | yes — Harmony channel markers preserved                                                                                                                                                                                                             |
| 18  | First-chunk marker     | `SGLangSSEAccumulator` `first_chunk` metadata                                                                                                                                                                                                       |
| 19  | ISL rule               | **I-1**                                                                                                                                                                                                                                             |
| 20  | OSL rule               | **O-1** — but ⚠ currently **O-2**                                                                                                                                                                                                                  |
| 21  | TPOT rule              | **T-1**                                                                                                                                                                                                                                             |
| 22  | Bias vs server         | unmeasured                                                                                                                                                                                                                                          |
| 23  | Deviations             | `output_ids` and `meta_info.completion_tokens` are captured into `QueryResult.metadata` and then dropped — `session.py` forwards only `finish_reason` and `worker_id` to the event stream. An exact, client-verifiable OSL is available and unused. |

### 5.2 GPT-OSS-120B / vLLM

| #   | Field                  | Value                                                                                                                                                                               |
| --- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `openai/gpt-oss-120b`                                                                                                                                                               |
| 2   | `api_type` + route     | `openai_completions` → `POST /v1/completions`                                                                                                                                       |
| 3   | Server + flags         | vLLM; `/v1/completions` runs no parsers by construction                                                                                                                             |
| 4   | Streaming              | on                                                                                                                                                                                  |
| 5   | Reference tokenizer    | `harmony:HARMONY_GPT_OSS` in; `openai/gpt-oss-120b` HF out                                                                                                                          |
| 6   | Chat template          | `n/a`                                                                                                                                                                               |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                |
| 8   | Reasoning delimiters   | 0 — inline                                                                                                                                                                          |
| 9   | Count EOS              | no (current)                                                                                                                                                                        |
| 10  | Input form             | `token_ids`                                                                                                                                                                         |
| 11  | Who tokenizes input    | client                                                                                                                                                                              |
| 12  | Input record           | `input_tokens`                                                                                                                                                                      |
| 13  | Invisible input tokens | none                                                                                                                                                                                |
| 14  | Response fields        | `text`                                                                                                                                                                              |
| 15  | Server output parsing  | none — structurally locked                                                                                                                                                          |
| 16  | Output token IDs       | no (vLLM can echo them; not requested today)                                                                                                                                        |
| 17  | Special tokens in text | yes — `skip_special_tokens: false`                                                                                                                                                  |
| 18  | First-chunk marker     | `OpenAISSEAccumulator` `first_chunk`                                                                                                                                                |
| 19  | ISL rule               | **I-1**                                                                                                                                                                             |
| 20  | OSL rule               | **O-2**                                                                                                                                                                             |
| 21  | TPOT rule              | **T-1**                                                                                                                                                                             |
| 22  | Bias vs server         | unmeasured — `usage` is decoded and discarded                                                                                                                                       |
| 23  | Deviations             | Harmony markers are inside the counted text, so OSL depends on the tokenizer treating them as single special tokens. Requesting echoed token IDs would promote this row to **O-1**. |

### 5.3 DeepSeek-R1 / TensorRT-LLM

| #   | Field                  | Value                                                                                                                                                                                                                                                                                           |
| --- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `deepseek-ai/DeepSeek-R1`                                                                                                                                                                                                                                                                       |
| 2   | `api_type` + route     | `openai_completions` → `POST /v1/completions`                                                                                                                                                                                                                                                   |
| 3   | Server + flags         | trtllm-serve (OpenAI-compatible); no parsers on this route                                                                                                                                                                                                                                      |
| 4   | Streaming              | on                                                                                                                                                                                                                                                                                              |
| 5   | Reference tokenizer    | `deepseek-ai/DeepSeek-R1` (`--tokenizer`)                                                                                                                                                                                                                                                       |
| 6   | Chat template          | `n/a` — MLPerf prompt pre-tokenized offline                                                                                                                                                                                                                                                     |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                                                                                                                            |
| 8   | Reasoning delimiters   | 0 — `<think>…</think>` inline in `text`                                                                                                                                                                                                                                                         |
| 9   | Count EOS              | no (current)                                                                                                                                                                                                                                                                                    |
| 10  | Input form             | `token_ids`                                                                                                                                                                                                                                                                                     |
| 11  | Who tokenizes input    | client (offline, into the parquet)                                                                                                                                                                                                                                                              |
| 12  | Input record           | `input_tokens`                                                                                                                                                                                                                                                                                  |
| 13  | Invisible input tokens | none                                                                                                                                                                                                                                                                                            |
| 14  | Response fields        | `text`                                                                                                                                                                                                                                                                                          |
| 15  | Server output parsing  | none                                                                                                                                                                                                                                                                                            |
| 16  | Output token IDs       | no                                                                                                                                                                                                                                                                                              |
| 17  | Special tokens in text | server-dependent                                                                                                                                                                                                                                                                                |
| 18  | First-chunk marker     | `OpenAISSEAccumulator` `first_chunk`                                                                                                                                                                                                                                                            |
| 19  | ISL rule               | **I-1**                                                                                                                                                                                                                                                                                         |
| 20  | OSL rule               | **O-2**                                                                                                                                                                                                                                                                                         |
| 21  | TPOT rule              | **T-1**                                                                                                                                                                                                                                                                                         |
| 22  | Bias vs server         | unmeasured                                                                                                                                                                                                                                                                                      |
| 23  | Deviations             | `Harmonize()` sits in the `openai_completions` transform chain and no-ops only because `input_tokens` is already present. Its defaults are `openai/gpt-oss-120b` + `HARMONY_GPT_OSS`, so a DS-R1 dataset carrying only `prompt` would be silently encoded with the wrong tokenizer. Guard this. |

### 5.4 Llama-3.1-8B / CNN-DailyMail

| #   | Field                  | Value                                                                                                                                                                                                                      |
| --- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `meta-llama/Llama-3.1-8B-Instruct`                                                                                                                                                                                         |
| 2   | `api_type` + route     | `openai` → `POST /v1/chat/completions`                                                                                                                                                                                     |
| 3   | Server + flags         | no reasoning or tool parser configured                                                                                                                                                                                     |
| 4   | Streaming              | off (offline config)                                                                                                                                                                                                       |
| 5   | Reference tokenizer    | `meta-llama/Llama-3.1-8B-Instruct`                                                                                                                                                                                         |
| 6   | Chat template          | tokenizer's `chat_template` — **not currently pinned or hashed**                                                                                                                                                           |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                                                       |
| 8   | Reasoning delimiters   | 0                                                                                                                                                                                                                          |
| 9   | Count EOS              | no (current)                                                                                                                                                                                                               |
| 10  | Input form             | `chat_messages`                                                                                                                                                                                                            |
| 11  | Who tokenizes input    | **server**                                                                                                                                                                                                                 |
| 12  | Input record           | `prompt` (str) → wrapped by the adapter into `[{role:user, content:prompt}]` (+ `system`)                                                                                                                                  |
| 13  | Invisible input tokens | chat-template frame + BOS — currently **all of it**                                                                                                                                                                        |
| 14  | Response fields        | `content`                                                                                                                                                                                                                  |
| 15  | Server output parsing  | none — but **not config-locked**; enabling a parser would silently change row 14                                                                                                                                           |
| 16  | Output token IDs       | no                                                                                                                                                                                                                         |
| 17  | Special tokens in text | no                                                                                                                                                                                                                         |
| 18  | First-chunk marker     | n/a (non-streaming)                                                                                                                                                                                                        |
| 19  | ISL rule               | **I-2**                                                                                                                                                                                                                    |
| 20  | OSL rule               | **O-2**                                                                                                                                                                                                                    |
| 21  | TPOT rule              | **T-2**                                                                                                                                                                                                                    |
| 22  | Bias vs server         | unmeasured                                                                                                                                                                                                                 |
| 23  | Deviations             | ⚠ ISL currently uses **I-3** on the bare article string — the template frame, BOS, and system message are all missing. This is the mildest instance of the shared `PromptData.text` defect; OSL is the fleet's best case. |

### 5.5 Qwen3-VL-235B-A22B / Shopify

| #   | Field                  | Value                                                                                                                                                                                                                |
| --- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `Qwen/Qwen3-VL-235B-A22B-Instruct`                                                                                                                                                                                   |
| 2   | `api_type` + route     | `openai` → `POST /v1/chat/completions`                                                                                                                                                                               |
| 3   | Server + flags         | vision encoder active                                                                                                                                                                                                |
| 4   | Streaming              | off (offline config)                                                                                                                                                                                                 |
| 5   | Reference tokenizer    | `Qwen/Qwen3-VL-235B-A22B-Instruct`                                                                                                                                                                                   |
| 6   | Chat template          | tokenizer's `chat_template` — not pinned                                                                                                                                                                             |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                                                 |
| 8   | Reasoning delimiters   | 0                                                                                                                                                                                                                    |
| 9   | Count EOS              | no (current)                                                                                                                                                                                                         |
| 10  | Input form             | `content_parts`                                                                                                                                                                                                      |
| 11  | Who tokenizes input    | **server**                                                                                                                                                                                                           |
| 12  | Input record           | `prompt` = `[{type:text},{type:image_url,…base64…}]`, `system`                                                                                                                                                       |
| 13  | Invisible input tokens | **image tokens** — produced by the vision encoder, not derivable client-side                                                                                                                                         |
| 14  | Response fields        | `content`                                                                                                                                                                                                            |
| 15  | Server output parsing  | none; not config-locked                                                                                                                                                                                              |
| 16  | Output token IDs       | no                                                                                                                                                                                                                   |
| 17  | Special tokens in text | no                                                                                                                                                                                                                   |
| 18  | First-chunk marker     | n/a                                                                                                                                                                                                                  |
| 19  | ISL rule               | **I-4**                                                                                                                                                                                                              |
| 20  | OSL rule               | **O-2**                                                                                                                                                                                                              |
| 21  | TPOT rule              | **T-2**                                                                                                                                                                                                              |
| 22  | Bias vs server         | unmeasured                                                                                                                                                                                                           |
| 23  | Deviations             | ⚠ ISL is **not recorded at all**: a list-valued `prompt` yields `PromptData(text=None)`. I-4 would at least report the text component with `media_tokens: unavailable`, which is strictly more honest than silence. |

### 5.6 Agentic — Kimi

| #   | Field                  | Value                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | Kimi (served as `/model`)                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2   | `api_type` + route     | `openai` → `POST /v1/chat/completions`                                                                                                                                                                                                                                                                                                                                                                                                 |
| 3   | Server + flags         | SGLang `--reasoning-parser kimi_k2 --tool-call-parser kimi_k2`                                                                                                                                                                                                                                                                                                                                                                         |
| 4   | Streaming              | on                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 5   | Reference tokenizer    | ⚠ **unresolved** — `model_params.name: "/model"`, no `tokenizer_name` override                                                                                                                                                                                                                                                                                                                                                        |
| 6   | Chat template          | Kimi template — not pinned                                                                                                                                                                                                                                                                                                                                                                                                             |
| 7   | `chat_template_kwargs` | `{thinking: true, preserve_thinking: true}`                                                                                                                                                                                                                                                                                                                                                                                            |
| 8   | Reasoning delimiters   | to be measured                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 9   | Count EOS              | to be decided                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 10  | Input form             | `chat_messages`                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 11  | Who tokenizes input    | **server**                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 12  | Input record           | `messages` (full pre-built history: system + prior assistant turns with `reasoning_content` and `tool_calls` + tool results), `tools` (~62 defs), `chat_template_kwargs`                                                                                                                                                                                                                                                               |
| 13  | Invisible input tokens | SGLang injects `function.strict:false` and `defer_loading:null` into every tool def (**+496 tok**) and excludes a 3-tok generation stub from reported `prompt_tokens` → net **−493 tok/turn** vs client                                                                                                                                                                                                                                |
| 14  | Response fields        | `content+reasoning+tool_calls`                                                                                                                                                                                                                                                                                                                                                                                                         |
| 15  | Server output parsing  | reasoning **and** tool parser — **not config-locked**; structurally required (the harness needs parsed tool calls)                                                                                                                                                                                                                                                                                                                     |
| 16  | Output token IDs       | no                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 17  | Special tokens in text | no                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 18  | First-chunk marker     | `OpenAISSEAccumulator`, incl. the zero-length sentinel for pure tool-call deltas                                                                                                                                                                                                                                                                                                                                                       |
| 19  | ISL rule               | **I-2**                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 20  | OSL rule               | **O-4**                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 21  | TPOT rule              | **T-1**                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 22  | Bias vs server         | ISL −1.38%, OSL −3.26% (25-turn Kimi K3 / SGLang run; ISL delta is exactly −493/turn)                                                                                                                                                                                                                                                                                                                                                  |
| 23  | Deviations             | ⚠ **No token metrics are produced at all today.** `_check_tokenizer_exists("/model")` fails on the client, the aggregator is launched without `--tokenizer`, and ISL/OSL/TPOT are silently absent — not merely biased. Fix row 5 before anything else. Separately, ISL currently derives from `_extract_prompt_text()`, a newline join of message contents with no roles, no tool definitions, no template, and no generation prompt. |

### 5.7 Edge Agentic — Qwen3.6-27B

| #   | Field                  | Value                                                                                                                                                                                                                                                                                         |
| --- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model       | `Qwen3.6-27B-Q4_K_M` (GGUF)                                                                                                                                                                                                                                                                   |
| 2   | `api_type` + route     | `openai` → `POST /v1/chat/completions`                                                                                                                                                                                                                                                        |
| 3   | Server + flags         | `llama-server --reasoning off -np 1 --ctx-size 32768 --seed 42` — config-locked and attested in `compliance/checker.py`                                                                                                                                                                       |
| 4   | Streaming              | off                                                                                                                                                                                                                                                                                           |
| 5   | Reference tokenizer    | `Qwen/Qwen3.6-27B` (explicit `tokenizer_name` override)                                                                                                                                                                                                                                       |
| 6   | Chat template          | GGUF-embedded Jinja — client uses the HF tokenizer's template; **equivalence unverified**                                                                                                                                                                                                     |
| 7   | `chat_template_kwargs` | none                                                                                                                                                                                                                                                                                          |
| 8   | Reasoning delimiters   | 0 — reasoning disabled at the server                                                                                                                                                                                                                                                          |
| 9   | Count EOS              | to be decided                                                                                                                                                                                                                                                                                 |
| 10  | Input form             | `chat_messages`                                                                                                                                                                                                                                                                               |
| 11  | Who tokenizes input    | **server**                                                                                                                                                                                                                                                                                    |
| 12  | Input record           | `messages` (SWE-bench system prompt + growing turn history), `tools`                                                                                                                                                                                                                          |
| 13  | Invisible input tokens | any divergence between the GGUF-embedded template and the HF template                                                                                                                                                                                                                         |
| 14  | Response fields        | `content+tool_calls`                                                                                                                                                                                                                                                                          |
| 15  | Server output parsing  | tool parser only; reasoning **off** — config-locked                                                                                                                                                                                                                                           |
| 16  | Output token IDs       | no                                                                                                                                                                                                                                                                                            |
| 17  | Special tokens in text | no                                                                                                                                                                                                                                                                                            |
| 18  | First-chunk marker     | n/a                                                                                                                                                                                                                                                                                           |
| 19  | ISL rule               | **I-2**                                                                                                                                                                                                                                                                                       |
| 20  | OSL rule               | **O-4** with the reasoning term zero                                                                                                                                                                                                                                                          |
| 21  | TPOT rule              | **T-2**                                                                                                                                                                                                                                                                                       |
| 22  | Bias vs server         | unmeasured                                                                                                                                                                                                                                                                                    |
| 23  | Deviations             | ⚠ ISL currently uses `_extract_prompt_text()`; the under-count **grows with turn depth** because the untemplated history grows. This is the workload where the existing ruleset already demonstrates the config-lock pattern (row 3) — extending it is cheaper than modelling parser output. |

### 5.8 WAN 2.2 T2V / VideoGen

| #   | Field                 | Value                                                                                                                                                                                                                        |
| --- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Workload / model      | `wan22`                                                                                                                                                                                                                      |
| 2   | `api_type` + route    | `videogen` → `POST /v1/videos/generations`                                                                                                                                                                                   |
| 3   | Server + flags        | trtllm-serve; `response_format: video_path`                                                                                                                                                                                  |
| 4   | Streaming             | off                                                                                                                                                                                                                          |
| 5   | Reference tokenizer   | `none`                                                                                                                                                                                                                       |
| 6–9 | Reference pin         | `n/a`                                                                                                                                                                                                                        |
| 10  | Input form            | `raw_text` (prompt string; token accounting not meaningful)                                                                                                                                                                  |
| 11  | Who tokenizes input   | server (T5/UMT5 text encoder — not LLM tokens)                                                                                                                                                                               |
| 12  | Input record          | `prompt`                                                                                                                                                                                                                     |
| 13  | Invisible input       | all of it                                                                                                                                                                                                                    |
| 14  | Response fields       | `non-text` (`video_path`)                                                                                                                                                                                                    |
| 15  | Server output parsing | n/a                                                                                                                                                                                                                          |
| 16  | Output token IDs      | no                                                                                                                                                                                                                           |
| 17  | Special tokens        | n/a                                                                                                                                                                                                                          |
| 18  | First-chunk marker    | n/a                                                                                                                                                                                                                          |
| 19  | ISL rule              | **I-0**                                                                                                                                                                                                                      |
| 20  | OSL rule              | **O-0**                                                                                                                                                                                                                      |
| 21  | TPOT rule             | **T-2**                                                                                                                                                                                                                      |
| 22  | Bias vs server        | n/a                                                                                                                                                                                                                          |
| 23  | Deviations            | Token metrics are correctly absent. `PromptData(text=prompt)` is still populated; the metrics are off only because `_check_tokenizer_exists("wan22")` happens to fail. Declare `none` explicitly instead of relying on that. |

## 6. At-a-glance

| Workload         | Input form      | Who tokenizes in | Response fields                | ISL    | OSL          | TPOT | Status                                 |
| ---------------- | --------------- | ---------------- | ------------------------------ | ------ | ------------ | ---- | -------------------------------------- |
| GPT-OSS / SGLang | `token_ids`     | client           | `output_ids+text`              | I-1    | O-1 (⚠ O-2) | T-1  | exact ISL; exact OSL available, unused |
| GPT-OSS / vLLM   | `token_ids`     | client           | `text`                         | I-1    | O-2          | T-1  | exact ISL                              |
| DS-R1 / TRT-LLM  | `token_ids`     | client           | `text`                         | I-1    | O-2          | T-1  | exact ISL                              |
| Llama-3.1 CNN    | `chat_messages` | server           | `content`                      | ⚠ I-3 | O-2          | T-2  | ISL missing template frame             |
| Qwen3-VL         | `content_parts` | server           | `content`                      | ⚠ —   | O-2          | T-2  | ISL not recorded                       |
| Agentic Kimi     | `chat_messages` | server           | `content+reasoning+tool_calls` | ⚠ —   | ⚠ —         | ⚠ — | **no tokenizer resolves → no metrics** |
| Edge Agentic     | `chat_messages` | server           | `content+tool_calls`           | ⚠ I-3 | ⚠ approx    | T-2  | ISL under-count grows with turn depth  |
| WAN 2.2          | `raw_text`      | server           | `non-text`                     | I-0    | O-0          | T-2  | correct by absence                     |

Three of eight workloads report no usable ISL. One reports no token metrics at all.

## 7. Calibration artifact

Every run emits `token_accounting.json` beside the report:

- per sample: client ISL/OSL under the rules above, the server's `usage.prompt_tokens` /
  `completion_tokens`, and `len(output_ids)` where the response carries them;
- aggregate: the delta distribution, which populates row 22 of each spec sheet;
- where output token IDs are present: the result of the `decode(ids) == text` gate, plus the
  delta between `len(ids)` and the client's canonical encoding of the same text — a systematic
  gap there is the signature of a non-canonical tokenization.

Server numbers are recorded **only** as comparison artifacts. They never become metrics. The
point of the artifact is that row 22 is a measurement rather than an assumption — the Kimi
figures in §5.6 came from exactly this comparison run by hand, and every model after it deserves
the same evidence without the manual effort.

Getting `usage` on streaming runs additionally requires `stream_options: {"include_usage": true}`,
which `ChatCompletionRequest` does not currently carry.

## 8. Implementation consequences

Ordered by what unblocks the most.

1. **Resolve the tokenizer, or fail loudly.** A workload whose row 5 does not resolve currently
   produces no token metrics and no error. An undeclared or unresolvable reference tokenizer must
   be a startup failure, not silence.
2. **Carry the real input.** `PromptData` must hold `messages`, `tools`, and
   `chat_template_kwargs` so I-2 has something to work with. This is what PR #441 does and is the
   reason to land it.
3. **Forward the output evidence.** `QueryResult.metadata` is dropped at the event boundary;
   `output_ids` on the SGLang path is exact, verifiable, and already parsed.
4. **Apply O-3 rather than routing reasoning through the input template.** Text as text,
   structure through the template.
5. **Pin rows 5–7 in the ruleset as data**, with an unknown model raising rather than falling
   back to concatenation. That is the mechanism that keeps this from rotting as models are added.
6. **Extend the config-lock.** Every single-turn chat workload can run with no reasoning parser
   and no tool parser, which collapses row 14 to `content` and row 20 to O-2. Edge Agentic already
   does this for reasoning. Agentic is the only workload that structurally cannot.
7. **Shard the I-2 lane.** `apply_chat_template` is Jinja rendering in Python — it does not go
   through the Rust batch encoder. Once I-2 is the default for every chat workload it is the
   slowest path in the aggregator, and rendering it serially under the flush lock risks exhausting
   the drain budget and leaving ISL permanently uncounted. Render inside the shard workers and
   memoize on a `(messages, tools)` hash; in most benchmarks the system prompt and tool
   definitions are identical across every sample.

## 9. PR #441 checklist

PR #441 implements the I-2 direction. It should land with these fixes; items 1–3 are correctness.

1. Pass `return_dict=False` at every `tokenize=True` call site. On `transformers==5.5.0` the
   default is `True` and `len()` returns the `BatchEncoding` key count — structured ISL collapses
   to `2` and structured OSL/TPOT to `0`. The test fake returns a bare list, which is why the
   suite stays green; make it model a `BatchEncoding`.
2. Count reasoning as raw text per O-3. Routing it through the input template drops it silently —
   the render succeeds, so no fallback fires.
3. Batch and shard the prompt lane; stop awaiting each item under `_lock`. On `main` this lane was
   justified as rare; the PR makes it the path for every chat ISL sample.
4. Thread `chat_template_kwargs` through `PromptData` — required by I-2 and by row 7.
5. Guard empty and multimodal `messages`: `isinstance([], list)` is `True`, and
   `apply_chat_template([])` raises `IndexError`.
6. Normalize wire-JSON `arguments` strings in prompt history, and give the prompt path the same
   try/fallback the message path has.
7. Declare `tiktoken` in `pyproject.toml`, or the custom-tokenizer path is unreachable in a clean
   install.
8. Document and warn on the `token_ids` > `messages` > `prompt` priority when more than one is
   present.
9. Remove `_extract_prompt_text`, or wire it in as the render-failure fallback.
