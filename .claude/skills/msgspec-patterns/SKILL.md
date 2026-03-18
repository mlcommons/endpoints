---
name: msgspec-patterns
description: Reference guide for msgspec.Struct usage patterns and performance tips. Use when writing or reviewing code that defines msgspec Structs, encodes/decodes data, or needs performance optimization for serialization.
user-invocable: false
---

## Use Structs for Structured Data

Always prefer `msgspec.Struct` over `dict`, `dataclasses`, or `attrs` for structured data with a known schema. Structs are 5-60x faster for common operations.

## Struct Configuration Options

| Option                  | Description                                   | Default |
| ----------------------- | --------------------------------------------- | ------- |
| `omit_defaults`         | Omit fields with default values when encoding | `False` |
| `forbid_unknown_fields` | Error on unknown fields when decoding         | `False` |
| `frozen`                | Make instances immutable and hashable         | `False` |
| `kw_only`               | Make all fields keyword-only                  | `False` |
| `tag`                   | Enable tagged union support                   | `None`  |
| `array_like`            | Encode/decode as arrays instead of objects    | `False` |
| `gc`                    | Enable garbage collector tracking             | `True`  |

## Omit Default Values

Set `omit_defaults=True` when default values are known on both encoding and decoding ends. Reduces encoded message size and improves performance.

```python
class Config(msgspec.Struct, omit_defaults=True):
    host: str = "localhost"
    port: int = 8080
```

## Avoid Decoding Unused Fields

Define smaller "view" Struct types that only contain the fields you actually need. msgspec skips decoding fields not defined in your Struct.

## Use `encode_into` for Buffer Reuse

In hot loops, use `Encoder.encode_into()` with a pre-allocated `bytearray` instead of `encode()`. Always measure before adopting.

```python
encoder = msgspec.json.Encoder()
buffer = bytearray(1024)
n = encoder.encode_into(msg, buffer)
socket.sendall(memoryview(buffer)[:n])
```

## Use MessagePack for Internal APIs

`msgspec.msgpack` is more compact and can be more performant than `msgspec.json` for internal service communication.

## gc=False

Set `gc=False` on Struct types that will never participate in reference cycles. Reduces GC overhead by up to 75x and saves 16 bytes per instance. See the `msgspec-struct-gc-check` skill for the full safety analysis.

## array_like=True

Set `array_like=True` when both ends know the field schema. Encodes structs as arrays instead of objects, removing field names from the message.

```python
class Point(msgspec.Struct, array_like=True):
    x: float
    y: float
# Encodes as [1.0, 2.0] instead of {"x": 1.0, "y": 2.0}
```

## Tagged Unions

Use `tag=True` on Struct types when handling multiple message types in a single union for efficient type discrimination during decoding.

```python
class GetRequest(msgspec.Struct, tag=True):
    key: str

class PutRequest(msgspec.Struct, tag=True):
    key: str
    value: str

Request = GetRequest | PutRequest
decoder = msgspec.msgpack.Decoder(Request)
```

## NDJSON with encode_into

For line-delimited JSON, use `encode_into()` with `buffer.extend()` to avoid copies:

```python
encoder = msgspec.json.Encoder()
buffer = bytearray(64)
n = encoder.encode_into(msg, buffer)
file.write(memoryview(buffer)[:n])
file.write(b"\n")
```
