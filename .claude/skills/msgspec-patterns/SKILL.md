---
name: msgspec-patterns
description: Reference guide for msgspec.Struct usage patterns, performance tips, and gc=False safety analysis. Use when writing or reviewing msgspec Struct definitions, encoding/decoding code, or deciding whether gc=False is safe.
allowed-tools: Read, Grep, Glob
---

## Use Structs for Structured Data

Always prefer `msgspec.Struct` over `dict`, `dataclasses`, or `attrs` for structured data with a known schema. Structs are 5-60x faster for common operations and are optimized for encoding/decoding.

```python
# BAD
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int

# GOOD
import msgspec

class User(msgspec.Struct):
    name: str
    email: str
    age: int

user = User(name="alice", email="alice@example.com", age=30)
data = msgspec.json.encode(user)
decoded = msgspec.json.decode(data, type=User)
```

## Struct Configuration Options

| Option                  | Description                                   | Default  |
| ----------------------- | --------------------------------------------- | -------- |
| `omit_defaults`         | Omit fields with default values when encoding | `False`  |
| `forbid_unknown_fields` | Error on unknown fields when decoding         | `False`  |
| `frozen`                | Make instances immutable and hashable         | `False`  |
| `order`                 | Generate ordering methods (`__lt__`, etc.)    | `False`  |
| `eq`                    | Generate equality methods                     | `True`   |
| `kw_only`               | Make all fields keyword-only                  | `False`  |
| `tag`                   | Enable tagged union support                   | `None`   |
| `tag_field`             | Field name for the tag                        | `"type"` |
| `rename`                | Rename fields for encoding/decoding           | `None`   |
| `array_like`            | Encode/decode as arrays instead of objects    | `False`  |
| `gc`                    | Enable garbage collector tracking             | `True`   |
| `weakref`               | Enable weak reference support                 | `False`  |
| `dict`                  | Add `__dict__` attribute                      | `False`  |
| `cache_hash`            | Cache the hash value                          | `False`  |

## Omit Default Values

Set `omit_defaults=True` when default values are known on both encoding and decoding ends. Reduces encoded message size and improves performance.

```python
class Config(msgspec.Struct, omit_defaults=True):
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

config = Config(host="production.example.com")
msgspec.json.encode(config)
# b'{"host":"production.example.com"}' — port and debug omitted
```

## Avoid Decoding Unused Fields

Define smaller "view" Struct types that only contain the fields you actually need. msgspec skips decoding fields not defined in your Struct, reducing allocations and CPU time.

```python
# BAD: decodes entire object
class FullTweet(msgspec.Struct):
    id: int
    full_text: str
    user: dict
    entities: dict
    retweet_count: int
    favorite_count: int
    # ... many more fields

# GOOD: only these fields are decoded, the rest is skipped
class User(msgspec.Struct):
    name: str

class TweetView(msgspec.Struct):
    user: User
    full_text: str
    favorite_count: int
```

## array_like=True

Set `array_like=True` when both ends know the field schema. Encodes structs as arrays instead of objects, removing field names from the message — smaller and faster.

```python
class Point(msgspec.Struct, array_like=True):
    x: float
    y: float
    z: float

point = Point(1.0, 2.0, 3.0)
msgspec.json.encode(point)
# b'[1.0,2.0,3.0]' instead of b'{"x":1.0,"y":2.0,"z":3.0}'
```

## Tagged Unions

Use `tag=True` on Struct types when handling multiple message types in a single union for efficient type discrimination during decoding.

```python
class GetRequest(msgspec.Struct, tag=True):
    key: str

class PutRequest(msgspec.Struct, tag=True):
    key: str
    value: str

class DeleteRequest(msgspec.Struct, tag=True):
    key: str

Request = GetRequest | PutRequest | DeleteRequest
decoder = msgspec.msgpack.Decoder(Request)

data = msgspec.msgpack.encode(PutRequest(key="foo", value="bar"))
request = decoder.decode(data)

match request:
    case GetRequest(key):    print(f"Get: {key}")
    case PutRequest(key, value): print(f"Put: {key}={value}")
    case DeleteRequest(key): print(f"Delete: {key}")
```

## Use encode_into for Buffer Reuse

In hot loops, use `Encoder.encode_into()` with a pre-allocated `bytearray` instead of `encode()` to avoid allocating a new `bytes` object per call. Always measure before adopting.

```python
# BAD: new bytes object allocated each iteration
encoder = msgspec.msgpack.Encoder()
for msg in msgs:
    data = encoder.encode(msg)
    socket.sendall(data)

# GOOD: reuse a buffer
encoder = msgspec.msgpack.Encoder()
buffer = bytearray(1024)
for msg in msgs:
    n = encoder.encode_into(msg, buffer)
    socket.sendall(memoryview(buffer)[:n])
```

## NDJSON with encode_into

For line-delimited JSON, use `encode_into()` to avoid the copy from string concatenation:

```python
encoder = msgspec.json.Encoder()
buffer = bytearray(64)
for msg in messages:
    n = encoder.encode_into(msg, buffer)
    file.write(memoryview(buffer)[:n])
    file.write(b"\n")
```

## Length-Prefix Framing

Use `encode_into()` with an offset to efficiently prepend a message length without extra copies:

```python
def send_length_prefixed(socket, msg):
    encoder = msgspec.msgpack.Encoder()
    buffer = bytearray(64)
    n = encoder.encode_into(msg, buffer, 4)   # leave 4 bytes at front
    buffer[:4] = n.to_bytes(4, "big")
    socket.sendall(memoryview(buffer)[:4 + n])

async def prefixed_send(stream, buffer: bytes) -> None:
    stream.write(len(buffer).to_bytes(4, "big"))
    stream.write(buffer)
    await stream.drain()

async def prefixed_recv(stream) -> bytes:
    prefix = await stream.readexactly(4)
    n = int.from_bytes(prefix, "big")
    return await stream.readexactly(n)
```

## Use MessagePack for Internal APIs

`msgspec.msgpack` is more compact and can be more performant than `msgspec.json` for internal service communication.

```python
class Event(msgspec.Struct):
    type: str
    data: dict
    timestamp: float

encoder = msgspec.msgpack.Encoder()
decoder = msgspec.msgpack.Decoder(Event)
packed = encoder.encode(Event(type="login", data={"user_id": 123}, timestamp=1703424000.0))
```

## TOML Configuration Files

Use msgspec for parsing pyproject.toml and other TOML config files with validation:

```python
class BuildSystem(msgspec.Struct, omit_defaults=True, rename="kebab"):
    requires: list[str] = []
    build_backend: str | None = None

class Project(msgspec.Struct, omit_defaults=True, rename="kebab"):
    name: str | None = None
    version: str | None = None
    dependencies: list[str] = []

class PyProject(msgspec.Struct, omit_defaults=True, rename="kebab"):
    build_system: BuildSystem | None = None
    project: Project | None = None
    tool: dict[str, dict[str, Any]] = {}

def load_pyproject(path: str) -> PyProject:
    with open(path, "rb") as f:
        return msgspec.toml.decode(f.read(), type=PyProject)
```

---

## gc=False — Safety Analysis

Setting `gc=False` on a Struct means instances are **never tracked** by Python's garbage collector. This reduces GC pressure (up to 75x less GC pause time, 16 bytes saved per instance). The **only** risk: if a **reference cycle** involves only `gc=False` structs, that cycle will **never be collected** — memory leak.

Reference: [msgspec Structs – Disabling Garbage Collection](https://jcristharif.com/msgspec/structs.html#struct-gc)

### When to use this analysis

- Adding or modifying a class that inherits from `msgspec.Struct`
- Reviewing or refactoring code that defines or uses msgspec structs
- Deciding whether to add or remove `gc=False` on a Struct

### Verified safety constraints

All of the following must hold to use `gc=False` safely.

**1. No reference cycles**

- The struct (and any container it references) must never be part of a reference cycle.
- Multiple variables pointing to the same struct (`x = s; y = x`) are safe — that is not a cycle. A cycle is A → B → … → A.
- Returning a struct from a function is safe. What matters is whether any reference path leads back to the struct.

**2. No mutation that could create cycles**

- Do not mutate struct fields after construction in a way that could introduce a cycle (e.g. set a field to an object that references the struct, or append the struct to its own list/dict).
- Frozen structs (`frozen=True`) prevent field reassignment; `force_setattr` in `__post_init__` is one-time init only — acceptable.
- Assigning scalars (int, str, bool, float, None) to fields is always safe.

**3. Mutable containers (list, dict, set) on the struct**

- If the struct has list/dict/set fields, either:
  - Never mutate those containers after creation and never store in them any object that references the struct, or
  - Do not use `gc=False` (conservative).
- Reading from containers does not create cycles and is always allowed.

**4. Nested structs**

- If a struct holds another Struct (or containers that hold Structs), the same rules apply to the whole reference graph. No cycles, no mutation that could create cycles.

**5. Generic / mixins**

- With `gc=False`, the type must be compatible with `__slots__` (e.g. if using `Generic`, the mixin must define `__slots__ = ()`). See msgspec issue #631 / PR #635.

### Decision tree

```
Should I use gc=False?
│
├── Does your Struct only contain scalar types (int, float, str, bool, bytes)?
│   └── YES → SAFE
│
├── Does your Struct contain lists/dicts and you control what goes in them?
│   └── Will you EVER put the struct itself (or a parent) into those containers?
│       ├── NO → Probably safe, but audit carefully
│       └── YES/MAYBE → Do NOT use gc=False
│
├── Does your Struct reference another Struct of the same type (tree, linked list)?
│   └── YES → Do NOT use gc=False
│
├── Is your Struct part of a bidirectional parent-child relationship?
│   └── YES → Do NOT use gc=False
│
└── When in doubt → Do NOT use gc=False
```

### Examples

```python
# SAFE: only scalar values
class Point(msgspec.Struct, gc=False):
    x: float
    y: float
    z: float

# SAFE: immutable tuple of scalars
class Package(msgspec.Struct, gc=False):
    name: str
    version: str
    depends: tuple[str, ...]
    size: int

# UNSAFE: self-referential — do NOT use gc=False
class TreeNode(msgspec.Struct):  # no gc=False
    value: int
    children: list["TreeNode"]
    parent: "TreeNode | None" = None
```

### Real-world example: decoding large JSON

```python
class Package(msgspec.Struct, gc=False):
    build: str
    build_number: int
    depends: tuple[str, ...]   # tuple, not list — immutable
    md5: str
    name: str
    sha256: str
    version: str
    license: str = ""
    size: int = 0
    timestamp: int = 0

class RepoData(msgspec.Struct, gc=False):
    repodata_version: int
    info: dict
    packages: dict[str, Package]
    removed: tuple[str, ...]

decoder = msgspec.json.Decoder(RepoData)

def load_repo_data(path: str) -> RepoData:
    with open(path, "rb") as f:
        return decoder.decode(f.read())
```

### Checklist: can use gc=False

- [ ] Struct and everything it references can never participate in a reference cycle.
- [ ] No mutation of struct fields after construction that could introduce a cycle (frozen or init-only mutation is ok; scalar assignment is ok).
- [ ] Any list/dict/set fields are never mutated after creation.
- [ ] No storing the struct (or anything that references it) inside its own container fields.
- [ ] If Generic/mixins are used, `__slots__` compatibility is satisfied.

### Checklist: must NOT use gc=False

- [ ] Struct is mutated after creation in a way that could create a cycle.
- [ ] Container fields are mutated after creation and could hold the struct or back-references.
- [ ] Struct is used in a pattern where it's stored in a container that the struct also references.

### Per-struct analysis steps

1. List all fields and their types (scalars vs containers vs nested Structs).
2. Search the codebase for: assignments to this struct's fields, mutations of its container fields (`.append`, `.update`, etc.), and any place the struct instance is stored in a list/dict that might be referenced by the struct.
3. If only scalars or immutable types, or frozen with no container mutation → likely safe.
4. If mutable containers and they're never mutated → likely safe; otherwise → do not use `gc=False`.

### Risky structs: AT-RISK audit pattern

A struct is **risky** for `gc=False` if it has a condition that would normally disallow it (e.g. a mutable dict field) but that condition never arises in practice (e.g. the field is only ever read).

**Auditing a risky struct:**

1. Identify the at-risk condition (e.g. "has `metadata: dict` that could be mutated").
2. Search the codebase for all uses of that struct and of the at-risk field:
   - Field assignment: `obj.field = ...`, `obj.field[key] = ...`, `obj.field.append(...)`, `obj.field.update(...)`
   - Any code path that stores the struct (or something holding it) inside that container.
3. If the audit finds no such mutation or cycle-creating storage, `gc=False` is acceptable — **but add the AT-RISK marker** so future changes are re-audited.

**When audit passes** — set `gc=False` and add:

- A comment above the class stating why gc=False is used and when the audit was done:
  `# gc=False: audit YYYY-MM: <condition> is only read, never mutated.`
- A docstring line signalling that changes must trigger re-audit:
  `AT-RISK (gc=False): Has <brief condition>. Any change that <what would violate safety> must be audited; if so, remove gc=False.`

```python
# gc=False: audit 2026-03: metadata dict is only ever read, never mutated after construction.
class QueryResult(msgspec.Struct, frozen=True, array_like=True, gc=False):
    """Result of a completed inference query.

    AT-RISK (gc=False): Has mutable container field `metadata`. Any change that
    mutates `metadata` after construction or stores this struct in a container
    referenced by this struct must be audited; if so, remove gc=False.
    """
    ...
```

**When touching an AT-RISK struct:**

1. Re-run the audit searches above.
2. If your change mutates the at-risk field(s) or creates a cycle → remove `gc=False` and the AT-RISK comment.
3. If your change does not touch the at-risk field → existing `gc=False` and AT-RISK comment remain; optionally update the audit date.

---

## References

- [msgspec Structs](https://jcristharif.com/msgspec/structs.html)
- [msgspec Performance Tips](https://jcristharif.com/msgspec/perf-tips.html)
- [msgspec Structs – Disabling Garbage Collection](https://jcristharif.com/msgspec/structs.html#struct-gc)
- [msgspec #631 – Generic structs and gc=False](https://github.com/jcrist/msgspec/issues/631)
