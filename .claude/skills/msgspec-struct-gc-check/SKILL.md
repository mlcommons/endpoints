---
name: msgspec-struct-gc-check
description: Check whether msgspec.Struct types can safely use gc=False. Use when adding or changing msgspec.Struct definitions, or when reviewing code that uses msgspec structs.
allowed-tools: Read, Grep, Glob
---

# msgspec.Struct gc=False Safety Check

## When to use this skill

- Adding or modifying a class that inherits from `msgspec.Struct`
- Reviewing or refactoring code that defines or uses msgspec structs
- Deciding whether to add or remove `gc=False` on a Struct

## Why gc=False matters

Setting `gc=False` on a Struct means instances are **never tracked** by Python's garbage collector. This reduces GC pressure and can improve performance when many structs are allocated. The **only** risk: if a **reference cycle** involves only gc=False structs (or objects not tracked by GC), that cycle will **never be collected** (memory leak).

Reference: [msgspec Structs – Disabling Garbage Collection](https://jcristharif.com/msgspec/structs.html#struct-gc).

## Verified safety constraints

Use these constraints to decide if a Struct can use `gc=False`. All must hold.

### 1. No reference cycles

- The struct (and any container it references) must never be part of a reference cycle.
- **Multiple variables** pointing to the same struct (`x = s; y = x`) are **safe** — that is not a cycle. A cycle is A → B → … → A.
- **Returning** a struct from a function is **safe**. What matters is whether any reference path leads back to the struct (e.g. struct's list contains the struct or something that holds the struct).

### 2. No mutation that could create cycles

- **Do not mutate** struct fields after construction in a way that could introduce a cycle (e.g. set a field to an object that references the struct, or append the struct to its own list/dict).
- **Frozen structs** (`frozen=True`) prevent field reassignment; `force_setattr` in `__post_init__` is one-time init only, so that's acceptable.
- Assigning **scalars** (int, str, bool, float, None) to fields is safe — they cannot form cycles.

### 3. Mutable containers (list, dict, set) on the struct

- If the struct has list/dict/set fields, either:
  - **Never mutate** those containers after creation (no `.append`, `.update`, `[...] = ...`, etc.), and never store in them any object that references the struct, or
  - Do not use `gc=False` (conservative).
- **Reading** from containers (e.g. `x = struct.foobars[i]`) does not create cycles and is allowed.

### 4. Nested structs

- If a struct holds another Struct (or holds containers that hold Structs), the same rules apply to the whole reference graph: no cycles, no mutation that could create cycles. If any nested Struct uses `gc=False`, the whole graph must still be cycle-free.

### 5. Generic / mixins

- With `gc=False`, the type must be compatible with `__slots__` (e.g. if using `Generic`, the mixin must define `__slots__ = ()`). See msgspec issue #631 / PR #635.

## Checklist for "can use gc=False"

- [ ] Struct and everything it references can never participate in a reference cycle.
- [ ] No mutation of struct fields after construction that could introduce a cycle (frozen or init-only mutation is ok; scalar assignment is ok).
- [ ] Any list/dict/set fields are never mutated after creation, or we do not use gc=False.
- [ ] No storing the struct (or anything that references it) inside its own container fields.
- [ ] If Generic/mixins are used, `__slots__` compatibility is satisfied.

## Checklist for "must NOT use gc=False"

- [ ] Struct is mutated after creation in a way that could create a cycle (e.g. appending self to a list field).
- [ ] Container fields are mutated after creation and could hold the struct or back-references.
- [ ] Struct is used in a pattern where it's stored in a container that the struct (or its fields) also references.

## Quick per-struct analysis steps

1. List all fields and their types (scalars vs containers vs nested Structs).
2. Search the codebase for: assignments to this struct's fields, mutations of its container fields (`.append`, `.update`, etc.), and any place the struct instance is stored (e.g. in a list/dict that might be referenced by the struct).
3. If only scalars or immutable types, or frozen with no container mutation → likely safe for gc=False.
4. If mutable containers and they're never mutated (and never made to reference the struct) → likely safe; otherwise → do not use gc=False.

## Risky structs: audit and at-risk comment

A struct is **risky** for gc=False if it has a condition that would normally disallow gc=False (e.g. mutable list/dict/set fields), but that condition might never arise in practice (e.g. the field is only ever read, never mutated after construction).

### Auditing a risky struct

1. Identify the at-risk condition (e.g. "has `metadata: dict` that could be mutated").
2. Search the codebase for all uses of that struct and of the at-risk field:
   - Any assignment to the field: `obj.field = ...`, `obj.field[key] = ...`, `obj.field.append(...)`, `obj.field.update(...)`, etc.
   - Any code path that could store the struct (or something holding it) inside that container.
3. If the audit finds **no** such mutation or cycle-creating storage, the condition never arises and gc=False is acceptable **provided** you add the at-risk marker so future changes are re-audited.

### When audit passes

- Set `gc=False` on the struct.
- Add an **at-risk comment** and docstring note:

  - **Above the class**: a short comment stating why gc=False is used despite the at-risk condition, and when the audit was done (e.g. `# gc=False: audit YYYY-MM: <condition> is only read, never mutated.`).
  - **In the docstring**: a line that signals to future readers and to this skill that changes touching this struct must be re-audited. Use this format:

    `AT-RISK (gc=False): Has <brief condition>. Any change that <what would violate safety> must be audited; if so, remove gc=False.`

- Example (for a struct with a `metadata` dict that is only ever read):

  ```python
  # gc=False: audit 2026-03: metadata dict is only ever read, never mutated after construction.
  class QueryResult(msgspec.Struct, ..., gc=False):
      """Result of a completed inference query.

      AT-RISK (gc=False): Has mutable container field `metadata`. Any change that
      mutates `metadata` after construction or stores this struct in a container
      referenced by this struct must be audited; if so, remove gc=False.
      ...
  ```

### When touching an at-risk struct

If you are adding or changing code that uses a struct marked AT-RISK (gc=False):

1. Re-run the audit for that struct (searches above).
2. If your change mutates the at-risk field(s) or creates a cycle (e.g. stores the struct in its own container), **remove** `gc=False` from the struct and remove the at-risk comment/docstring line.
3. If your change does not touch the at-risk field or create cycles, the existing gc=False and at-risk comment remain; you may add a short note in the at-risk comment if the audit was re-checked (e.g. update the audit date).

## References

- [msgspec Structs – Disabling Garbage Collection](https://jcristharif.com/msgspec/structs.html#struct-gc)
- [msgspec Performance Tips – Use gc=False](https://jcristharif.com/msgspec/perf-tips.html#use-gc-false)
- [msgspec #631 – Generic structs and gc=False](https://github.com/jcrist/msgspec/issues/631)
