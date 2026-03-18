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

## Verified safety constraints

All must hold for gc=False to be safe.

### 1. No reference cycles

- The struct (and any container it references) must never be part of a reference cycle.
- **Multiple variables** pointing to the same struct (`x = s; y = x`) are **safe** — that is not a cycle.
- **Returning** a struct from a function is **safe**. What matters is whether any reference path leads back to the struct (e.g. struct's list contains the struct or something that holds the struct).

### 2. No mutation that could create cycles

- **Do not mutate** struct fields after construction in a way that could introduce a cycle.
- **Frozen structs** (`frozen=True`) prevent field reassignment; `force_setattr` in `__post_init__` is one-time init only, so that's acceptable.
- Assigning **scalars** (int, str, bool, float, None) to fields is safe — they cannot form cycles.

### 3. Mutable containers (list, dict, set) on the struct

- If the struct has list/dict/set fields, either:
  - **Never mutate** those containers after creation, and never store in them any object that references the struct, or
  - Do not use `gc=False` (conservative).
- **Reading** from containers does not create cycles and is allowed.

### 4. Nested structs

- If a struct holds another Struct, the same rules apply to the whole reference graph: no cycles, no mutation that could create cycles.

### 5. Generic / mixins

- With `gc=False`, the type must be compatible with `__slots__` (e.g. if using `Generic`, the mixin must define `__slots__ = ()`).

## Quick per-struct analysis steps

1. List all fields and their types (scalars vs containers vs nested Structs).
2. Search the codebase for: assignments to this struct's fields, mutations of its container fields (`.append`, `.update`, etc.), and any place the struct instance is stored.
3. If only scalars or immutable types, or frozen with no container mutation -> likely safe for gc=False.
4. If mutable containers and they're never mutated (and never made to reference the struct) -> likely safe; otherwise -> do not use gc=False.

## Risky structs: audit and at-risk comment

A struct is **risky** for gc=False if it has a condition that would normally disallow gc=False (e.g. mutable list/dict/set fields), but that condition might never arise in practice.

### When audit passes

- Set `gc=False` on the struct.
- Add an **at-risk comment** above the class:

  `# gc=False: audit YYYY-MM: <condition> is only read, never mutated.`

- Add a docstring note:

  `AT-RISK (gc=False): Has <brief condition>. Any change that <what would violate safety> must be audited; if so, remove gc=False.`

### When touching an at-risk struct

1. Re-run the audit for that struct.
2. If your change mutates the at-risk field(s) or creates a cycle, **remove** `gc=False`.
3. If your change does not touch the at-risk field, the existing gc=False remains; you may update the audit date.
