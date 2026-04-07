# Project Management Design: Labels, Board, Templates, and CONTRIBUTING.md

**Date:** 2026-04-07
**Author:** Zhihan Jiang (nvzhihanj)
**Status:** Draft

## Context

The mlcommons/endpoints repository has 57 open issues with inconsistent labeling,
no issue templates, a minimal CONTRIBUTING.md, and no active project board. The
project has 3-4 core contributors (NVIDIA) and growing community participation
(Intel, MLCommons, external). The goal is to establish project management
infrastructure that serves the **broader MLCommons community** as the primary
audience — making it easy for external contributors to self-serve, pick up issues,
and understand the project roadmap.

### Research Basis

This design is informed by analysis of label taxonomies and project management
practices from: Kubernetes, PyTorch, vLLM, Ray, SGLang, MLCommons/inference,
and guidance from opensource.guide, GitHub Docs, CNCF, and Linux Foundation.

### Phased Approach

- **Phase 1 (now):** Labels, board, templates, CONTRIBUTING.md, issue migration
- **Phase 2 (when issue volume > 100 or contributors > 10):** Size/effort labels,
  stale bot automation, iteration/sprint fields, disable blank issues

---

## 1. Label Taxonomy (~28 labels)

### Design Principles

- **Prefixed naming** (`type:`, `priority:`, `area:`, `status:`) for filterability
  and visual grouping — inspired by Ray and PyTorch
- **Coarse area labels** (7) grouping related modules — start coarse, split later
- **Severity-gradient colors** for priority — hotter = more urgent
- **Single color family** per label category for visual coherence

### Type Labels

| Label | Color | Description |
|-------|-------|-------------|
| `type: bug` | `#d73a4a` | Something isn't working |
| `type: feature` | `#a2eeef` | New feature or capability |
| `type: enhancement` | `#bfd4f2` | Improvement to existing functionality |
| `type: performance` | `#3ddd26` | Performance regression or improvement |
| `type: documentation` | `#0075ca` | Documentation only |
| `type: question` | `#d876e3` | Usage question or clarification |
| `type: RFC` | `#76fde7` | Request for comments / design proposal |
| `type: chore` | `#ededed` | Maintenance, deps, CI, tooling |

### Priority Labels

| Label | Color | Description |
|-------|-------|-------------|
| `priority: ShowStopper` | `#000000` | Drop everything — critical blocker, all hands on deck |
| `priority: P0` | `#b60205` | Critical — blocks release or users |
| `priority: P1` | `#d93f0b` | High — must address this cycle |
| `priority: P2` | `#fbca04` | Medium — address within quarter |
| `priority: P3` | `#0e8a16` | Low — backlog, nice to have |

### Area Labels

| Label | Color | Description |
|-------|-------|-------------|
| `area: core-engine` | `#c5def5` | Load generator, scheduler, async utils |
| `area: client` | `#c5def5` | Endpoint client, HTTP, transport, ZMQ |
| `area: metrics` | `#c5def5` | Event recorder, metrics reporter, reporting |
| `area: dataset` | `#c5def5` | Dataset manager, formats, predefined datasets |
| `area: config-cli` | `#c5def5` | Config schema, CLI commands, YAML |
| `area: evaluation` | `#c5def5` | Accuracy evaluation, scoring, extractors |
| `area: adapters` | `#c5def5` | OpenAI, SGLang protocol adapters |

### Status Labels

| Label | Color | Description |
|-------|-------|-------------|
| `status: needs-triage` | `#e99695` | New issue, awaiting review |
| `status: needs-info` | `#f9d0c4` | Awaiting more details from reporter |
| `status: blocked` | `#b60205` | Blocked on external dependency or decision |

### Community Labels (keep existing)

| Label | Color | Description |
|-------|-------|-------------|
| `good first issue` | `#7057ff` | Good for newcomers |
| `help wanted` | `#008672` | Extra attention needed |

### Other (keep existing)

| Label | Color | Description |
|-------|-------|-------------|
| `mlcommons` | `#e0703c` | MLCommons ruleset/submission integration |
| `dependencies` | `#9083cd` | Dependency updates |
| `security` | `#b60205` | Security vulnerability or hardening |
| `duplicate` | `#cfd3d7` | Duplicate issue |
| `invalid` | `#e4e669` | Not valid |
| `wontfix` | `#ffffff` | Will not be worked on |

### Labels to Remove

These are replaced by the prefixed equivalents above:

| Old Label | Replaced By |
|-----------|-------------|
| `bug` | `type: bug` |
| `feature` | `type: feature` |
| `enhancement` | `type: enhancement` |
| `documentation` | `type: documentation` |
| `performance` | `type: performance` |
| `question` | `type: question` |
| `P0` | `priority: P0` |
| `P1` | `priority: P1` |
| `P2` | `priority: P2` |
| `ShowStopper` | `priority: ShowStopper` |
| `testing` | `type: chore` (context-dependent) |
| `accuracy` | `area: evaluation` |
| `dataset` | `area: dataset` |
| `Roadmap` | `type: RFC` |
| `blocked` | `status: blocked` |
| `rules` | `mlcommons` |
| `MLCommons` | `mlcommons` (lowercase) |

---

## 2. Project Board #57 Structure

### Status Columns

```
Inbox → Triage → Ready → In Progress → In Review → Done
```

| Column | Purpose | Entry Criteria |
|--------|---------|----------------|
| **Inbox** | New issues land here automatically | Auto-added when issue opened |
| **Triage** | Being evaluated for priority/area/assignee | Someone picked it up to review |
| **Ready** | Triaged, prioritized, ready to work on | Has priority + area labels |
| **In Progress** | Actively being worked on | Assigned, PR may be in flight |
| **In Review** | PR submitted, awaiting review | Linked PR exists |
| **Done** | Merged/resolved/closed | Auto-set when issue closed |

### Custom Fields

| Field | Type | Values |
|-------|------|--------|
| Priority | Single select | ShowStopper, P0, P1, P2, P3 |
| Area | Single select | core-engine, client, metrics, dataset, config-cli, evaluation, adapters, mlcommons |
| Target Release | Single select | v0.5.0, v1.0.0 (add as needed) |

### Views (4)

**1. Kanban (default)**
- Layout: Board
- Columns: Status field
- Group by: Priority (ShowStopper at top → P3 at bottom)
- Filter: status ≠ Done

**2. Priority Table**
- Layout: Table
- Sort: Priority ascending (ShowStopper first), then updated date descending
- Columns: Title, Priority, Area, Status, Assignee, Target Release
- Filter: status ≠ Done

**3. By Assignee**
- Layout: Table
- Group by: Assignee
- Sort: Priority ascending within each group
- Columns: Title, Priority, Area, Status
- Filter: status ≠ Done

**4. Stale Issues**
- Layout: Table
- Sort: Updated date ascending (oldest first)
- Columns: Title, Priority, Area, Status, Assignee, Last Updated
- Filter: status ≠ Done AND last updated more than 30 days ago

### Automations

| Trigger | Action |
|---------|--------|
| Issue added to project | Set status → Inbox |
| Issue closed | Set status → Done |
| PR merged closing issue | Set status → Done |
| Item in Done 14+ days | Auto-archive |

---

## 3. Issue Templates

### Files

- `.github/ISSUE_TEMPLATE/100-bug-report.yml` — Bug Report
- `.github/ISSUE_TEMPLATE/200-feature-request.yml` — Feature Request
- `.github/ISSUE_TEMPLATE/300-performance.yml` — Performance Issue
- `.github/ISSUE_TEMPLATE/400-dataset-integration.yml` — Dataset Integration
- `.github/ISSUE_TEMPLATE/config.yml` — Template chooser config

### 100-bug-report.yml

```yaml
name: Bug Report
description: Report a bug or unexpected behavior
title: "[Bug]: "
labels: ["type: bug", "status: needs-triage"]
body:
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: What happened vs. what you expected
      placeholder: "When I run X, I expected Y but got Z"
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      value: |
        1.
        2.
        3.
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: OS, Python version, package version
      placeholder: "OS: Ubuntu 22.04, Python 3.12, inference-endpoint v0.1.0"
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      render: shell
  - type: checkboxes
    id: checklist
    attributes:
      label: Before submitting
      options:
        - label: I searched existing issues and found no duplicates
          required: true
```

### 200-feature-request.yml

```yaml
name: Feature Request
description: Suggest a new feature or enhancement
title: "[Feature]: "
labels: ["type: feature", "status: needs-triage"]
body:
  - type: textarea
    id: motivation
    attributes:
      label: Motivation
      description: What problem does this solve? Why do you need it?
    validations:
      required: true
  - type: textarea
    id: proposal
    attributes:
      label: Proposed Solution
      description: How should this work? Include API sketches if relevant.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
  - type: textarea
    id: context
    attributes:
      label: Additional Context
```

### 300-performance.yml

```yaml
name: Performance Issue
description: Report a performance regression or improvement opportunity
title: "[Perf]: "
labels: ["type: performance", "status: needs-triage"]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: What performance issue did you observe?
      placeholder: "QPS dropped from X to Y after upgrading to version Z"
    validations:
      required: true
  - type: textarea
    id: benchmark
    attributes:
      label: Benchmark Command
      description: The exact command you ran
      render: shell
    validations:
      required: true
  - type: textarea
    id: results
    attributes:
      label: Results
      description: Expected vs actual numbers (QPS, latency, TTFT, TPOT, etc.)
      placeholder: |
        Expected: ~5000 QPS, p99 latency < 200ms
        Actual: ~2000 QPS, p99 latency 800ms
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Hardware, OS, Python version, endpoint server details
      placeholder: |
        Hardware: 8x A100 80GB
        OS: Ubuntu 22.04
        Python: 3.12
        Server: vLLM 0.6.0, Llama-3-70B
        Workers: 4
    validations:
      required: true
  - type: textarea
    id: profiling
    attributes:
      label: Profiling Data (optional)
      description: Any profiling output, flame graphs, or bottleneck analysis
      render: shell
  - type: checkboxes
    id: checklist
    attributes:
      label: Before submitting
      options:
        - label: I searched existing issues and found no duplicates
          required: true
        - label: I ran with default settings before tuning
          required: false
```

### 400-dataset-integration.yml

```yaml
name: Dataset Integration
description: Request support for a new dataset or evaluation benchmark
title: "[Dataset]: "
labels: ["type: feature", "area: dataset", "status: needs-triage"]
body:
  - type: textarea
    id: dataset
    attributes:
      label: Dataset Information
      description: Name, URL, and brief description
      placeholder: |
        Name: MATH-500
        URL: https://huggingface.co/datasets/...
        Description: 500 competition math problems for testing reasoning
    validations:
      required: true
  - type: dropdown
    id: format
    attributes:
      label: Dataset Format
      options:
        - JSONL
        - HuggingFace Dataset
        - CSV
        - JSON
        - Parquet
        - Other
    validations:
      required: true
  - type: textarea
    id: evaluation
    attributes:
      label: Evaluation Method
      description: How should responses be scored?
      placeholder: "Exact match after extracting boxed answer, or pass@1 for code"
    validations:
      required: true
  - type: textarea
    id: samples
    attributes:
      label: Scale
      description: Number of samples, expected prompt/response lengths
      placeholder: "500 samples, avg prompt ~200 tokens, avg response ~500 tokens"
  - type: textarea
    id: context
    attributes:
      label: Additional Context
      description: Related benchmarks, papers, or prior art
```

### config.yml

```yaml
blank_issues_enabled: true
contact_links:
  - name: Questions & Discussion
    url: https://github.com/mlcommons/endpoints/discussions
    about: Ask questions and discuss ideas before filing an issue
```

---

## 4. CONTRIBUTING.md

Replace the existing minimal CONTRIBUTING.md with an expanded version (~250 lines)
covering:

1. **Ways to Contribute** — links to all 4 issue templates, plus docs, PR reviews,
   `good first issue` and `help wanted` labels
2. **Development Setup** — prerequisites, fork/clone, venv, `pip install -e ".[dev,test]"`,
   pre-commit install, local echo server testing
3. **Code Style and Conventions** — ruff, mypy, line length 88, double quotes,
   conventional commits, license headers, serialization conventions
   (msgspec vs pydantic), performance-sensitive code guidelines
4. **Testing** — pytest commands, markers (`unit`, `integration`, `slow`,
   `performance`), `@pytest.mark.asyncio(mode="strict")`, >90% coverage target,
   use real fixtures over mocks
5. **Submitting Changes** — branch naming (`feat/`, `fix/`, `docs/`), PR template,
   CI checks, review expectations (2-3 business days), review criteria
6. **Issue Guidelines** — search first, use templates, issue lifecycle
   (Inbox → Triage → Ready → In Progress → In Review → Done), priority levels table
7. **MLCommons CLA** — existing CLA requirements preserved

---

## 5. Issue Migration Plan

### Duplicate Resolution

Close duplicates with a comment explaining the closure and linking to the primary
issue. Copy any unique context from the duplicate into a comment on the primary
issue so no information is lost.

| Close | Primary | Reason |
|-------|---------|--------|
| #205 "fully async benchmark" | #255 "Make Loadgen Async" | Same goal, #255 is cleaner |
| #170 "warmup with random dataset" | #86 "Warmup runs" | Subset of #86 |
| #226 "Initial multi-turn enabling" | #232 "multi-turn implementation" | Same feature |
| #29 "submission checker for 6.0" | #79 "submission checker compat mode" | #29 is version-specific, superseded |
| #207 "speedup tokenizer report" | #208 "optimize report generation" | #207 is a specific approach to #208 |
| #83 "Q1 Roadmap" | #223 "Phase 2 Roadmap" | Superseded |

**Evaluation:** #73 "random dataset support" — keep if random dataset has value
beyond warmup use case; otherwise close as duplicate of #86.

### Label Reassignment

All 57 open issues are reassigned from old labels to the new prefixed taxonomy.
Full mapping follows, organized by priority tier.

#### ShowStopper

| # | Title | Labels |
|---|-------|--------|
| 84 | Pareto clarification | `priority: ShowStopper`, `area: config-cli`, `mlcommons` |
| 8 | Parity with MLPerf LoadGen | `priority: ShowStopper`, `type: performance`, `area: core-engine` |
| 4 | Accuracy evaluation for LLMs | `priority: ShowStopper`, `type: feature`, `area: evaluation` |

#### P0

| # | Title | Labels |
|---|-------|--------|
| 86 | Warmup runs | `priority: P0`, `type: feature`, `area: core-engine` |
| 183 | Pub/Sub event recorder | `priority: P0`, `type: feature`, `area: metrics` |
| 138 | CI stress test upper bound | `priority: P0`, `type: chore`, `area: core-engine` |
| 6 | Final report structure | `priority: P0`, `type: feature`, `area: metrics` |
| 5 | Submission ruleset + config | `priority: P0`, `type: feature`, `area: config-cli`, `mlcommons` |

#### P1

| # | Title | Labels |
|---|-------|--------|
| 9 | Roofline analysis | `priority: P1`, `type: performance`, `area: core-engine` |
| 255 | Make Loadgen Async | `priority: P1`, `type: feature`, `area: core-engine` |
| 269 | Low concurrency timeouts | `priority: P1`, `type: bug`, `area: client` |
| 237 | CLI fix --load-pattern + --target-qps | `priority: P1`, `type: bug`, `area: config-cli` |
| 219 | target_qps hardcoded in Offline | `priority: P1`, `type: bug`, `area: config-cli` |
| 221 | RuntimeSettings non-reproducible | `priority: P1`, `type: bug`, `area: config-cli` |
| 202 | max_throughput connection timeouts | `priority: P1`, `type: bug`, `area: client` |
| 199 | Perf discrepancy submission vs perf config | `priority: P1`, `type: bug`, `area: config-cli` |
| 217 | BURST and STEP load patterns | `priority: P1`, `type: feature`, `area: core-engine` |
| 222 | KVStore/ServiceLauncher lack tests | `priority: P1`, `type: chore`, `area: core-engine` |
| 220 | SGLang adapter tests skipped | `priority: P1`, `type: chore`, `area: adapters` |
| 182 | Text vs token perf on TRTLLM | `priority: P1`, `type: performance`, `area: metrics` |
| 179 | Humanity's Last Exam | `priority: P1`, `type: feature`, `area: evaluation`, `area: dataset` |
| 178 | Healthbench integration | `priority: P1`, `type: feature`, `area: evaluation`, `area: dataset` |
| 177 | MATH500 dataset | `priority: P1`, `type: feature`, `area: evaluation`, `area: dataset` |
| 176 | MMLU/MMLU-Pro | `priority: P1`, `type: feature`, `area: evaluation`, `area: dataset` |
| 173 | Investigate mlcr failures | `priority: P1`, `type: bug`, `mlcommons` |
| 113 | DeepSeek | `priority: P1`, `type: feature` |
| 210 | Wan2.2-T2V support | `priority: P1`, `type: feature` |
| 10 | System bottleneck tests | `priority: P1`, `type: performance`, `area: core-engine` |
| 7 | Runtime visualization | `priority: P1`, `type: feature`, `area: metrics` |

#### P2

| # | Title | Labels |
|---|-------|--------|
| 268 | Phase 2 model selection | `priority: P2`, `type: feature` |
| 254 | Handling failed requests | `priority: P2`, `type: feature`, `area: client` |
| 232 | Multi-turn implementation | `priority: P2`, `type: feature`, `area: dataset` |
| 224 | Multiple perf configs | `priority: P2`, `type: feature`, `area: config-cli` |
| 208 | Optimize report generation | `priority: P2`, `type: performance`, `area: metrics` |
| 158 | SGLang adapter + OpenAI compat | `priority: P2`, `type: feature`, `area: adapters` |
| 125 | Multi-concurrency scans | `priority: P2`, `type: feature`, `area: core-engine` |
| 115 | Clarify default metric | `priority: P2`, `type: enhancement`, `area: config-cli` |
| 79 | Submission checker compat mode | `priority: P2`, `type: feature`, `mlcommons` |
| 73 | Random dataset support | `priority: P2`, `type: feature`, `area: dataset` |
| 68 | Official model name mapping | `priority: P2`, `type: feature`, `area: config-cli`, `mlcommons` |
| 58 | Config-template mapping | `priority: P2`, `type: feature`, `area: config-cli`, `mlcommons` |
| 213 | PostGres dup element | `priority: P2`, `type: bug`, `mlcommons` |
| 133 | llama.cpp incompatibility | `priority: P2`, `type: bug`, `area: client` |
| 174 | Better error logging mlcr | `priority: P2`, `type: enhancement`, `mlcommons` |
| 229 | Endpoints test environment | `priority: P2`, `type: chore` |
| 228 | Endpoints Vision document | `priority: P2`, `type: documentation` |
| 227 | DB and Object Store elements | `priority: P2`, `type: feature` |
| 212 | UBI Storage layer | `priority: P2`, `type: feature` |

#### P3

| # | Title | Labels |
|---|-------|--------|
| 99 | Local mode errors | `priority: P3`, `type: bug`, `good first issue` |
| 50 | LlaMa3-405b support | `priority: P3`, `type: feature` |
| 204 | Documentation cleanup | `priority: P3`, `type: documentation` |
| 190 | Skills, design docs, tooling | `priority: P3`, `type: chore` |
| 181 | Sweep qwen scripts | `priority: P3`, `type: feature` |

#### Other (no priority)

| # | Title | Labels |
|---|-------|--------|
| 223 | Phase 2 Roadmap | `type: RFC` |
| 267 | Bump transformers | `type: chore`, `dependencies`, `security` |

### Q2 Board Population

**Add to board #57 (~40 issues):** All ShowStopper, P0, P1, and P2 issues.
Initial status: **Triage** (existing issues need priority confirmation from team).

**Not on Q2 board (~5 issues):** P3 issues (#99, #50, #204, #190, #181) and
dependabot (#267).

### Milestones

Create milestones as releases are planned:
- `v0.5.0` — first milestone, assign issues as release scope is defined
- `v1.0.0` — future

---

## 6. Phase 2 (Future)

Trigger when issue volume > 100 or contributors > 10:

- Add `size: S`, `size: M`, `size: L`, `size: XL` effort labels
- Disable blank issues in `config.yml`
- Add stale bot (apply `status: stale` after 90 days, close after 30 more)
- Add iteration/sprint fields to board if team adopts time-boxed cycles
- Split coarse area labels if any accumulates > 20 issues

---

## 7. Migration Procedure

Order of operations for the migration:

1. **Create new labels** — all `type:`, `priority:`, `area:`, `status:` labels
2. **Relabel existing issues** — apply new labels per the mapping above
3. **Remove old labels from issues** — strip legacy labels
4. **Close duplicates** — comment with explanation + link to primary, copy unique
   context to primary issue
5. **Delete old labels** — remove legacy labels from the repository
6. **Add issues to board #57** — all ShowStopper through P2
7. **Set board status** — all migrated issues start in Triage
8. **Configure board automations** — auto-add, auto-done, auto-archive
9. **Create issue templates** — add all 4 YAML templates + config.yml
10. **Update CONTRIBUTING.md** — replace with expanded version
11. **Commit and push** — templates + CONTRIBUTING.md in a single PR
