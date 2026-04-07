# Project Management Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up labels, project board, issue templates, CONTRIBUTING.md, and migrate all 57 open issues for the mlcommons/endpoints GitHub repository.

**Architecture:** All GitHub API interactions use `curl` with auth token (the `gh` CLI has TLS certificate issues in this environment). Board configuration uses the GitHub GraphQL API for Projects V2. File changes (templates, CONTRIBUTING.md) are committed locally and pushed as a PR.

**Tech Stack:** GitHub REST API, GitHub GraphQL API, curl, bash, git

**IMPORTANT — API access pattern:** The `gh` CLI cannot make API calls due to TLS errors. Every API call must use this pattern:
```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" "https://api.github.com/..."
```
For GraphQL:
```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{"query":"..."}'
```

**IMPORTANT — Label names with colons:** GitHub label names containing spaces and colons must be URL-encoded in REST API paths. For example, `type: bug` becomes `type%3A%20bug` in URLs. When creating labels via POST body (JSON), use the literal name.

---

## File Structure

No new source code files. Changes are:

- **Create:** `.github/ISSUE_TEMPLATE/100-bug-report.yml`
- **Create:** `.github/ISSUE_TEMPLATE/200-feature-request.yml`
- **Create:** `.github/ISSUE_TEMPLATE/300-performance.yml`
- **Create:** `.github/ISSUE_TEMPLATE/400-dataset-integration.yml`
- **Create:** `.github/ISSUE_TEMPLATE/config.yml`
- **Modify:** `CONTRIBUTING.md` (full rewrite)

All other changes are GitHub API operations (labels, board, issues) — no local files.

---

### Task 1: Create New Labels

Create all 23 new labels on the repository via the REST API. Existing labels that are being kept (`good first issue`, `help wanted`, `dependencies`, `security`, `duplicate`, `invalid`, `wontfix`) are untouched. The `mlcommons` label needs to be created fresh (the old `MLCommons` with capital M will be removed later).

**Files:** None (API only)

- [ ] **Step 1: Create all type labels**

Run this script. It creates 8 type labels:

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

for label_json in \
  '{"name":"type: bug","color":"d73a4a","description":"Something isn'\''t working"}' \
  '{"name":"type: feature","color":"a2eeef","description":"New feature or capability"}' \
  '{"name":"type: enhancement","color":"bfd4f2","description":"Improvement to existing functionality"}' \
  '{"name":"type: performance","color":"3ddd26","description":"Performance regression or improvement"}' \
  '{"name":"type: documentation","color":"0075ca","description":"Documentation only"}' \
  '{"name":"type: question","color":"d876e3","description":"Usage question or clarification"}' \
  '{"name":"type: RFC","color":"76fde7","description":"Request for comments / design proposal"}' \
  '{"name":"type: chore","color":"ededed","description":"Maintenance, deps, CI, tooling"}'; do
  echo "Creating: $(echo "$label_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["name"])')"
  curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/labels" \
    -d "$label_json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  -> {d.get(\"name\", d.get(\"message\", \"error\"))}")'
done
```

Expected: 8 lines showing each label name created successfully.

- [ ] **Step 2: Create all priority labels**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

for label_json in \
  '{"name":"priority: ShowStopper","color":"000000","description":"Drop everything — critical blocker, all hands on deck"}' \
  '{"name":"priority: P0","color":"b60205","description":"Critical — blocks release or users"}' \
  '{"name":"priority: P1","color":"d93f0b","description":"High — must address this cycle"}' \
  '{"name":"priority: P2","color":"fbca04","description":"Medium — address within quarter"}' \
  '{"name":"priority: P3","color":"0e8a16","description":"Low — backlog, nice to have"}'; do
  echo "Creating: $(echo "$label_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["name"])')"
  curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/labels" \
    -d "$label_json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  -> {d.get(\"name\", d.get(\"message\", \"error\"))}")'
done
```

Expected: 5 labels created.

- [ ] **Step 3: Create all area labels**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

for label_json in \
  '{"name":"area: core-engine","color":"c5def5","description":"Load generator, scheduler, async utils"}' \
  '{"name":"area: client","color":"c5def5","description":"Endpoint client, HTTP, transport, ZMQ"}' \
  '{"name":"area: metrics","color":"c5def5","description":"Event recorder, metrics reporter, reporting"}' \
  '{"name":"area: dataset","color":"c5def5","description":"Dataset manager, formats, predefined datasets"}' \
  '{"name":"area: config-cli","color":"c5def5","description":"Config schema, CLI commands, YAML"}' \
  '{"name":"area: evaluation","color":"c5def5","description":"Accuracy evaluation, scoring, extractors"}' \
  '{"name":"area: adapters","color":"c5def5","description":"OpenAI, SGLang protocol adapters"}'; do
  echo "Creating: $(echo "$label_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["name"])')"
  curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/labels" \
    -d "$label_json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  -> {d.get(\"name\", d.get(\"message\", \"error\"))}")'
done
```

Expected: 7 labels created.

- [ ] **Step 4: Create status labels and mlcommons label**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

for label_json in \
  '{"name":"status: needs-triage","color":"e99695","description":"New issue, awaiting review"}' \
  '{"name":"status: needs-info","color":"f9d0c4","description":"Awaiting more details from reporter"}' \
  '{"name":"status: blocked","color":"b60205","description":"Blocked on external dependency or decision"}' \
  '{"name":"mlcommons","color":"e0703c","description":"MLCommons ruleset/submission integration"}'; do
  echo "Creating: $(echo "$label_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["name"])')"
  curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/labels" \
    -d "$label_json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"  -> {d.get(\"name\", d.get(\"message\", \"error\"))}")'
done
```

Expected: 4 labels created (mlcommons may say "already_exists" if the old `MLCommons` case-insensitively matches — if so, update it in a later step).

- [ ] **Step 5: Verify all new labels exist**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" \
  "https://api.github.com/repos/mlcommons/endpoints/labels?per_page=100" | \
  python3 -c "
import sys, json
labels = json.load(sys.stdin)
names = sorted([l['name'] for l in labels])
print(f'Total labels: {len(names)}')
for n in names:
    print(f'  {n}')
"
```

Expected: All new `type:`, `priority:`, `area:`, `status:` labels present alongside existing labels.

---

### Task 2: Relabel All Open Issues

Apply new labels and remove old labels for every open issue, following the spec's mapping exactly. This is done in batches by priority tier.

**Files:** None (API only)

**IMPORTANT:** The GitHub `PUT /repos/{owner}/{repo}/issues/{number}/labels` endpoint **replaces** all labels on an issue. So each call must include the complete set of new labels for that issue.

- [ ] **Step 1: Relabel ShowStopper issues**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# #84 - Pareto clarification
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/84/labels" \
  -d '{"labels":["priority: ShowStopper","area: config-cli","mlcommons"]}' | python3 -c 'import sys,json; print(f"#84: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #8 - Parity with MLPerf LoadGen
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/8/labels" \
  -d '{"labels":["priority: ShowStopper","type: performance","area: core-engine"]}' | python3 -c 'import sys,json; print(f"#8: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #4 - Accuracy evaluation for LLMs
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/4/labels" \
  -d '{"labels":["priority: ShowStopper","type: feature","area: evaluation"]}' | python3 -c 'import sys,json; print(f"#4: {[l[\"name\"] for l in json.load(sys.stdin)]}")'
```

Expected: Each issue prints its new label set.

- [ ] **Step 2: Relabel P0 issues**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# #86 - Warmup runs
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/86/labels" \
  -d '{"labels":["priority: P0","type: feature","area: core-engine"]}' | python3 -c 'import sys,json; print(f"#86: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #232 - Multi-turn implementation
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/232/labels" \
  -d '{"labels":["priority: P0","type: feature","area: dataset"]}' | python3 -c 'import sys,json; print(f"#232: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #183 - Pub/Sub event recorder
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/183/labels" \
  -d '{"labels":["priority: P0","type: feature","area: metrics"]}' | python3 -c 'import sys,json; print(f"#183: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #138 - CI stress test upper bound
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/138/labels" \
  -d '{"labels":["priority: P0","type: chore","area: core-engine"]}' | python3 -c 'import sys,json; print(f"#138: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #6 - Final report structure
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/6/labels" \
  -d '{"labels":["priority: P0","type: feature","area: metrics"]}' | python3 -c 'import sys,json; print(f"#6: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# #5 - Submission ruleset + config
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/5/labels" \
  -d '{"labels":["priority: P0","type: feature","area: config-cli","mlcommons"]}' | python3 -c 'import sys,json; print(f"#5: {[l[\"name\"] for l in json.load(sys.stdin)]}")'
```

Expected: 6 issues relabeled.

- [ ] **Step 3: Relabel P1 issues**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

declare -A P1_LABELS
P1_LABELS[9]='["priority: P1","type: performance","area: core-engine"]'
P1_LABELS[255]='["priority: P1","type: feature","area: core-engine"]'
P1_LABELS[269]='["priority: P1","type: bug","area: client"]'
P1_LABELS[237]='["priority: P1","type: bug","area: config-cli"]'
P1_LABELS[219]='["priority: P1","type: bug","area: config-cli"]'
P1_LABELS[221]='["priority: P1","type: bug","area: config-cli"]'
P1_LABELS[202]='["priority: P1","type: bug","area: client"]'
P1_LABELS[199]='["priority: P1","type: bug","area: config-cli"]'
P1_LABELS[222]='["priority: P1","type: chore","area: core-engine"]'
P1_LABELS[220]='["priority: P1","type: chore","area: adapters"]'
P1_LABELS[182]='["priority: P1","type: performance","area: metrics"]'
P1_LABELS[177]='["priority: P1","type: feature","area: evaluation","area: dataset"]'
P1_LABELS[176]='["priority: P1","type: feature","area: evaluation","area: dataset"]'
P1_LABELS[113]='["priority: P1","type: feature"]'
P1_LABELS[210]='["priority: P1","type: feature"]'
P1_LABELS[268]='["priority: P1","type: feature"]'
P1_LABELS[10]='["priority: P1","type: performance","area: core-engine"]'
P1_LABELS[7]='["priority: P1","type: feature","area: metrics"]'

for issue in "${!P1_LABELS[@]}"; do
  curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/issues/$issue/labels" \
    -d "{\"labels\":${P1_LABELS[$issue]}}" | python3 -c "import sys,json; print(f'#$issue: {[l[\"name\"] for l in json.load(sys.stdin)]}')"
done
```

Expected: 18 issues relabeled.

- [ ] **Step 4: Relabel P2 issues**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

declare -A P2_LABELS
P2_LABELS[254]='["priority: P2","type: feature","area: client"]'
P2_LABELS[217]='["priority: P2","type: feature","area: core-engine"]'
P2_LABELS[179]='["priority: P2","type: feature","area: evaluation","area: dataset"]'
P2_LABELS[178]='["priority: P2","type: feature","area: evaluation","area: dataset"]'
P2_LABELS[173]='["priority: P2","type: bug","mlcommons"]'
P2_LABELS[224]='["priority: P2","type: feature","area: config-cli"]'
P2_LABELS[208]='["priority: P2","type: performance","area: metrics"]'
P2_LABELS[158]='["priority: P2","type: feature","area: adapters"]'
P2_LABELS[125]='["priority: P2","type: feature","area: core-engine"]'
P2_LABELS[115]='["priority: P2","type: enhancement","area: config-cli"]'
P2_LABELS[79]='["priority: P2","type: feature","mlcommons"]'
P2_LABELS[73]='["priority: P2","type: feature","area: dataset"]'
P2_LABELS[68]='["priority: P2","type: feature","area: config-cli","mlcommons"]'
P2_LABELS[58]='["priority: P2","type: feature","area: config-cli","mlcommons"]'
P2_LABELS[213]='["priority: P2","type: bug","mlcommons"]'
P2_LABELS[133]='["priority: P2","type: bug","area: client"]'
P2_LABELS[174]='["priority: P2","type: enhancement","mlcommons"]'
P2_LABELS[229]='["priority: P2","type: chore"]'
P2_LABELS[228]='["priority: P2","type: documentation"]'
P2_LABELS[227]='["priority: P2","type: feature"]'
P2_LABELS[212]='["priority: P2","type: feature"]'

for issue in "${!P2_LABELS[@]}"; do
  curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/issues/$issue/labels" \
    -d "{\"labels\":${P2_LABELS[$issue]}}" | python3 -c "import sys,json; print(f'#$issue: {[l[\"name\"] for l in json.load(sys.stdin)]}')"
done
```

Expected: 21 issues relabeled.

- [ ] **Step 5: Relabel P3 and other issues**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# P3 issues
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/99/labels" \
  -d '{"labels":["priority: P3","type: bug","good first issue"]}' | python3 -c 'import sys,json; print(f"#99: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/50/labels" \
  -d '{"labels":["priority: P3","type: feature"]}' | python3 -c 'import sys,json; print(f"#50: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/204/labels" \
  -d '{"labels":["priority: P3","type: documentation"]}' | python3 -c 'import sys,json; print(f"#204: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/190/labels" \
  -d '{"labels":["priority: P3","type: chore"]}' | python3 -c 'import sys,json; print(f"#190: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/181/labels" \
  -d '{"labels":["priority: P3","type: feature"]}' | python3 -c 'import sys,json; print(f"#181: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

# Other (no priority)
curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/223/labels" \
  -d '{"labels":["type: RFC"]}' | python3 -c 'import sys,json; print(f"#223: {[l[\"name\"] for l in json.load(sys.stdin)]}")'

curl -s -X PUT -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/267/labels" \
  -d '{"labels":["type: chore","dependencies","security"]}' | python3 -c 'import sys,json; print(f"#267: {[l[\"name\"] for l in json.load(sys.stdin)]}")'
```

Expected: 7 issues relabeled.

- [ ] **Step 6: Verify relabeling — spot check 5 issues**

```bash
TOKEN=$(gh auth token 2>&1)
for issue in 84 232 269 208 99; do
  curl -s -H "Authorization: token $TOKEN" \
    "https://api.github.com/repos/mlcommons/endpoints/issues/$issue" | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(f'#{d[\"number\"]} {d[\"title\"]}: {[l[\"name\"] for l in d[\"labels\"]]}')"
done
```

Expected: Each issue shows only its new prefixed labels.

---

### Task 3: Close Duplicate Issues

For each duplicate, first read its body to preserve unique context, then comment on the primary issue with that context, then close the duplicate with an explanation.

**Files:** None (API only)

- [ ] **Step 1: Close #205 as duplicate of #255 (async benchmark)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# Get #205 body for context preservation
BODY_205=$(curl -s -H "Authorization: token $TOKEN" "https://api.github.com/repos/$REPO/issues/205" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("body","") or "(no body)")')

# Comment on primary #255 with context from #205
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/255/comments" \
  -d "$(python3 -c "
import json
body = '''Context preserved from duplicate #205 (fully async benchmark):

$BODY_205'''
print(json.dumps({'body': body}))
")" | python3 -c 'import sys,json; print(f"Commented on #255: {json.load(sys.stdin).get(\"id\",\"error\")}")'

# Comment on #205 explaining closure
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/205/comments" \
  -d '{"body":"Closing as duplicate of #255 (Make Loadgen Async). Both issues target the same goal of making the benchmark fully async. Unique context from this issue has been copied to #255."}' | python3 -c 'import sys,json; print(f"Commented on #205: {json.load(sys.stdin).get(\"id\",\"error\")}")'

# Close #205
curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/205" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c 'import sys,json; print(f"#205 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

Expected: #205 closed, context preserved on #255.

- [ ] **Step 2: Close #170 as duplicate of #86 (warmup)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

BODY_170=$(curl -s -H "Authorization: token $TOKEN" "https://api.github.com/repos/$REPO/issues/170" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("body","") or "(no body)")')

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/86/comments" \
  -d "$(python3 -c "
import json
body = '''Context preserved from duplicate #170 (warmup with random dataset):

$BODY_170'''
print(json.dumps({'body': body}))
")" | python3 -c 'import sys,json; print(f"Commented on #86: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/170/comments" \
  -d '{"body":"Closing as duplicate of #86 (Warmup runs). This issue describes a specific warmup implementation approach (random dataset) which is a subset of #86. Unique context has been copied to #86."}' | python3 -c 'import sys,json; print(f"Commented on #170: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/170" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c 'import sys,json; print(f"#170 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

- [ ] **Step 3: Close #226 as duplicate of #232 (multi-turn)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

BODY_226=$(curl -s -H "Authorization: token $TOKEN" "https://api.github.com/repos/$REPO/issues/226" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("body","") or "(no body)")')

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/232/comments" \
  -d "$(python3 -c "
import json
body = '''Context preserved from duplicate #226 (Initial multi-turn enabling):

$BODY_226'''
print(json.dumps({'body': body}))
")" | python3 -c 'import sys,json; print(f"Commented on #232: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/226/comments" \
  -d '{"body":"Closing as duplicate of #232 (multi-turn implementation). Both track the same multi-turn feature. Unique context has been copied to #232."}' | python3 -c 'import sys,json; print(f"Commented on #226: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/226" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c 'import sys,json; print(f"#226 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

- [ ] **Step 4: Close #29 as superseded by #79 (submission checker)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

BODY_29=$(curl -s -H "Authorization: token $TOKEN" "https://api.github.com/repos/$REPO/issues/29" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("body","") or "(no body)")')

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/79/comments" \
  -d "$(python3 -c "
import json
body = '''Context preserved from superseded #29 (submission checker for 6.0):

$BODY_29'''
print(json.dumps({'body': body}))
")" | python3 -c 'import sys,json; print(f"Commented on #79: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/29/comments" \
  -d '{"body":"Closing as superseded by #79 (submission checker compatibility mode). #29 was version-specific (6.0) while #79 covers the general compatibility feature. Context has been preserved on #79."}' | python3 -c 'import sys,json; print(f"Commented on #29: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/29" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c 'import sys,json; print(f"#29 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

- [ ] **Step 5: Close #207 as duplicate of #208 (report generation)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

BODY_207=$(curl -s -H "Authorization: token $TOKEN" "https://api.github.com/repos/$REPO/issues/207" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("body","") or "(no body)")')

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/208/comments" \
  -d "$(python3 -c "
import json
body = '''Context preserved from duplicate #207 (speedup tokenizer report generation):

$BODY_207'''
print(json.dumps({'body': body}))
")" | python3 -c 'import sys,json; print(f"Commented on #208: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/207/comments" \
  -d '{"body":"Closing as duplicate of #208 (optimize report generation). #207 describes a specific approach (parallel tokenization) to #208'\''s broader goal. Context has been preserved on #208."}' | python3 -c 'import sys,json; print(f"Commented on #207: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/207" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c 'import sys,json; print(f"#207 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

- [ ] **Step 6: Close #83 as superseded by #223 (roadmap)**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/83/comments" \
  -d '{"body":"Closing as superseded by #223 (Phase 2 Roadmap). The Q1 roadmap is complete and Phase 2 planning has taken over."}' | python3 -c 'import sys,json; print(f"Commented on #83: {json.load(sys.stdin).get(\"id\",\"error\")}")'

curl -s -X PATCH -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/83" \
  -d '{"state":"closed","state_reason":"completed"}' | python3 -c 'import sys,json; print(f"#83 state: {json.load(sys.stdin).get(\"state\",\"error\")}")'
```

---

### Task 4: Delete Legacy Labels

Remove old labels that have been replaced. Only delete after all issues have been relabeled (Task 2 complete).

**Files:** None (API only)

- [ ] **Step 1: Delete all legacy labels**

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# URL-encode label names: spaces→%20, colons are fine in DELETE paths
for label in "bug" "feature" "enhancement" "documentation" "performance" "question" \
  "P0" "P1" "P2" "ShowStopper" "testing" "accuracy" "dataset" "Roadmap" "blocked" \
  "rules" "MLCommons"; do
  encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$label'))")
  echo -n "Deleting '$label'... "
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
    -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/labels/$encoded")
  if [ "$STATUS" = "204" ]; then echo "deleted"; elif [ "$STATUS" = "404" ]; then echo "not found (already gone)"; else echo "status $STATUS"; fi
done
```

Expected: Each label prints "deleted" or "not found". No errors.

- [ ] **Step 2: Verify final label set**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" \
  "https://api.github.com/repos/mlcommons/endpoints/labels?per_page=100" | \
  python3 -c "
import sys, json
labels = json.load(sys.stdin)
names = sorted([l['name'] for l in labels])
print(f'Total labels: {len(names)}')
for n in names:
    print(f'  {n}')
"
```

Expected: Only new prefixed labels plus kept labels (`good first issue`, `help wanted`, `mlcommons`, `dependencies`, `security`, `duplicate`, `invalid`, `wontfix`). No old labels remain.

---

### Task 5: Configure Project Board #57

Set up the board with status field options, custom fields, and 4 views using the GraphQL API.

**Files:** None (API only)

**NOTE:** The board already exists with ID `PVT_kwDOBAnwDc4BTQvY`. We need to configure its fields and views.

- [ ] **Step 1: Get the board's field IDs**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{"query":"{ node(id: \"PVT_kwDOBAnwDc4BTQvY\") { ... on ProjectV2 { fields(first: 20) { nodes { ... on ProjectV2Field { id name } ... on ProjectV2SingleSelectField { id name options { id name } } ... on ProjectV2IterationField { id name } } } } } }"}' | python3 -m json.tool
```

Expected: JSON listing all existing fields with their IDs. Look for the "Status" field and its current options. Record the Status field ID for next steps.

- [ ] **Step 2: Update the Status field with 6 options**

Using the Status field ID from Step 1, update its options. The GraphQL mutation is `updateProjectV2Field`. First, clear existing options and set the 6 new ones.

**Note:** You must adapt the field ID from Step 1's output. Replace `STATUS_FIELD_ID` below with the actual ID.

```bash
TOKEN=$(gh auth token 2>&1)

# Get current status field ID (adapt if needed)
STATUS_FIELD_ID="<from step 1>"

# Update status field options using the updateProjectV2SingleSelectField mutation
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{
    "query": "mutation { updateProjectV2Field(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", fieldId: \"'"$STATUS_FIELD_ID"'\", singleSelectOptions: [{name: \"Inbox\", color: GRAY}, {name: \"Triage\", color: YELLOW}, {name: \"Ready\", color: BLUE}, {name: \"In Progress\", color: ORANGE}, {name: \"In Review\", color: PURPLE}, {name: \"Done\", color: GREEN}]}) { projectV2Field { ... on ProjectV2SingleSelectField { id options { id name } } } }"
  }' | python3 -m json.tool
```

Expected: Returns the updated Status field with 6 options.

- [ ] **Step 3: Create Priority custom field**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{
    "query": "mutation { createProjectV2Field(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", dataType: SINGLE_SELECT, name: \"Priority\", singleSelectOptions: [{name: \"ShowStopper\", color: RED}, {name: \"P0\", color: RED}, {name: \"P1\", color: ORANGE}, {name: \"P2\", color: YELLOW}, {name: \"P3\", color: GREEN}]}) { projectV2Field { ... on ProjectV2SingleSelectField { id name options { id name } } } }"
  }' | python3 -m json.tool
```

Expected: Priority field created with 5 options.

- [ ] **Step 4: Create Area custom field**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{
    "query": "mutation { createProjectV2Field(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", dataType: SINGLE_SELECT, name: \"Area\", singleSelectOptions: [{name: \"core-engine\", color: BLUE}, {name: \"client\", color: BLUE}, {name: \"metrics\", color: BLUE}, {name: \"dataset\", color: BLUE}, {name: \"config-cli\", color: BLUE}, {name: \"evaluation\", color: BLUE}, {name: \"adapters\", color: BLUE}, {name: \"mlcommons\", color: PURPLE}]}) { projectV2Field { ... on ProjectV2SingleSelectField { id name options { id name } } } }"
  }' | python3 -m json.tool
```

Expected: Area field created with 8 options.

- [ ] **Step 5: Create Target Release custom field**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{
    "query": "mutation { createProjectV2Field(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", dataType: SINGLE_SELECT, name: \"Target Release\", singleSelectOptions: [{name: \"v0.5.0\", color: GRAY}, {name: \"v1.0.0\", color: GRAY}]}) { projectV2Field { ... on ProjectV2SingleSelectField { id name options { id name } } } }"
  }' | python3 -m json.tool
```

Expected: Target Release field created.

- [ ] **Step 6: Verify all fields exist**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  -X POST https://api.github.com/graphql \
  -d '{"query":"{ node(id: \"PVT_kwDOBAnwDc4BTQvY\") { ... on ProjectV2 { fields(first: 20) { nodes { ... on ProjectV2Field { id name } ... on ProjectV2SingleSelectField { id name options { id name } } } } } } }"}' | python3 -m json.tool
```

Expected: Status (6 options), Priority (5 options), Area (8 options), Target Release (2 options) all present.

---

### Task 6: Add Issues to Board #57

Add all ShowStopper through P2 issues (~40 after dedup) to the project board and set their status to Triage.

**Files:** None (API only)

- [ ] **Step 1: Get issue node IDs for all Q2 issues**

We need the GraphQL node IDs for each issue to add them to the project. Batch-fetch them:

```bash
TOKEN=$(gh auth token 2>&1)

# All issue numbers to add to board (ShowStopper + P0 + P1 + P2)
ISSUES="84 8 4 86 232 183 138 6 5 9 255 269 237 219 221 202 199 222 220 182 177 176 113 210 268 10 7 254 217 179 178 173 224 208 158 125 115 79 73 68 58 213 133 174 229 228 227 212"

for issue in $ISSUES; do
  NODE_ID=$(curl -s -H "Authorization: token $TOKEN" \
    "https://api.github.com/repos/mlcommons/endpoints/issues/$issue" | \
    python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"])')
  echo "$issue $NODE_ID"
done
```

Expected: A list of issue numbers and their node IDs. Save this output — you'll need it for Step 2.

- [ ] **Step 2: Add each issue to the project**

For each issue, use the `addProjectV2ItemById` mutation. Process in batches to avoid rate limiting:

```bash
TOKEN=$(gh auth token 2>&1)
PROJECT_ID="PVT_kwDOBAnwDc4BTQvY"

# Use the node IDs from Step 1. Example for one issue:
# curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
#   -d '{"query":"mutation { addProjectV2ItemById(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", contentId: \"NODE_ID_HERE\"}) { item { id } } }"}'

# Batch all issues:
ISSUES="84 8 4 86 232 183 138 6 5 9 255 269 237 219 221 202 199 222 220 182 177 176 113 210 268 10 7 254 217 179 178 173 224 208 158 125 115 79 73 68 58 213 133 174 229 228 227 212"

for issue in $ISSUES; do
  NODE_ID=$(curl -s -H "Authorization: token $TOKEN" \
    "https://api.github.com/repos/mlcommons/endpoints/issues/$issue" | \
    python3 -c 'import sys,json; print(json.load(sys.stdin)["node_id"])')

  ITEM_ID=$(curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
    -d "{\"query\":\"mutation { addProjectV2ItemById(input: {projectId: \\\"$PROJECT_ID\\\", contentId: \\\"$NODE_ID\\\"}) { item { id } } }\"}" | \
    python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["addProjectV2ItemById"]["item"]["id"])')

  echo "#$issue added: $ITEM_ID"
  sleep 0.5  # Rate limit courtesy
done
```

Expected: Each issue prints its project item ID. All ~47 issues added.

- [ ] **Step 3: Set all items to Triage status**

After adding items, set their Status field to "Triage". You need the Status field ID and the "Triage" option ID from Task 5 Step 1/2.

```bash
TOKEN=$(gh auth token 2>&1)
PROJECT_ID="PVT_kwDOBAnwDc4BTQvY"
STATUS_FIELD_ID="<from Task 5>"
TRIAGE_OPTION_ID="<from Task 5>"

# For each item added in Step 2, set status to Triage
# Use the item IDs printed in Step 2
for ITEM_ID in <paste item IDs from step 2>; do
  curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
    -d "{\"query\":\"mutation { updateProjectV2ItemFieldValue(input: {projectId: \\\"$PROJECT_ID\\\", itemId: \\\"$ITEM_ID\\\", fieldId: \\\"$STATUS_FIELD_ID\\\", value: {singleSelectOptionId: \\\"$TRIAGE_OPTION_ID\\\"}}) { projectV2Item { id } } }\"}" | \
    python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Set triage: {d}")'
  sleep 0.3
done
```

Expected: All items set to Triage status.

- [ ] **Step 4: Verify board population**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"{ node(id: \"PVT_kwDOBAnwDc4BTQvY\") { ... on ProjectV2 { items(first: 100) { totalCount nodes { content { ... on Issue { number title } } } } } } }"}' | \
  python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data['data']['node']['items']
print(f'Total items on board: {items[\"totalCount\"]}')
for item in items['nodes']:
    c = item['content']
    print(f'  #{c[\"number\"]} {c[\"title\"]}')
"
```

Expected: ~47 issues listed on the board.

---

### Task 7: Create Board Views

Create the 4 views on the project board. The default view already exists (rename to Kanban); create 3 additional views.

**Files:** None (API only)

- [ ] **Step 1: List existing views**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"{ node(id: \"PVT_kwDOBAnwDc4BTQvY\") { ... on ProjectV2 { views(first: 10) { nodes { id name number layout } } } } }"}' | python3 -m json.tool
```

Expected: At least one default view. Record its ID.

- [ ] **Step 2: Update default view to Kanban board layout**

```bash
TOKEN=$(gh auth token 2>&1)
DEFAULT_VIEW_ID="<from step 1>"

curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d "{\"query\":\"mutation { updateProjectV2View(input: {projectId: \\\"PVT_kwDOBAnwDc4BTQvY\\\", viewId: \\\"$DEFAULT_VIEW_ID\\\", name: \\\"Kanban\\\", layout: BOARD_LAYOUT}) { projectV2View { id name layout } } }\"}" | python3 -m json.tool
```

Expected: Default view renamed to "Kanban" with BOARD_LAYOUT.

- [ ] **Step 3: Create Priority Table view**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"mutation { createProjectV2View(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", name: \"Priority Table\", layout: TABLE_LAYOUT}) { projectV2View { id name layout } } }"}' | python3 -m json.tool
```

Expected: New "Priority Table" view created with TABLE_LAYOUT.

- [ ] **Step 4: Create By Assignee view**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"mutation { createProjectV2View(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", name: \"By Assignee\", layout: TABLE_LAYOUT}) { projectV2View { id name layout } } }"}' | python3 -m json.tool
```

Expected: New "By Assignee" view created.

- [ ] **Step 5: Create Stale Issues view**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"mutation { createProjectV2View(input: {projectId: \"PVT_kwDOBAnwDc4BTQvY\", name: \"Stale Issues\", layout: TABLE_LAYOUT}) { projectV2View { id name layout } } }"}' | python3 -m json.tool
```

Expected: New "Stale Issues" view created.

- [ ] **Step 6: Verify all 4 views exist**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -H "Authorization: token $TOKEN" -X POST https://api.github.com/graphql \
  -d '{"query":"{ node(id: \"PVT_kwDOBAnwDc4BTQvY\") { ... on ProjectV2 { views(first: 10) { nodes { id name number layout } } } } }"}' | python3 -m json.tool
```

Expected: 4 views — Kanban (BOARD_LAYOUT), Priority Table (TABLE_LAYOUT), By Assignee (TABLE_LAYOUT), Stale Issues (TABLE_LAYOUT).

**NOTE:** View-level sorting, grouping, and filtering must be configured manually in the GitHub web UI after views are created. The GraphQL API supports creating views and setting layout, but fine-grained sort/group/filter configuration is not fully exposed via API. After this task, open https://github.com/orgs/mlcommons/projects/57 and configure:
- Kanban: Group by Priority
- Priority Table: Sort by Priority field ascending
- By Assignee: Group by Assignee
- Stale Issues: Sort by Updated ascending, filter to items not updated in 30+ days

---

### Task 8: Create Issue Templates

Write the 4 YAML issue form templates and the config file to the local repo.

**Files:**
- Create: `.github/ISSUE_TEMPLATE/100-bug-report.yml`
- Create: `.github/ISSUE_TEMPLATE/200-feature-request.yml`
- Create: `.github/ISSUE_TEMPLATE/300-performance.yml`
- Create: `.github/ISSUE_TEMPLATE/400-dataset-integration.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`

- [ ] **Step 1: Create the ISSUE_TEMPLATE directory**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

- [ ] **Step 2: Write 100-bug-report.yml**

Write to `.github/ISSUE_TEMPLATE/100-bug-report.yml` with the exact content from the design spec Section 3, `100-bug-report.yml`.

- [ ] **Step 3: Write 200-feature-request.yml**

Write to `.github/ISSUE_TEMPLATE/200-feature-request.yml` with the exact content from the design spec Section 3, `200-feature-request.yml`.

- [ ] **Step 4: Write 300-performance.yml**

Write to `.github/ISSUE_TEMPLATE/300-performance.yml` with the exact content from the design spec Section 3, `300-performance.yml`.

- [ ] **Step 5: Write 400-dataset-integration.yml**

Write to `.github/ISSUE_TEMPLATE/400-dataset-integration.yml` with the exact content from the design spec Section 3, `400-dataset-integration.yml`.

- [ ] **Step 6: Write config.yml**

Write to `.github/ISSUE_TEMPLATE/config.yml`:

```yaml
blank_issues_enabled: true
contact_links:
  - name: Questions & Discussion
    url: https://github.com/mlcommons/endpoints/discussions
    about: Ask questions and discuss ideas before filing an issue
```

- [ ] **Step 7: Verify all template files exist**

```bash
ls -la .github/ISSUE_TEMPLATE/
```

Expected: 5 files — `100-bug-report.yml`, `200-feature-request.yml`, `300-performance.yml`, `400-dataset-integration.yml`, `config.yml`.

- [ ] **Step 8: Commit issue templates**

```bash
git add .github/ISSUE_TEMPLATE/
git commit -m "chore: add issue templates (bug, feature, performance, dataset)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Update CONTRIBUTING.md

Replace the existing 10-line CONTRIBUTING.md with the expanded ~250-line version.

**Files:**
- Modify: `CONTRIBUTING.md` (full rewrite)

- [ ] **Step 1: Write the new CONTRIBUTING.md**

Write the full CONTRIBUTING.md content as designed in Section 4 of the spec. The full text was presented during brainstorming and approved. It includes these sections:

1. Welcome and Table of Contents
2. Ways to Contribute (links to all 4 issue templates)
3. Development Setup (prerequisites, fork/clone, venv, pip install, pre-commit, echo server)
4. Code Style and Conventions (ruff, mypy, line length 88, conventional commits, serialization, performance-sensitive code)
5. Testing (pytest commands, markers, async mode, coverage, fixtures)
6. Submitting Changes (branch naming, PR process, review criteria)
7. Issue Guidelines (templates, lifecycle, priority levels table)
8. MLCommons CLA (existing CLA requirements preserved)
9. Questions section

- [ ] **Step 2: Commit CONTRIBUTING.md**

```bash
git add CONTRIBUTING.md
git commit -m "docs: expand CONTRIBUTING.md with development guide, testing, and issue guidelines

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Link Open PRs to Issues

Add comments on open PRs that implement issues different from their own number, creating explicit linkage.

**Files:** None (API only)

- [ ] **Step 1: Link PRs to their corresponding issues**

Only PRs where the PR number differs from the issue it implements need explicit linking:

```bash
TOKEN=$(gh auth token 2>&1)
REPO="mlcommons/endpoints"

# PR #226 implements issue #232 (multi-turn)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/226/comments" \
  -d '{"body":"Relates to #232 (multi-turn implementation). This PR provides the initial multi-turn enabling work tracked by #232."}' | python3 -c 'import sys,json; print(f"PR #226 linked to #232: {json.load(sys.stdin).get(\"id\",\"error\")}")'

# PR #207 implements issue #208 (report generation optimization)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/207/comments" \
  -d '{"body":"Relates to #208 (optimize report generation). This PR implements parallel tokenization as one approach to #208."}' | python3 -c 'import sys,json; print(f"PR #207 linked to #208: {json.load(sys.stdin).get(\"id\",\"error\")}")'

# PR #170 implements issue #86 (warmup runs)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/170/comments" \
  -d '{"body":"Relates to #86 (Warmup runs). This PR implements warmup with random dataset as part of #86."}' | python3 -c 'import sys,json; print(f"PR #170 linked to #86: {json.load(sys.stdin).get(\"id\",\"error\")}")'

# PR #205 relates to issue #255 (Make Loadgen Async)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/issues/205/comments" \
  -d '{"body":"Relates to #255 (Make Loadgen Async). Both this PR and #255 target the same async benchmark goal."}' | python3 -c 'import sys,json; print(f"PR #205 linked to #255: {json.load(sys.stdin).get(\"id\",\"error\")}")'
```

Expected: 4 comments posted linking PRs to their primary issues.

---

### Task 11: Push and Create PR

Push the local commits (issue templates + CONTRIBUTING.md) as a PR to the repository.

**Files:** None (git operations)

- [ ] **Step 1: Create a feature branch**

```bash
git checkout -b chore/project-management-setup
```

- [ ] **Step 2: Cherry-pick the commits onto the branch**

If you committed on main, reset main and cherry-pick onto the new branch. Otherwise if you're already on the branch, skip this.

- [ ] **Step 3: Push to remote**

```bash
git push -u origin chore/project-management-setup
```

- [ ] **Step 4: Create the PR**

```bash
TOKEN=$(gh auth token 2>&1)
curl -s -X POST -H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/mlcommons/endpoints/pulls" \
  -d '{
    "title": "chore: add issue templates, expand CONTRIBUTING.md, and project management setup",
    "body": "## Summary\n\n- Add 4 YAML issue form templates (bug report, feature request, performance issue, dataset integration)\n- Expand CONTRIBUTING.md with development setup, code style, testing, PR process, and issue guidelines\n- Part of the project management infrastructure setup (labels, board, and issue migration done via API)\n\n## Related\n\nDesign spec: docs/superpowers/specs/2026-04-07-project-management-design.md\n\n## Test plan\n\n- [ ] Verify issue templates render correctly on GitHub (New Issue page)\n- [ ] Verify CONTRIBUTING.md renders correctly\n- [ ] Verify all links in CONTRIBUTING.md work\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)",
    "head": "chore/project-management-setup",
    "base": "main"
  }' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"PR created: {d.get(\"html_url\", d.get(\"message\", \"error\"))}")'
```

Expected: PR URL printed.

---

### Task 12: Enable Board Automations

Configure the built-in automations on project board #57 via the GitHub web UI.

**Files:** None (manual UI configuration)

**NOTE:** GitHub Projects V2 built-in automations (auto-add, auto-archive, auto-set status on close) are not configurable via the GraphQL API. They must be enabled manually.

- [ ] **Step 1: Open project settings**

Navigate to: https://github.com/orgs/mlcommons/projects/57/settings

- [ ] **Step 2: Enable "Auto-add" workflow**

Under Workflows → Auto-add to project:
- Enable the workflow
- Filter: `is:issue is:open repo:mlcommons/endpoints`
- This ensures all new issues are automatically added to the board with Inbox status

- [ ] **Step 3: Enable "Item closed" workflow**

Under Workflows → Item closed:
- Enable the workflow
- Set status to: Done

- [ ] **Step 4: Enable "Pull request merged" workflow**

Under Workflows → Pull request merged:
- Enable the workflow
- Set status to: Done

- [ ] **Step 5: Enable "Auto-archive items"**

Under Workflows → Auto-archive items:
- Enable the workflow
- Archive items that have been Done for 14 days

---

### Task 13: Configure Board Views in UI

Fine-tune the sort, group, and filter settings for each view in the GitHub web UI.

**Files:** None (manual UI configuration)

- [ ] **Step 1: Configure Kanban view**

Open: https://github.com/orgs/mlcommons/projects/57/views/1
- Set layout to Board (should already be set)
- Column field: Status
- Group by: Priority (ShowStopper at top)
- Filter: `status:Inbox,Triage,Ready,"In Progress","In Review"`

- [ ] **Step 2: Configure Priority Table view**

Open the Priority Table view
- Sort by: Priority ascending (ShowStopper first)
- Show columns: Title, Priority, Area, Status, Assignee, Target Release
- Filter: exclude Done items

- [ ] **Step 3: Configure By Assignee view**

Open the By Assignee view
- Group by: Assignee
- Sort by: Priority ascending within each group
- Show columns: Title, Priority, Area, Status

- [ ] **Step 4: Configure Stale Issues view**

Open the Stale Issues view
- Sort by: Updated date ascending (oldest first)
- Show columns: Title, Priority, Area, Status, Assignee, Updated
- Filter: exclude Done, show only items not updated in 30+ days
