---
name: review-pro
description: "Multi-AI parallel code review council that combines Codex and Claude perspectives, posting inline comments on specific files and lines. Use when the user asks to review a PR/MR, review code changes, or says 'review my code', 'review PR #123', 'code review', 'review my changes'."
user-invocable: true
---

# Review Pro — Multi-AI Code Review Council

Run two independent code reviews in parallel (Codex and Claude), synthesize findings, then post **inline comments on specific files and lines** in the PR/MR — not a single blob comment.

## Why two reviewers?

Each AI has different strengths and blind spots. Codex is strong at spotting code patterns and style issues. Claude excels at architectural reasoning and catching subtle bugs. By combining both perspectives, you get broader coverage than any single review — and the synthesis step filters out noise so the user sees only what truly matters.

## Arguments

The user provides a PR/MR number or URL. If omitted, use the current branch's PR.

Examples:
- `/review-pro 161`
- `/review-pro https://github.com/owner/repo/pull/161`
- `/review-pro` (reviews current branch's PR)

## Workflow

### Step 0: Detect platform and resolve PR/MR

Determine whether we are on GitHub or GitLab:
```bash
# Check for GitHub remote
git remote -v | grep -q github.com && echo "github" || echo "gitlab"
```

Resolve the PR/MR number:
- If the user provided a number, use it directly
- If the user provided a URL, extract the number
- If neither, find the PR/MR for the current branch:
  - GitHub: `gh pr view --json number --jq '.number'`
  - GitLab: `glab mr view --json iid --jq '.iid'` (or parse from `glab mr list --source-branch $(git branch --show-current)`)

Get the head commit SHA (needed for inline comments):
- GitHub: `gh pr view <number> --json headRefOid --jq '.headRefOid'`
- GitLab: `glab api projects/:id/merge_requests/<iid> | jq -r '.sha'`

Get the repo identifier:
- GitHub: `gh repo view --json nameWithOwner --jq '.nameWithOwner'` (returns `owner/repo`)
- GitLab: `glab api projects/:id --jq '.id'` (returns project ID)

### Step 1: Get the diff

Get the PR/MR diff for review:
- GitHub: `gh pr diff <number>`
- GitLab: `glab mr diff <iid>`

### Step 2: Launch two parallel reviews

Spawn two subagents **in the same turn** — both running concurrently:

1. **Codex review** — run in a background Bash command:
   ```bash
   codex exec --full-auto --skip-git-repo-check \
     "Review the code changes in PR #<number> (use 'gh pr diff <number>' to see the diff). For EACH issue you find, output it in this exact format:

   ISSUE_START
   FILE: <relative file path>
   LINE: <line number in the new version of the file>
   SEVERITY: critical|high|medium|low
   BODY: <your review comment — be specific and actionable>
   ISSUE_END

   Focus on: bugs, correctness, security, performance, and design. Skip style nitpicks." \
     2>&1 | tee review.codex.raw
   ```

2. **Claude review** — use an Agent subagent:
   - Read the diff
   - Perform a thorough code review covering: correctness, bugs, performance, security, and design
   - For each issue, output in the same structured format:
     ```
     ISSUE_START
     FILE: <path>
     LINE: <line>
     SEVERITY: critical|high|medium|low
     BODY: <comment>
     ISSUE_END
     ```
   - Write to `review.claude.raw`

Both MUST be launched in parallel (same message, multiple tool calls).

### Step 3: Parse and synthesize

Once both reviews complete:

1. **Parse** both `review.codex.raw` and `review.claude.raw` into structured issue lists
2. **Cross-reference** — issues flagged by both reviewers get boosted confidence
3. **Deduplicate** — merge issues on the same file+line range, noting which reviewer(s) flagged each
4. **Filter** — drop low-severity style nitpicks; keep bugs, correctness, security, performance, design issues
5. **Prioritize** — rank by: bugs/correctness > security > performance > design

### Step 4: Post inline comments

Post each issue as an **inline comment on the specific file and line** in the PR/MR.

#### GitHub — Post a single review with inline comments

Build a JSON payload and submit via `gh api`:

```bash
# Build the review JSON with inline comments
cat > /tmp/review_payload.json <<'REVIEW_EOF'
{
  "commit_id": "<HEAD_SHA>",
  "body": "## Review Pro — Multi-AI Code Review\n\nFound <N> issues. Each comment below was flagged by [Codex], [Claude], or [Both].",
  "event": "COMMENT",
  "comments": [
    {
      "path": "<file_path>",
      "line": <line_number>,
      "side": "RIGHT",
      "body": "[<reviewer(s)>] **<severity>**: <comment body>"
    }
  ]
}
REVIEW_EOF

gh api repos/<owner>/<repo>/pulls/<number>/reviews --input /tmp/review_payload.json
```

**Important notes for GitHub inline comments:**
- `line` must be a line within the diff hunk (not arbitrary file lines). If a line is outside the diff, fall back to posting a general comment mentioning the file and line.
- `side` should be `"RIGHT"` for lines in the new version of the file.
- `commit_id` must be the HEAD SHA of the PR.
- `event` should be `"COMMENT"` (neutral review — no approve/reject).

#### GitLab — Post individual discussion threads

GitLab requires one API call per inline comment:

```bash
# For each issue, create a discussion thread
glab api projects/<project_id>/merge_requests/<iid>/discussions \
  -X POST \
  -f "body=[<reviewer(s)>] **<severity>**: <comment body>" \
  -f "position[position_type]=text" \
  -f "position[base_sha]=<base_sha>" \
  -f "position[head_sha]=<head_sha>" \
  -f "position[start_sha]=<start_sha>" \
  -f "position[new_path]=<file_path>" \
  -f "position[old_path]=<file_path>" \
  -f "position[new_line]=<line_number>"
```

To get the required SHAs for GitLab:
```bash
# Get diff refs
glab api projects/<project_id>/merge_requests/<iid>/versions | jq '.[0] | {base_commit_sha, head_commit_sha, start_commit_sha}'
```

### Step 5: Post summary comment

After inline comments are posted, add a top-level summary comment:

```markdown
## Review Pro — Multi-AI Code Review Council

Reviewed by: **Codex** + **Claude**

Found **<N>** issues across **<M>** files:
- <count> critical/high
- <count> medium
- <count> low

Each issue is posted as an inline comment on the relevant file and line.

| # | File | Line | Severity | Reviewer(s) | Summary |
|---|------|------|----------|-------------|---------|
| 1 | `path/to/file.py` | 42 | high | Both | Brief description |
| ... | | | | | |
```

Post via:
- GitHub: `gh pr comment <number> --body "<summary>"`
- GitLab: `glab mr comment <iid> --message "<summary>"`

## Fallback behavior

- If `codex` CLI is not installed, do a Claude-only review and note this in the summary.
- If inline comment posting fails for a specific comment (e.g., line not in diff), fall back to including it in the summary table with the file:line reference.
- If the review finds no issues, post a simple comment: "Review Pro: No issues found. Reviewed by Codex + Claude."

## Notes

- The review scope is the PR/MR diff. The user can specify a commit range if needed.
- All comments are posted with `event: "COMMENT"` — the skill never approves or requests changes automatically.
- For very large PRs (>100 files), focus on source files and skip generated/vendored code.
- Clean up temporary files (`review.codex.raw`, `review.claude.raw`, `/tmp/review_payload.json`) after posting.
