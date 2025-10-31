# GitHub Setup Guide

> **Note**: This is a living document that will be refined as the project evolves.

Quick reference for setting up GitHub workflows and branch protection for the MLPerf Inference Endpoint project.

## 📋 Available GitHub Workflows

### Pre-commit Hooks (`.github/workflows/pre-commit.yml`)

Runs pre-commit hooks (ruff, formatters) on every PR and push.

### Auto-Reviewer (`.github/workflows/auto-review.yml`)

Automatically requests code reviews on new PRs.

### Branch Validator (`.github/workflows/branch-validator.yml`)

Enforces branch naming: `feature/*`, `bugfix/*`, `hotfix/*`, `docs/*`, `test/*`, `refactor/*`, `chore/*`, `release/*`.

### Test Workflow (`.github/workflows/test.yml`)

Runs pytest test suite and generates coverage reports.

### PR Template (`.github/pull_request_template.md`)

Standardized PR description template.

## ⚙️ Manual Configuration (GitHub Web UI)

### Branch Protection for `main`

**Settings** → **Branches** → **Add rule**:

- Branch pattern: `main`
- ✓ Require PR before merging (1 approval)
- ✓ Require status checks: pre-commit, test, branch-validator
- ✓ Require conversation resolution
- ✓ Auto-delete head branches

[Full documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

### Repository Settings

**Settings** → **General** → **Pull Requests**:

- Enable squash merging, merge commits, and rebase merging
- Enable auto-delete of head branches

### GitHub Actions Permissions

**Settings** → **Actions** → **General**:

- Allow all actions and reusable workflows
- Enable read and write permissions (for auto-reviewer)

## 🧪 Quick Test

```bash
# Create test branch
git checkout -b feature/test-setup
git push origin feature/test-setup

# Create PR and verify:
# - PR template loads
# - Workflows trigger
# - Branch protection enforced
```

## 🔧 Customization

### Add Reviewers

Edit `.github/workflows/auto-review.yml`:

```yaml
const defaultReviewers = ['username1', 'username2'];
```

### Modify Branch Patterns

Edit `.github/workflows/branch-validator.yml` to add/remove allowed patterns.

### Update PR Template

Edit `.github/pull_request_template.md` for project-specific requirements.

## 📚 References

- [GitHub Actions](https://docs.github.com/en/actions)
- [Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [PR Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
