# GitHub Setup Guide

This guide walks you through setting up all the GitHub features for the MLPerf Inference Endpoint project.

## 🚀 What We've Set Up

### 1. ✅ Pre-commit Hooks in CI/CD

- **File**: `.github/workflows/pre-commit.yml`
- **What it does**: Runs pre-commit hooks on every PR and push
- **Status**: Ready to use

### 2. ✅ Default Reviewers

- **File**: `.github/workflows/auto-review.yml`
- **What it does**: Automatically requests reviews from you (zhihanj) for all PRs
- **Status**: Ready to use

### 3. ✅ Branch Name Validator

- **File**: `.github/workflows/branch-validator.yml`
- **What it does**: Enforces branch naming conventions
- **Status**: Ready to use

### 4. ✅ PR Template

- **File**: `.github/pull_request_template.md`
- **What it does**: Provides a comprehensive template for all PRs
- **Status**: Ready to use

### 5. ✅ Test Workflow

- **File**: `.github/workflows/test.yml`
- **What it does**: Runs tests and generates coverage reports
- **Status**: Ready to use

## 🔧 Manual GitHub Configuration Required

### Step 1: Enable Branch Protection Rules

1. **Go to your repository**: `https://github.com/nvzhihanj/Inference-endpoint`
2. **Click Settings** → **Branches**
3. **Click "Add rule"** for the `main` branch
4. **Configure the following**:

```
Branch name pattern: main

✓ Require a pull request before merging
  ✓ Require approvals: 1
  ✓ Dismiss stale PR approvals when new commits are pushed

✓ Require status checks to pass before merging
  ✓ Require branches to be up to date before merging
  ✓ Status checks: pre-commit, test, branch-validator

✓ Require conversation resolution before merging
✓ Require signed commits (optional but recommended)
✓ Require linear history (optional)
✓ Restrict pushes that create files that are executable
```

5. **Click "Create"**

### Step 2: Configure Repository Settings

1. **Go to Settings** → **General**
2. **Scroll down to "Pull Requests"**
3. **Enable**:
   - ✅ "Allow squash merging"
   - ✅ "Allow merge commits"
   - ✅ "Allow rebase merging"
   - ✅ "Automatically delete head branches"

### Step 3: Set Up Issue Templates (Optional)

1. **Go to Settings** → **General**
2. **Scroll down to "Issues"**
3. **Enable**:
   - ✅ "Issues"
   - ✅ "Allow users to create issues"

## 🧪 Testing the Setup

### Test 1: Create a Test Branch

```bash
git checkout -b feature/test-github-setup
git push origin feature/test-github-setup
```

### Test 2: Create a Test PR

1. Go to GitHub and create a PR from `feature/test-github-setup` to `main`
2. Verify:
   - ✅ PR template appears
   - ✅ Branch name validation passes
   - ✅ Pre-commit hooks run
   - ✅ Tests run
   - ✅ You're automatically requested as reviewer

### Test 3: Test Branch Protection

1. Try to merge the test PR
2. Verify it's blocked until:
   - ✅ All status checks pass
   - ✅ You approve the review
   - ✅ Branch is up to date

## 🔍 How Each Feature Works

### Pre-commit Hooks in CI/CD

- **Trigger**: Every PR and push to main
- **What happens**: Runs all pre-commit hooks (ruff, ruff-format, etc.)
- **Result**: PR is blocked if any hooks fail

### Default Reviewers

- **Trigger**: PR opened or marked ready for review
- **What happens**: Automatically requests review from you
- **Customizable**: Easy to add more reviewers later

### Branch Name Validator

- **Trigger**: PR opened/updated
- **What happens**: Checks branch name against patterns
- **Valid patterns**:
  - `feature/component-name`
  - `bugfix/issue-description`
  - `hotfix/critical-fix`
  - `docs/documentation-update`
  - `test/testing-improvements`
  - `refactor/code-improvement`
  - `chore/maintenance-tasks`
  - `release/version-number`

### PR Template

- **Trigger**: Every new PR
- **What happens**: Pre-fills PR description with comprehensive template
- **Includes**: Type of change, testing checklist, review checklist

## 🚨 Troubleshooting

### Issue: Workflows not running

- **Check**: Repository has GitHub Actions enabled
- **Go to**: Settings → Actions → General
- **Enable**: "Allow all actions and reusable workflows"

### Issue: Branch protection not working

- **Check**: Branch protection rules are configured correctly
- **Verify**: Status checks are required and passing
- **Check**: Repository permissions allow branch protection

### Issue: Auto-reviewer not working

- **Check**: GitHub Actions have permission to request reviews
- **Go to**: Settings → Actions → General
- **Enable**: "Read and write permissions"

## 🔄 Updating Configuration

### Adding New Reviewers

Edit `.github/workflows/auto-review.yml`:

```yaml
const defaultReviewers = ['nvzhihanj', 'new-reviewer-username'];
```

### Adding New Branch Patterns

Edit `.github/workflows/branch-validator.yml`:

```yaml
const validPatterns = [
/^feature\/.+/,
/^new-pattern\/.+/,
// ... existing patterns
];
```

### Modifying PR Template

Edit `.github/pull_request_template.md` to add/remove sections.

## 📊 Monitoring

### Check Workflow Status

- **Go to**: Actions tab in your repository
- **Monitor**: Success/failure rates of workflows
- **Investigate**: Failed workflows for issues

### Check Branch Protection

- **Go to**: Settings → Branches
- **Verify**: Protection rules are active
- **Monitor**: Any bypass attempts

## 🎯 Next Steps

1. **Commit and push** these new files
2. **Configure branch protection** manually in GitHub
3. **Test the setup** with a sample PR
4. **Share with your team** how to use the new features
5. **Monitor and adjust** as needed

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Pull Request Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)
