# Setting Up Git Pre-Commit Hooks

This repository includes a pre-commit hook script that helps prevent accidentally committing sensitive information. This is especially important when working with API keys and personal data.

## What the Pre-Commit Hook Does

The pre-commit hook checks for:

1. Potential API keys in any staged files
2. `.env` files that might contain sensitive environment variables
3. Files in the `vector_db` directory that contain personal data

When it finds any of these, it warns you and gives you a chance to abort the commit.

## Installation

To install the pre-commit hook:

1. Copy the pre-commit script to Git's hooks directory:
   ```bash
   cp pre-commit.sh .git/hooks/pre-commit
   ```

2. Make it executable:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

## Usage

Once installed, the pre-commit hook will run automatically every time you make a commit. If it detects sensitive information:

1. It will display a warning message listing the problematic files
2. It will prompt you to either:
   - Press Ctrl+C to abort the commit (recommended)
   - Press Enter to proceed anyway (not recommended)

## Bypassing the Hook

In situations where you need to bypass the hook (not recommended for most cases):

```bash
git commit --no-verify
```

**Note:** Only use this if you are absolutely certain that you're not committing sensitive information.

## Customizing the Hook

If you need to customize the hook for your specific needs:

1. Edit the `pre-commit.sh` file
2. Reinstall it following the installation steps above

## Best Practices

Even with the pre-commit hook, follow these best practices:

1. Never store API keys or credentials directly in your code
2. Use `.env` files for local configuration, but never commit them
3. Regularly check `.gitignore` to ensure sensitive directories like `vector_db` are excluded
4. Review your changes carefully before committing 