#!/bin/bash

# Pre-commit hook to prevent committing sensitive information

# Check for possible API keys in staged files
APIKEY_REGEX="['\"]?[a-zA-Z0-9_-]{30,}['\"]?"
FILES_WITH_APIKEYS=$(git diff --cached --name-only | xargs grep -l $APIKEY_REGEX 2>/dev/null)

# Check for .env files
ENV_FILES=$(git diff --cached --name-only | grep -E '\.env$')

# Check for files in vector_db
VECTORDB_FILES=$(git diff --cached --name-only | grep -E '^vector_db/.+$' | grep -v -E '(\.gitkeep|README\.md)$')

# Initialize warning flag
WARNING=0

# Check for API keys in files
if [ -n "$FILES_WITH_APIKEYS" ]; then
  echo "⚠️  WARNING: Possible API keys found in these files:"
  echo "$FILES_WITH_APIKEYS"
  WARNING=1
fi

# Check for .env files
if [ -n "$ENV_FILES" ]; then
  echo "⚠️  WARNING: Attempting to commit .env files:"
  echo "$ENV_FILES"
  WARNING=1
fi

# Check for vector_db files
if [ -n "$VECTORDB_FILES" ]; then
  echo "⚠️  WARNING: Attempting to commit files from vector_db:"
  echo "$VECTORDB_FILES"
  WARNING=1
fi

# Provide guidance if warnings were triggered
if [ $WARNING -eq 1 ]; then
  echo ""
  echo "These files may contain sensitive information that should not be committed."
  echo "Please review them carefully and consider removing them from your commit."
  echo ""
  echo "To proceed with the commit anyway, use the --no-verify flag:"
  echo "  git commit --no-verify"
  echo ""
  echo "To stop this commit, press Ctrl+C"
  echo "To continue anyway (NOT RECOMMENDED), press Enter"
  read -p "> " CONTINUE
fi

# Installation instructions
echo ""
echo "To install this pre-commit hook, run:"
echo "  cp pre-commit.sh .git/hooks/pre-commit"
echo "  chmod +x .git/hooks/pre-commit"

exit 0 