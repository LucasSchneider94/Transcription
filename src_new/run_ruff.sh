#!/bin/bash
# Run ruff to find and remove unused imports and code

echo "Running ruff to check for unused code..."
echo ""

cd /Users/lucasschneider/Desktop/Privat/Transcription_Project/Transcription/src_new

# Run ruff check with autofix for unused imports
ruff check --fix --select F401,F841 .

# Run ruff to find other issues
echo ""
echo "Full ruff check:"
ruff check .

echo ""
echo "✓ Ruff cleanup complete"
