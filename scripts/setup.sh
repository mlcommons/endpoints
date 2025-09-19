#!/bin/bash
# Setup script for MLPerf Inference Endpoint Benchmarking System

set -e  # Exit on any error

echo "🚀 Setting up MLPerf Inference Endpoint Benchmarking System..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.12+ required, found Python $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📥 Installing package in development mode..."
pip install -e .

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -r requirements/dev.txt

# Install pre-commit hooks
echo "🔒 Installing pre-commit hooks..."
pre-commit install

# Verify installation
echo "🧪 Verifying installation..."
inference-endpoint --version

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run tests: pytest"
echo "3. Try CLI: inference-endpoint --help"
echo "4. Check development guide: docs/DEVELOPMENT.md"
echo ""
echo "Happy coding! 🚀"
