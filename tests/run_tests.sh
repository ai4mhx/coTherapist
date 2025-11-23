#!/bin/bash
# Test runner for coTherapist

echo "======================================================================"
echo "coTherapist Test Suite"
echo "======================================================================"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not installed. Run: pip install -r requirements.txt"
    exit 1
fi

echo "✓ Dependencies OK"
echo ""

# Run unit tests
echo "Running Unit Tests..."
echo "----------------------------------------------------------------------"

echo "1. Testing Safety Filter..."
python tests/unit/test_safety.py
if [ $? -eq 0 ]; then
    echo "   ✓ Safety tests passed"
else
    echo "   ✗ Safety tests failed"
    exit 1
fi
echo ""

echo "2. Testing COTHERF Evaluation..."
python tests/unit/test_evaluation.py
if [ $? -eq 0 ]; then
    echo "   ✓ Evaluation tests passed"
else
    echo "   ✗ Evaluation tests failed"
    exit 1
fi
echo ""

echo "3. Testing Psychometric Analysis..."
python tests/unit/test_psychometric.py
if [ $? -eq 0 ]; then
    echo "   ✓ Psychometric tests passed"
else
    echo "   ✗ Psychometric tests failed"
    exit 1
fi
echo ""

echo "======================================================================"
echo "All Tests Passed! ✓"
echo "======================================================================"
