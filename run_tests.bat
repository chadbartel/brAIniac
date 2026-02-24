@echo off
REM Quick script to run the brAIniac test suite (Windows)

echo ============================================
echo brAIniac Test Suite Runner
echo ============================================
echo.

REM Check if Poetry is available
poetry --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Poetry not found. Please install Poetry first.
    exit /b 1
)

echo Installing test dependencies...
poetry install --with dev

echo.
echo ============================================
echo Running Tests
echo ============================================
echo.

REM Run pytest with coverage
poetry run pytest -v --cov=core --cov=servers --cov-report=term-missing --cov-report=html

echo.
echo ============================================
echo Test Results
echo ============================================
echo.

if errorlevel 0 (
    echo ✅ All tests passed!
    echo.
    echo Coverage report generated in htmlcov/index.html
    echo Open it in your browser to view detailed coverage.
) else (
    echo ❌ Some tests failed. See output above for details.
    exit /b 1
)

echo.
echo To run the web test interface:
echo   poetry run python tests/web_test_interface.py
echo.
pause
