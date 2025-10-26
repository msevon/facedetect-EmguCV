@echo off
echo Building Face Detection Application...
dotnet build -c Release

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo Build successful! Running application...
echo.

dotnet run --configuration Release

pause

