# Always use this instead of plain 'func start'
# Forces the Azure Functions host to use Python 3.10 (where all packages are installed)
# Running plain 'func start' picks Python 3.14 (system) which has NO packages installed

$env:PY_PYTHON = "3.10"
Write-Host "Starting Azure Functions with Python 3.10..." -ForegroundColor Cyan
func start
