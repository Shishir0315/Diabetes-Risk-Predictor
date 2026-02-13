
$pathToAdd = "C:\Users\SHISHIR M S\AppData\Local\Programs\Python\Python311\Scripts"
$mainPythonPath = "C:\Users\SHISHIR M S\AppData\Local\Programs\Python\Python311"

$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

if (-not $currentPath.Contains($pathToAdd)) {
    $newPath = $currentPath + ";" + $pathToAdd
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added $pathToAdd to User Path."
} else {
    Write-Host "$pathToAdd is already in User Path."
}

if (-not $currentPath.Contains($mainPythonPath)) {
    $newPath = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $mainPythonPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added $mainPythonPath to User Path."
} else {
    Write-Host "$mainPythonPath is already in User Path."
}

Write-Host "Please restart your terminal/IDE for changes to take effect."
