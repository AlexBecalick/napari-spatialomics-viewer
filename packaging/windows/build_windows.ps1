$ErrorActionPreference = "Stop"

if (-not $IsWindows) {
    throw "The Windows installer must be built on Windows."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SpecFile = Join-Path $ProjectRoot "packaging\pyinstaller\napari_compare.spec"
$DistDir = Join-Path $ProjectRoot "dist"
$WorkDir = Join-Path $ProjectRoot "build\pyinstaller-windows"
$ArtifactsDir = Join-Path $ProjectRoot "artifacts"
$AppDir = Join-Path $DistDir "NapariCompareXeniumMERSCOPE"
$AppExe = Join-Path $AppDir "NapariCompareXeniumMERSCOPE.exe"

$Version = python -c "from importlib.metadata import version; print(version('napari-compare-xenium-merscope'))"
if ($LASTEXITCODE -ne 0) { throw "Could not determine the package version." }

Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $AppDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $ArtifactsDir | Out-Null

python -m PyInstaller --noconfirm --clean --distpath $DistDir --workpath $WorkDir $SpecFile
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $AppExe)) {
    throw "PyInstaller did not produce $AppExe"
}

$InnoCompiler = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $InnoCompiler)) {
    throw "Inno Setup 6 was not found at $InnoCompiler"
}

$InstallerDefinition = Join-Path $PSScriptRoot "installer.iss"
& $InnoCompiler "/DAppVersion=$Version" "/DSourceDir=$AppDir" "/DOutputDir=$ArtifactsDir" $InstallerDefinition
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }

Write-Host "Windows application: $AppExe"
Write-Host "Windows installer: $ArtifactsDir"
