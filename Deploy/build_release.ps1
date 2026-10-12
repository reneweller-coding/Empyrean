<#
.SYNOPSIS
    Empyrean -- builds a release: binaries, tests, manual, staging, checksums, portable zip, setup.

.DESCRIPTION
    After Totality's Deploy\build_release.ps1, for an instrument that needs CUDA (so: MSVC and nvcc, no icx).

      1. configure and build build\release (CMakePresets.json's release preset: static runtime, code for every
         GeForce RTX generation, sm_75 to sm_120, and PTX for what comes after)
      2. run the tests (ctest: doccheck, the engine's and the probe's self-tests, pluginval at strictness 10), unless
         -SkipTests; then pluginval twice more with seeds of its own (a parameter missing from the state can pass
         one seed by coincidence)
      3. the manual: Tools\manual\make_shots.py with the release's standalone, make_manual.py with its empy_render
      4. stage what is installed under dist\stage, and nothing else
      5. check the stage: every file there, the binaries without a DLL dependency on the MSVC or the CUDA runtime
         (nvcuda.dll, the driver's, is loaded at run time and is the only thing the card needs)
      6. SHA256SUMS.txt, the portable zip, and the setup with Inno Setup (ISCC), unless -NoSetup

    Unsigned (the user's decision: no paid certificate); Windows' SmartScreen warns once.
    The version comes from the project() line of CMakeLists.txt, the one source of truth.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File Deploy\build_release.ps1
    powershell -ExecutionPolicy Bypass -File Deploy\build_release.ps1 -NoSetup -SkipTests
#>
param(
    [switch]$NoSetup,
    [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$deploy = Join-Path $root "Deploy"
$build = Join-Path $root "build\release"
$product = "Empyrean"

$project = Get-Content (Join-Path $root "CMakeLists.txt") -Raw
if ($project -notmatch 'project\(\s*\w+\s+VERSION\s+([0-9.]+)') { throw "no version in CMakeLists.txt" }
$Version = $Matches[1]
Write-Host "$product $Version"

# 1. Build.
Push-Location $root
try {
    & cmake --preset release
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
    & cmake --build --preset release --parallel
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
} finally { Pop-Location }
$render = Join-Path $build "Release\empy_render.exe"
$art = Join-Path $build "Plugin\$($product)_artefacts\Release"
$standalone = Join-Path $art "Standalone\$product.exe"
$vst3 = Join-Path $art "VST3\$product.vst3"

# 2. Tests, and pluginval with two more seeds.
if (-not $SkipTests) {
    & ctest --test-dir $build -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
    $pluginval = Join-Path $root "ThirdParty\pluginval\pluginval.exe"
    if (-not (Test-Path $pluginval)) { $pluginval = Join-Path $root "..\BerlinSchoolGenerator\ThirdParty\pluginval\pluginval.exe" }
    if (Test-Path $pluginval) {
        foreach ($seed in @("0x5eed", "0xadd17")) {
            $p = Start-Process -FilePath $pluginval -ArgumentList @("--strictness-level", "10", "--random-seed", $seed, "--timeout-ms", "900000",
                                                                    "--validate", "`"$vst3`"") -Wait -PassThru -NoNewWindow
            if ($p.ExitCode -ne 0) { throw "pluginval failed with seed $seed ($($p.ExitCode))" }
        }
        Write-Host "pluginval: strictness 10 passed with three seeds"
    } else {
        Write-Host "pluginval not found: the extra seeds are skipped"
    }
}

# 3. The manual, from this build: the screenshots are taken again (docs\screenshots), muted.
& python (Join-Path $root "Tools\manual\make_shots.py") $standalone
if ($LASTEXITCODE -ne 0) { throw "screenshots failed" }
& python (Join-Path $root "Tools\manual\make_manual.py") --render $render
if ($LASTEXITCODE -ne 0) { throw "manual failed" }
$manual = Join-Path $root "docs\manual\$product-Manual.pdf"
if (-not (Test-Path $manual)) { throw "no manual PDF (Edge or Chrome prints it)" }

# 4. Stage.
$stage = Join-Path $root "dist\stage"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item $standalone $stage
Copy-Item -Recurse $vst3 $stage
Copy-Item $render $stage
Copy-Item $manual $stage
Copy-Item (Join-Path $deploy "empyrean.ico") $stage
Copy-Item (Join-Path $root "LICENSE") (Join-Path $stage "LICENSE.txt")
Copy-Item (Join-Path $root "README.md") (Join-Path $stage "README.txt")

# 5. Check: the files, and no dependency on the dynamic MSVC runtime or on a CUDA runtime DLL.
$expected = @("$product.exe", "empy_render.exe", "$product.vst3", "$product-Manual.pdf", "empyrean.ico", "LICENSE.txt", "README.txt")
foreach ($f in $expected) { if (-not (Test-Path (Join-Path $stage $f))) { throw "missing in the stage: $f" } }
$dumpbin = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dumpbin) {
    $binaries = @((Join-Path $stage "$product.exe"), (Join-Path $stage "empy_render.exe")) +
                @(Get-ChildItem -Recurse (Join-Path $stage "$product.vst3") -Filter *.vst3 -File | ForEach-Object { $_.FullName })
    foreach ($b in $binaries) {
        $deps = & $dumpbin.FullName /DEPENDENTS $b | Select-String -Pattern "(?i)((vcruntime|msvcp)\d+.*|cudart\w*|cublas\w*|nvrtc\w*)\.dll"
        if ($deps) { throw "$b depends on a runtime DLL (MSVC's or CUDA's): $deps" }
    }
    Write-Host "no dynamic MSVC or CUDA runtime in $($binaries.Count) binaries"
} else {
    Write-Host "dumpbin not found: the runtime check is skipped"
}

# 6. Checksums, portable zip, setup.
$out = Join-Path $root "dist"
$sums = Get-ChildItem -Recurse -File $stage | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
    "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $rel
}
$sums | Set-Content -Encoding utf8 (Join-Path $stage "SHA256SUMS.txt")
$zip = Join-Path $out "$product-$Version-portable.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
Write-Host "wrote $zip"
if (-not $NoSetup) {
    $iscc = Get-ChildItem "C:\Program Files\Inno Setup *\ISCC.exe", "C:\Program Files (x86)\Inno Setup *\ISCC.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if (-not $iscc) { throw "Inno Setup not found. winget install JRSoftware.InnoSetup, or run with -NoSetup." }
    & $iscc.FullName "/DVersion=$Version" (Join-Path $deploy "$product.iss")
    if ($LASTEXITCODE -ne 0) { throw "setup failed" }
    Write-Host "wrote $(Join-Path $out "$product-$Version-Setup.exe")"
}
