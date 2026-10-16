# Fetches what the Quest build needs into ThirdParty/ (ignored by git, beside the vendored dr_libs):
#   openxr-loader/  Khronos OpenXR loader for Android (prefab AAR from Maven Central: headers + libopenxr_loader.so)
#
# Noctuary (the sibling project) already carries it. When its ThirdParty folder is there, this script makes a
# directory junction into it instead of downloading it again; -Copy copies and -Download ignores the sibling.
# After Totality's Quest\fetch_thirdparty.ps1 (Empyrean's app makes no sound: no Oboe).
param(
    [string]$Sibling = "G:\Tools\VRAudio\AmbientSynth\ThirdParty",
    [switch]$Copy,
    [switch]$Download
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$third = Join-Path $root "ThirdParty"
New-Item -ItemType Directory -Force $third | Out-Null

$src = Join-Path $Sibling "openxr-loader"
$dst = Join-Path $third "openxr-loader"
if (Test-Path (Join-Path $dst "prefab")) {
    Write-Host "openxr-loader already here"
} elseif (-not $Download -and (Test-Path (Join-Path $src "prefab"))) {
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    if ($Copy) {
        Copy-Item -Recurse $src $dst
        Write-Host "openxr-loader copied from $src"
    } else {
        & cmd /c mklink /J "`"$dst`"" "`"$src`"" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "cannot link $dst -> $src (use -Copy)" }
        Write-Host "openxr-loader linked to $src"
    }
} else {
    $meta = Invoke-WebRequest -UseBasicParsing "https://repo1.maven.org/maven2/org/khronos/openxr/openxr_loader_for_android/maven-metadata.xml"
    $version = ([xml]$meta.Content).metadata.versioning.release
    $aar = Join-Path $third "openxr_loader_for_android.zip"
    Invoke-WebRequest -UseBasicParsing "https://repo1.maven.org/maven2/org/khronos/openxr/openxr_loader_for_android/$version/openxr_loader_for_android-$version.aar" -OutFile $aar
    New-Item -ItemType Directory -Force $dst | Out-Null
    Expand-Archive -Path $aar -DestinationPath $dst -Force
    Remove-Item $aar
    Write-Host "OpenXR loader $version"
}
Write-Host "ThirdParty ready: $third"
