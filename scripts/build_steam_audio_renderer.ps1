$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path $ProjectDir "model-audio/steam_audio_renderer"
$BuildDir = if ($env:STEAM_AUDIO_RENDERER_BUILD_DIR) {
    $env:STEAM_AUDIO_RENDERER_BUILD_DIR
} else {
    Join-Path $SourceDir "build"
}

if (-not $env:STEAM_AUDIO_SDK_DIR) {
    throw "请先设置 STEAM_AUDIO_SDK_DIR，指向已解压的 Steam Audio SDK。"
}
$Header = Join-Path $env:STEAM_AUDIO_SDK_DIR "include/phonon.h"
if (-not (Test-Path -Path $Header -PathType Leaf)) {
    throw "Steam Audio SDK 缺少 include/phonon.h: $env:STEAM_AUDIO_SDK_DIR"
}
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "未找到 cmake（要求 3.17 或更高版本）。"
}

cmake -S $SourceDir -B $BuildDir -A x64
cmake --build $BuildDir --config Release --parallel
ctest --test-dir $BuildDir -C Release --output-on-failure

Write-Host "renderer: $(Join-Path $BuildDir 'Release/steam-audio-render.exe')"

