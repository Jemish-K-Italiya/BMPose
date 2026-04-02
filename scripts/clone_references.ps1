param(
    [string]$Destination = "third_party"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

if (-not (Test-Path (Join-Path $Destination "VideoPose3D"))) {
    git clone --depth 1 https://github.com/facebookresearch/VideoPose3D.git (Join-Path $Destination "VideoPose3D")
}

if (-not (Test-Path (Join-Path $Destination "mediapipe"))) {
    git clone --depth 1 https://github.com/google-ai-edge/mediapipe.git (Join-Path $Destination "mediapipe")
}

Write-Host "Reference repositories are available under $Destination"
