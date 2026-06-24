param(
    [string]$Python = "python",
    [ValidateSet("Auto", "Cuda128", "Cpu")]
    [string]$TorchBuild = "Auto",
    [switch]$InstallAiTools,
    [switch]$DownloadAiModels,
    [switch]$SkipDevTools,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-NvidiaGpuNames {
    try {
        $Names = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }
        return @($Names | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    } catch {
        return @()
    }
}

function Test-Rtx50Series {
    param(
        [string[]]$GpuNames
    )

    foreach ($Name in $GpuNames) {
        if ($Name -match "(?i)\bRTX\s+(PRO\s+)?50\d{2}\b") {
            return $true
        }
    }
    return $false
}

$GpuNames = @(Get-NvidiaGpuNames)
$HasNvidiaGpu = $GpuNames.Count -gt 0
$HasRtx50SeriesGpu = Test-Rtx50Series $GpuNames
$ResolvedTorchBuild = $TorchBuild
if ($ResolvedTorchBuild -eq "Auto") {
    if ($HasNvidiaGpu) {
        $ResolvedTorchBuild = "Cuda128"
    } else {
        $ResolvedTorchBuild = "Cpu"
    }
}

Write-Host "Repository: $RepoRoot"
if ($HasNvidiaGpu) {
    Write-Host "Detected NVIDIA GPU(s): $($GpuNames -join '; ')"
    if ($HasRtx50SeriesGpu) {
        Write-Host "RTX 50-series GPU detected. CUDA 12.8 PyTorch is required for this generation."
    } else {
        Write-Host "NVIDIA GPU detected, but not RTX 50-series. CUDA 12.8 PyTorch is still selected in Auto mode for GPU support."
    }
} else {
    Write-Host "No NVIDIA GPU detected. Auto mode will install the CPU PyTorch build."
}
Write-Host "Torch build: $ResolvedTorchBuild"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvDir"
    Invoke-Native $Python @("-m", "venv", $VenvDir)
} else {
    Write-Host "Using existing virtual environment at $VenvDir"
}

Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($ResolvedTorchBuild -eq "Cuda128") {
    Write-Host "Installing PyTorch CUDA 12.8 build"
    Invoke-Native $VenvPython @(
        "-m", "pip", "install",
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torchaudio==2.10.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    )
} elseif ($ResolvedTorchBuild -eq "Cpu") {
    Write-Host "Installing PyTorch CPU build"
    Invoke-Native $VenvPython @(
        "-m", "pip", "install",
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )
} else {
    throw "Unsupported torch build: $ResolvedTorchBuild"
}

$ShouldInstallAiTools = [bool]$InstallAiTools -or [bool]$DownloadAiModels

Write-Host "Installing LaMa-compatible project dependencies"
Invoke-Native $VenvPython @(
    "-m", "pip", "install",
    "numpy==1.26.4",
    "opencv-python==4.11.0.86",
    "Pillow==9.5.0",
    "fire==0.5.0",
    "six",
    "termcolor"
)

Invoke-Native $VenvPython @("-m", "pip", "install", "simple-lama-inpainting==0.1.2", "--no-deps")
Invoke-Native $VenvPython @("-m", "pip", "install", "-e", $RepoRoot, "--no-deps")

if ($ShouldInstallAiTools) {
    Write-Host "Installing optional AI detection tools"
    Invoke-Native $VenvPython @(
        "-m", "pip", "install",
        "einops>=0.8",
        "modelscope>=1.29",
        "psutil>=5.9",
        "pycocotools>=2.0.10",
        "setuptools<81",
        "triton-windows"
    )
    Invoke-Native $VenvPython @(
        "-m", "pip", "install",
        "git+https://github.com/facebookresearch/sam3.git@5dd401d1c5c1d5c3eedff06d41b77af824517619"
    )
}

if (-not $SkipDevTools) {
    Write-Host "Installing test and lint tools"
    Invoke-Native $VenvPython @(
        "-m", "pip", "install",
        "pytest>=8.0",
        "pytest-xdist>=3.6",
        "ruff>=0.8"
    )
}

Invoke-Native $VenvPython @("-m", "pip", "check")

if ($DownloadAiModels) {
    Write-Host "Downloading SAM 3.1 model"
    Invoke-Native $VenvPython @("-m", "remove_watermark", "--download-sam3-model")
}

if (-not $SkipDevTools) {
    Write-Host "Verifying test and lint tools"
    Invoke-Native $VenvPython @("-m", "pytest", "--version")
    Invoke-Native $VenvPython @("-m", "ruff", "--version")
}

if (-not $SkipSmokeTest) {
    if ($ResolvedTorchBuild -eq "Cuda128") {
        Write-Host "Running CUDA and LaMa smoke test"
    $SmokeTest = @'
import numpy as np
import torch
from remove_watermark.core import build_simple_lama_runner

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment.")
print(f"gpu={torch.cuda.get_device_name(0)}")

x = torch.ones((512, 512), device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"tensor_device={y.device}")

runner = build_simple_lama_runner("cuda")
image = np.full((96, 96, 3), 220, dtype=np.uint8)
image[32:64, 32:64] = [30, 30, 30]
mask = np.zeros((96, 96), dtype=np.uint8)
mask[32:64, 32:64] = 255
result = runner(image, mask)
print(f"lama_result={result.shape},{result.dtype}")
'@
    } else {
        Write-Host "Running CPU LaMa smoke test"
        $SmokeTest = @'
import numpy as np
import torch
from remove_watermark.core import build_simple_lama_runner

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")

runner = build_simple_lama_runner("cpu")
image = np.full((96, 96, 3), 220, dtype=np.uint8)
image[32:64, 32:64] = [30, 30, 30]
mask = np.zeros((96, 96), dtype=np.uint8)
mask[32:64, 32:64] = 255
result = runner(image, mask)
print(f"lama_result={result.shape},{result.dtype}")
'@
    }
    $SmokeTest | & $VenvPython -
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch/LaMa smoke test failed."
    }
}

Write-Host ""
if ($ResolvedTorchBuild -eq "Cuda128") {
    Write-Host "Ready. Run with:"
    Write-Host ".\.venv\Scripts\python.exe -m remove_watermark"
} else {
    Write-Host "Ready. Run with:"
    Write-Host ".\.venv\Scripts\python.exe -m remove_watermark"
}
if (-not $SkipDevTools) {
    Write-Host "Check the full environment with:"
    Write-Host ".\.venv\Scripts\python.exe -m pytest -q"
    Write-Host ".\.venv\Scripts\python.exe -m ruff check ."
}
