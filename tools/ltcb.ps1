# ltcb.ps1 - Measures bwarp as the Large Text Compression Benchmark reports it.
#
# For enwik8 and enwik9: the compressed size, and the wall-clock time of the whole compress and decompress
# processes (CUDA start-up and teardown included), the best of -Runs; checks each round trip. Then the
# decompresser size: the zipped source (git archive of HEAD), and the total LTCB ranks by.
#
#   powershell -File tools\ltcb.ps1 -Data <dir with enwik8 and enwik9> [-Runs 3] [-Exe <bwarp.exe>]
#
# Part of bwarp, a GPU-based BWT file compressor.
# Copyright (c) 2026 Alex Schumacher
# Licensed under the MIT License; see the LICENSE file for details.

param(
    [Parameter(Mandatory = $true)][string]$Data,
    [int]$Runs = 3,
    [string]$Exe = '',
    [string]$Work = $env:TEMP
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path  # $PSScriptRoot is empty in parameter defaults under PowerShell 5.1
$Exe = (Resolve-Path $(if ($Exe) { $Exe } else { "$root\build-cuda\bin\Release\bwarp.exe" })).Path
$options = @('--segment', '16', '--slice', '16')  # the LTCB settings: 0.5% smaller on enwik9 than the defaults

function Find-Input([string]$name) {
    foreach ($candidate in "$Data\$name", "$Data\$name\$name") {
        if (Test-Path $candidate -PathType Leaf) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "$name not found under $Data"
}

function Time-Best([string[]]$arguments) {
    $best = [double]::MaxValue

    for ($r = 0; $r -lt $Runs; $r++) {
        $watch = [Diagnostics.Stopwatch]::StartNew()
        & $Exe @arguments | Out-Null
        $watch.Stop()

        if ($LASTEXITCODE -ne 0) {
            throw "bwarp $($arguments -join ' ') failed with exit code $LASTEXITCODE"
        }

        $best = [Math]::Min($best, $watch.Elapsed.TotalSeconds)
    }

    return $best
}

$results = @()

foreach ($name in 'enwik8', 'enwik9') {
    $in = Find-Input $name
    $packed = Join-Path $Work "$name.bwarp"
    $out = Join-Path $Work "$name.out"
    $bytes = (Get-Item $in).Length

    $compress = Time-Best (@('compress', $in, $packed, '--force', '--quiet') + $options)
    $decompress = Time-Best @('decompress', $packed, $out, '--force', '--quiet')

    if ((Get-FileHash $in).Hash -ne (Get-FileHash $out).Hash) {
        throw "$name does not round-trip"
    }

    $results += [pscustomobject]@{
        File = $name
        Compressed = (Get-Item $packed).Length
        'Compress ns/byte' = [Math]::Round(1e9 * $compress / $bytes, 2)
        'Decompress ns/byte' = [Math]::Round(1e9 * $decompress / $bytes, 2)
    }

    Remove-Item $packed, $out
}

$zip = Join-Path $Work 'bwarp-src.zip'
git -C $root archive --format=zip -9 -o $zip HEAD src CMakeLists.txt CMakePresets.json build.bat LICENSE
$source = (Get-Item $zip).Length
Remove-Item $zip

$results | Format-Table -AutoSize | Out-String -Width 120 | Write-Output
'decompresser (zipped source of HEAD): {0:N0} bytes' -f $source
'enwik9 + decompresser:                {0:N0} bytes' -f ($results[1].Compressed + $source)
'options: {0}; best of {1} runs; {2}' -f ($options -join ' '), $Runs,
    ((& nvidia-smi '--query-gpu=name,driver_version' '--format=csv,noheader') -join ' ')
