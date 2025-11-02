# PowerShell script to train model while preventing sleep
# Usage: .\train_no_sleep.ps1

Write-Host "Starting training with sleep prevention..." -ForegroundColor Green

# Import the required .NET type for preventing sleep
Add-Type -TypeDefinition @"
    using System;
    using System.Runtime.InteropServices;
    public class SleepPreventer {
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);

        public const uint ES_CONTINUOUS = 0x80000000;
        public const uint ES_SYSTEM_REQUIRED = 0x00000001;
        public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    }
"@

# Prevent sleep (keep system awake, allow display to turn off)
Write-Host "Preventing system sleep..." -ForegroundColor Yellow
[SleepPreventer]::SetThreadExecutionState(
    [SleepPreventer]::ES_CONTINUOUS -bor
    [SleepPreventer]::ES_SYSTEM_REQUIRED
)

# Run the training
Write-Host "`nStarting model training...`n" -ForegroundColor Cyan
python -m src.model.train `
    --data-dir data/raw_simplified `
    --model-type resnet `
    --num-classes 3 `
    --epochs 30 `
    --batch-size 8 `
    --lr 0.0001 `
    --device cpu

# Allow sleep again
Write-Host "`nTraining complete! Re-enabling sleep..." -ForegroundColor Green
[SleepPreventer]::SetThreadExecutionState([SleepPreventer]::ES_CONTINUOUS)

Write-Host "Done!" -ForegroundColor Green
