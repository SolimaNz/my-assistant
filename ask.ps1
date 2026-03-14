param(
    [Parameter(Mandatory=$true)]
    [string]$question
)

$response = Invoke-RestMethod -Uri "http://127.0.0.1:5000/ask" -Method POST -Body (@{question=$question} | ConvertTo-Json) -ContentType "application/json"
Write-Host "`nResponse:`n" -ForegroundColor Cyan
Write-Host $response.answer -ForegroundColor Yellow