@echo off
chcp 65001 >nul
echo 正在查找训练进程（命令行含 train.py 的 python/uv 进程）...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='uv.exe'\" | Where-Object { $_.CommandLine -like '*train.py*' } | ForEach-Object { Write-Host ('结束进程 PID=' + $_.ProcessId + ' : ' + $_.Name); taskkill /PID $_.ProcessId /T /F }"
echo 完成。
pause
