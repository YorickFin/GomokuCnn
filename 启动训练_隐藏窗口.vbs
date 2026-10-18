' Hidden-window launcher for training (隐藏窗口后台训练)
' Stop training: double-click 停止训练.bat
Option Explicit

Dim shell, fso, workdir
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Always run from the folder this script lives in (project root)
workdir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = workdir

' window style 0 = hidden; bWaitOnReturn = False (return immediately)
' stdout/stderr -> train_console.log (crash tracebacks land here)
shell.Run "cmd /c set PYTHONUNBUFFERED=1&& uv run train.py > train_console.log 2>&1", 0, False
