@echo off
setlocal
set "NODEDIR=%~dp0"
echo AetherScale cache recovery

echo.
echo Close ComfyUI before running this tool.
echo If Windows is still flushing an old huge mmap after Python was killed,
echo disk activity can continue until the Cache Manager drains those dirty pages.
echo v0.9.2 prevents that backlog; for an already-dirty old mapping a reboot is the fastest reset.
echo.

if defined AETHERSCALE_CACHE_DIR (
  if exist "%AETHERSCALE_CACHE_DIR%" (
    echo Removing configured cache: "%AETHERSCALE_CACHE_DIR%"
    rmdir /s /q "%AETHERSCALE_CACHE_DIR%"
  )
)

if exist "%NODEDIR%.aetherscale_cache" (
  echo Removing legacy cache: "%NODEDIR%.aetherscale_cache"
  rmdir /s /q "%NODEDIR%.aetherscale_cache"
)

for %%D in ("%NODEDIR%..\..\temp\aetherscale_cache") do (
  if exist "%%~fD" (
    echo Removing ComfyUI temp cache: "%%~fD"
    rmdir /s /q "%%~fD"
  )
)

echo Done.
endlocal
