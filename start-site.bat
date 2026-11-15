@echo off
rem Double-click to serve the site and open it in your browser. Close this window to stop.
rem The window also prints the address to type on a phone that is on the same Wi-Fi.
cd /d "%~dp0"
start "" http://localhost:5533
node tools\serve.mjs site 5533 0.0.0.0
