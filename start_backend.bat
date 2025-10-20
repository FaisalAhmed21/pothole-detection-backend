@echo off
echo ====================================
echo Starting Pothole Detection Backend
echo ====================================
echo.

REM Add firewall rule (requires admin)
echo Adding firewall rule for port 5000...
netsh advfirewall firewall delete rule name="Python Flask Port 5000" >nul 2>&1
netsh advfirewall firewall add rule name="Python Flask Port 5000" dir=in action=allow protocol=TCP localport=5000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Could not add firewall rule. You may need to run this as Administrator.
    echo Right-click this file and select "Run as administrator"
    echo.
    echo Press any key to continue anyway...
    pause >nul
)

echo.
echo Getting IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "192.168.56"') do (
    set IP=%%a
    set IP=!IP: =!
    echo Your IP: !IP!
)

echo.
echo ====================================
echo Backend will be available at:
echo http://!IP!:5000
echo ====================================
echo.
echo Make sure your phone is connected to the same Wi-Fi network!
echo.
echo Starting Flask server...
echo.

python app.py
