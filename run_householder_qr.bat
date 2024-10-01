@echo off
set MPIRUN_FLAGS=-np 4
mpiexec %MPIRUN_FLAGS% householder_qr.exe
pause
