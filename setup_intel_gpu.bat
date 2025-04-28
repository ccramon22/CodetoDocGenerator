@echo off
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
set "PATH=%ONEAPI_ROOT%\compiler\latest\windows\bin;%PATH%"
set "PATH=%ONEAPI_ROOT%\mkl\latest\bin;%PATH%"
set "PATH=%ONEAPI_ROOT%\dnnl\latest\cpu\bin;%PATH%"
set "PATH=%ONEAPI_ROOT%\tbb\latest\bin;%PATH%"
set "PATH=%ONEAPI_ROOT%\dpcpp-ct\latest\bin;%PATH%"
set "PATH=%ONEAPI_ROOT%\debugger\latest\bin;%PATH%"

echo Environment variables set for Intel GPU support
echo Please run this script before starting PyCharm or running your Python code 