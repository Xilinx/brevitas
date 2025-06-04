@ECHO OFF

:: -----------------------------------------------------------------  
:: Default variables (can be overridden in the environment)  
:: -----------------------------------------------------------------  
IF NOT DEFINED SPHINXOPTS    SET "SPHINXOPTS="  
IF NOT DEFINED SPHINXBUILD   SET "SPHINXBUILD=sphinx-build"  
IF NOT DEFINED MULTIVERSION  SET "MULTIVERSION=sphinx-multiversion"  
IF NOT DEFINED SOURCEDIR     SET "SOURCEDIR=source"  
IF NOT DEFINED BUILDDIR      SET "BUILDDIR=build"  
IF NOT DEFINED BUILD_VERSION SET "BUILD_VERSION="  
  
:: -----------------------------------------------------------------  
:: Dispatch to the selected target  
:: -----------------------------------------------------------------  
IF "%~1"==""        GOTO :help  
IF /I "%~1"=="help" GOTO :help  
IF /I "%~1"=="html" GOTO :html  
IF /I "%~1"=="github" GOTO :github  
IF /I "%~1"=="livehtml" GOTO :livehtml  
  
ECHO Unknown target "%~1" 1>&2  
GOTO :help  
  
:: -----------------------------------------------------------------  
:help  
ECHO Available targets:  
ECHO   html       Build documentation for all versions using sphinx-multiversion  
ECHO   github     Build and copy HTML docs to ..\docs  
ECHO   livehtml   Live-reload documentation (single version, not multiversion)  
ECHO   help       Show this message  
GOTO :EOF  
  
:: -----------------------------------------------------------------  
:html  
REM Propagate BUILD_VERSION to the child process via %VERSION%  
SET "VERSION=%BUILD_VERSION%"  
%MULTIVERSION% "%SOURCEDIR%" "%BUILDDIR%/html" %SPHINXOPTS%  
GOTO :EOF  
  
:: -----------------------------------------------------------------  
:github  
REM First build the documentation (same as :html)  
CALL :html  
  
REM Copy the generated HTML to ..\docs
IF NOT EXIST "..\docs" MKDIR "..\docs"  
XCOPY "%BUILDDIR%\html\*" "..\docs\" /E /I /Y >NUL  
  
REM Copy the top-level index.html from the source tree  
COPY "%SOURCEDIR%\index.html" "..\docs\" >NUL  
  
REM Create .nojekyll markers 
TYPE NUL > "..\docs\.nojekyll"  
GOTO :EOF  
  
:: -----------------------------------------------------------------  
:livehtml  
sphinx-autobuild "%SOURCEDIR%" "%BUILDDIR%/livehtml" %SPHINXOPTS%  
GOTO :EOF  
