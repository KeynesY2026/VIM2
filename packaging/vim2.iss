#define AppVersion "0.1.0"
[Setup]
AppId={{74C9AB99-86B1-4725-A47A-E8C85DC37FC7}
AppName=VIM2
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\VIM2
DefaultGroupName=VIM2
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist\installers
OutputBaseFilename=VIM2-0.1.0-windows-x64-Setup
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\VIM2.exe
WizardStyle=modern

[Files]
Source: "..\dist\build\VIM2\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\VIM2"; Filename: "{app}\VIM2.exe"
Name: "{autodesktop}\VIM2"; Filename: "{app}\VIM2.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create desktop shortcut"; Flags: unchecked
