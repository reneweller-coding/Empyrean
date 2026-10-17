; Empyrean -- the Windows installer (after Totality's Deploy\Totality.iss).
;
; Built by Deploy\build_release.ps1, which stages everything under dist\stage first and only then calls the
; compiler. Nothing in here reaches into a build tree: what is in the staging folder is exactly what gets
; installed, so the payload can be looked at before the setup is made.
;
;   ISCC.exe /DVersion=1.0.0 Deploy\Empyrean.iss
;
; The binaries are linked against the static MSVC runtime and the static CUDA runtime: no redistributable, no DLL
; beside the executable. What the graphics card needs is NVIDIA's driver (its nvcuda.dll), which the setup looks
; for and does not bring. Every sound is synthesised: nothing is downloaded, the VST3 needs no data files.

#ifndef Version
  #define Version "1.0.0"
#endif
#define AppName "Empyrean"
#define Publisher "Rene Weller"
#define Stage "..\dist\stage"

[Setup]
AppId={{92857995-F4B2-4347-BA30-134BEB61469C}
AppName={#AppName}
AppVersion={#Version}
AppVerName={#AppName} {#Version}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile={#Stage}\LICENSE.txt
OutputDir=..\dist
OutputBaseFilename={#AppName}-{#Version}-Setup
SetupIconFile={#Stage}\empyrean.ico
UninstallDisplayIcon={app}\{#AppName}.exe
UninstallDisplayName={#AppName} {#Version}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Machine-wide by default, because the VST3 belongs in the shared plug-in folder; "just for me" (or
; /CURRENTUSER) installs into the per-user folders. Every path below is an {auto...} one for that reason.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
VersionInfoVersion={#Version}
VersionInfoCompany={#Publisher}
VersionInfoDescription={#AppName} installer

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[CustomMessages]
en.CompStandalone=Standalone application
en.CompVst3=VST3 plug-in (for a DAW)
en.CompRender=Offline renderer (empy_render, command line)
en.CompJourneys=Empyrean Journeys (journeys into WAV, FLAC, Ogg Vorbis or MP3)
en.CompManual=Manual (PDF)
en.TaskDesktop=Create a desktop shortcut
en.NoAvx2=This processor reports no AVX2 support.%n%n{#AppName} is built for AVX2, which every x86-64 processor since 2013 has. Without it, it will not start.%n%nInstall anyway?
en.NoCuda=No NVIDIA driver with CUDA was found on this computer.%n%n{#AppName} computes its twenty thousand partials per voice on an NVIDIA GeForce RTX card (20 series or newer). Without one it starts, but stays silent.%n%nInstall anyway?
de.CompStandalone=Eigenstaendiges Programm
de.CompVst3=VST3-Plugin (fuer eine DAW)
de.CompRender=Offline-Renderer (empy_render, Kommandozeile)
de.CompJourneys=Empyrean Journeys (Journeys als WAV, FLAC, Ogg Vorbis oder MP3)
de.CompManual=Handbuch (PDF)
de.TaskDesktop=Verknuepfung auf dem Desktop anlegen
de.NoAvx2=Dieser Prozessor meldet keine AVX2-Unterstuetzung.%n%n{#AppName} ist fuer AVX2 gebaut, das jeder x86-64-Prozessor seit 2013 hat. Ohne AVX2 startet es nicht.%n%nTrotzdem installieren?
de.NoCuda=Auf diesem Computer wurde kein NVIDIA-Treiber mit CUDA gefunden.%n%n{#AppName} berechnet seine zwanzigtausend Teiltoene pro Stimme auf einer NVIDIA-GeForce-RTX-Karte (Serie 20 oder neuer). Ohne eine solche startet es, bleibt aber stumm.%n%nTrotzdem installieren?

[Types]
Name: "full"; Description: "Full"
Name: "custom"; Description: "Custom"; Flags: iscustom

[Components]
Name: "standalone"; Description: "{cm:CompStandalone}"; Types: full custom; Flags: fixed
Name: "vst3";       Description: "{cm:CompVst3}";       Types: full custom
Name: "render";     Description: "{cm:CompRender}";     Types: full custom
Name: "journeys";   Description: "{cm:CompJourneys}";   Types: full custom
Name: "manual";     Description: "{cm:CompManual}";     Types: full custom

[Tasks]
Name: "desktopicon"; Description: "{cm:TaskDesktop}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; Components: standalone

[InstallDelete]
; Before a single file is copied: what this installer itself put there, so an update leaves nothing stale
; behind (the lesson Noctuary paid for).
Type: filesandordirs; Name: "{autocf}\VST3\{#AppName}.vst3"
Type: files;          Name: "{app}\{#AppName}-Manual.pdf"

[Files]
Source: "{#Stage}\{#AppName}.exe";          DestDir: "{app}";  Flags: ignoreversion; Components: standalone
Source: "{#Stage}\empyrean.ico";             DestDir: "{app}";  Flags: ignoreversion; Components: standalone
Source: "{#Stage}\README.txt";              DestDir: "{app}";  Flags: ignoreversion isreadme; Components: standalone
Source: "{#Stage}\LICENSE.txt";             DestDir: "{app}";  Flags: ignoreversion; Components: standalone
Source: "{#Stage}\SHA256SUMS.txt";          DestDir: "{app}";  Flags: ignoreversion; Components: standalone
Source: "{#Stage}\empy_render.exe";      DestDir: "{app}";  Flags: ignoreversion; Components: render
Source: "{#Stage}\EmpyreanJourneys.exe"; DestDir: "{app}";  Flags: ignoreversion; Components: journeys
Source: "{#Stage}\{#AppName}-Manual.pdf";   DestDir: "{app}";  Flags: ignoreversion; Components: manual
Source: "{#Stage}\{#AppName}.vst3\*";       DestDir: "{autocf}\VST3\{#AppName}.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: vst3

[Icons]
Name: "{autoprograms}\{#AppName}";          Filename: "{app}\{#AppName}.exe"; IconFilename: "{app}\empyrean.ico"
Name: "{autoprograms}\{#AppName} Manual";   Filename: "{app}\{#AppName}-Manual.pdf"; Components: manual
Name: "{autoprograms}\{#AppName} Journeys"; Filename: "{app}\EmpyreanJourneys.exe"; IconFilename: "{app}\empyrean.ico"; Components: journeys
Name: "{autodesktop}\{#AppName}";           Filename: "{app}\{#AppName}.exe"; IconFilename: "{app}\empyrean.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppName}.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function IsProcessorFeaturePresent(Feature: Integer): Boolean;
  external 'IsProcessorFeaturePresent@kernel32.dll stdcall';

// PF_AVX2_INSTRUCTIONS_AVAILABLE = 40: the host code is built for AVX2 (Simd.h). nvcuda.dll in System32: an
// NVIDIA driver with CUDA (the card's generation is the program's to judge; it names a card it has no code for).
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not IsProcessorFeaturePresent(40) then
    Result := MsgBox(CustomMessage('NoAvx2'), mbConfirmation, MB_YESNO) = IDYES;
  if Result and not FileExists(ExpandConstant('{sys}\nvcuda.dll')) then
    Result := MsgBox(CustomMessage('NoCuda'), mbConfirmation, MB_YESNO) = IDYES;
end;
