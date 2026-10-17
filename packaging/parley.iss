; Inno Setup 6 script for the Parley installer. Build it with `uv run --group build python scripts/make_installer.py`
; (passes the defines below; do not compile by hand without them). Result: dist\installer\Parley-Setup-<ver>.exe
;
; Per-user, no admin: installs to %LOCALAPPDATA%\Programs\Parley. It ships the tree built by scripts/build.py:
;   Parley.exe (launcher), current.json (pointer, state good), app\<ver>\ (parley-app.exe + _internal).
; The Parley home (%LOCALAPPDATA%\parley: models, config, CUDA pack) is never created, read or deleted here.
;
; Defines (from scripts/make_installer.py): AppVersion = parley.__version__ (single version source),
; SourceDir = dist\install, OutputDir = dist\installer.

#ifndef AppVersion
  #error Pass /DAppVersion=x.y.z (use scripts/make_installer.py)
#endif
#ifndef SourceDir
  #error Pass /DSourceDir=<dist\install> (use scripts/make_installer.py)
#endif
#ifndef OutputDir
  #define OutputDir "..\dist\installer"
#endif

[Setup]
; Fixed identity of the product: the same AppId lets a newer installer upgrade an older install in place.
AppId={{B4E0FB10-E4F2-4600-9C5B-21DFBFED683D}
AppName=Parley
AppVersion={#AppVersion}
AppVerName=Parley {#AppVersion}
VersionInfoVersion={#AppVersion}
VersionInfoProductName=Parley
DefaultDirName={localappdata}\Programs\Parley
DefaultGroupName=Parley
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=Parley-Setup-{#AppVersion}
; Compression: LZMA2 at "max" with one solid block. The tree is ~600 MB of mostly DLLs; solid compression
; finds the redundancy between them, and a few minutes at build time is cheap next to every user's download.
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Parley
UninstallDisplayIcon={app}\Parley.exe
; Close a running Parley (the launcher or the app) before replacing files, so an upgrade does not hit locked files.
CloseApplications=yes
; ... and do not relaunch it afterwards (a scripted silent upgrade must not start the app).
RestartApplications=no
; The install dir keeps Inno's default per-user ACLs (inherited from %LOCALAPPDATA%\Programs: not writable by
; other standard users). Nothing here widens them. The in-app updater later calls install.harden_dir() on the
; install root and app\, which only removes inherited entries and keeps this user, SYSTEM and Administrators.

; --- Code signing (not done yet) -------------------------------------------------------------------------
; To sign the installer and its uninstaller later: define a SignTool in the Inno IDE or on the ISCC command
; line (/Smysign=...) and uncomment the next two lines. Sign Parley.exe and parley-app.exe before building.
;SignTool=mysign $f
;SignedUninstaller=yes

[Messages]
; Shown by the uninstaller (not in silent mode). %1 is the application name.
ConfirmUninstall=Remove %1 from this computer?%n%nThis deletes the program files only. Your Parley data (downloaded models, settings and the GPU libraries, in the "parley" folder under AppData\Local) is NOT removed; delete that folder yourself if you want the space back.

[Files]
Source: "{#SourceDir}\Parley.exe"; DestDir: "{app}"; Flags: ignoreversion
; Always replaced: it must point at the app version shipped in this installer.
Source: "{#SourceDir}\current.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Upgrade: drop every older app tree (and the update pointer's backup) so the install is exactly the shipped
; version; the older tree would otherwise be orphaned by the fresh current.json.
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{app}\current.json.prev"
Type: files; Name: "{app}\current.json.tmp"

[UninstallDelete]
; Only what Parley itself creates and Inno does not track: app\<newer>\ trees from the in-app updater, the pointer
; and its backups/temp files, a replaced launcher (Parley.exe.old). Never the bare {app}: the user may have chosen a
; folder that holds other files. Inno removes {app} itself once it is empty.
Type: filesandordirs; Name: "{app}\app"
Type: files; Name: "{app}\current.json*"
Type: files; Name: "{app}\Parley.exe.*"

[Icons]
; Console launcher for now; a windowed/tray launcher belongs to the later tray-app item.
Name: "{userprograms}\Parley"; Filename: "{app}\Parley.exe"; WorkingDir: "{app}"

[Run]
; Finish page checkbox; never runs in a silent install (tests and scripted installs must not start the app).
Filename: "{app}\Parley.exe"; Description: "{cm:LaunchProgram,Parley}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Code]
function VersionPart(var S: String): Integer;
var
  P: Integer;
begin
  P := Pos('.', S);
  if P = 0 then
  begin
    Result := StrToIntDef(S, 0);
    S := '';
  end else
  begin
    Result := StrToIntDef(Copy(S, 1, P - 1), 0);
    S := Copy(S, P + 1, Length(S));
  end;
end;

function CompareVersions(A, B: String): Integer;
var
  I, X, Y: Integer;
begin
  Result := 0;
  for I := 1 to 3 do
  begin
    X := VersionPart(A);
    Y := VersionPart(B);
    if X <> Y then
    begin
      if X > Y then Result := 1 else Result := -1;
      Exit;
    end;
  end;
end;

{ The version recorded in an existing install's current.json ('' if there is none or it is unreadable). }
function InstalledVersion(Dir: String): String;
var
  S: AnsiString;
  P, Q: Integer;
  T: String;
begin
  { Missing or corrupt current.json: return '' and the install proceeds (the launcher repairs the pointer). }
  Result := '';
  if not LoadStringFromFile(AddBackslash(Dir) + 'current.json', S) then Exit;
  T := S;
  P := Pos('"version"', T);
  if P = 0 then Exit;
  T := Copy(T, P + 9, 48);
  { whitespace-tolerant: "version" : "x" and compact "version":"x" }
  T := TrimLeft(T);
  if (T = '') or (T[1] <> ':') then Exit;
  T := TrimLeft(Copy(T, 2, Length(T)));
  if (T = '') or (T[1] <> '"') then Exit;
  T := Copy(T, 2, 32);
  Q := Pos('"', T);
  if Q = 0 then Exit;
  Result := Copy(T, 1, Q - 1);
end;

{ The in-app updater hardlinks unchanged files into a new tree and makes them read-only; the hardlink shares the
  attribute, so the older tree's files become read-only too and Inno's own file deletion fails on them (error 5).
  Clear the attribute under the install dir's app folder before anything is deleted (upgrade) or uninstalled. }
procedure ClearReadOnly(Dir: String);
var
  Code: Integer;
begin
  if DirExists(Dir) then
    Exec(ExpandConstant('{sys}\attrib.exe'), '-R "' + AddBackslash(Dir) + '*" /S /D', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    ClearReadOnly(ExpandConstant('{app}\app'));
end;

{ Refuse to replace a newer install (for example one the in-app updater moved forward) with this older installer. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Have: String;
begin
  Result := '';
  Have := InstalledVersion(ExpandConstant('{app}'));
  if (Have <> '') and (CompareVersions(Have, '{#AppVersion}') > 0) then
    Result := 'Parley ' + Have + ' is already installed here and is newer than this installer ({#AppVersion}). Nothing was changed.'
  else
    ClearReadOnly(ExpandConstant('{app}\app'));
end;
