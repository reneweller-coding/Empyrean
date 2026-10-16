# Empyrean for Meta Quest

You in the cloud: the headset sends your head and your hands to Empyrean on the desktop, which computes the sound on
its graphics card (twenty thousand partials a voice need the desktop's GPU, so the app makes no sound itself). Native
OpenXR, no game engine -- `NativeActivity` + `android_native_app_glue`, EGL, GLES 3, the Khronos OpenXR loader,
`XR_EXT_hand_tracking`. The frame of the app (session, swapchains, point renderer, font, hands, bridge) is Totality's
Quest app, without its synthesizer and its audio.

```
Quest/
  CMakeLists.txt        NDK build of libempyquest.so (links openxr_loader only)
  AndroidManifest.xml   NativeActivity, hasCode=false, hand-tracking permission/features, VR category
  src/main.cpp          the app: OpenXR session, the head and the hands, the bridge, the sky and the panel
  res/mipmap-*/         the launcher icon at five densities (from Deploy/empyrean_512.png)
  fetch_thirdparty.ps1  the OpenXR loader (prefab AAR) into ../ThirdParty (a junction into Noctuary's)
  build_apk.ps1         CMake/NDK -> aapt2 -> jar -> zipalign -> apksigner (debug key)
```

## Build

```powershell
powershell -File Quest\fetch_thirdparty.ps1
powershell -File Quest\build_apk.ps1
adb install -r bin\quest\EmpyreanQuest.apk
```

Needs NDK r27 (`C:\Android-Buildtools\sdk\ndk\27.2.12479018`), build-tools 34, platform android-34 and JDK 17 -- the
parameters at the top of `build_apk.ps1`. No Gradle and no ninja. The APK is under a megabyte.

Built on 16.10.2026 without a headset attached: it compiles and links for arm64 without a warning and the APK is
signed, but it has not run on a device yet.

## What it sends

Write the desktop's address into `empy.cfg` and push it to the headset:

```
bridge_host=192.168.1.20
bridge_port=9105
```

```
adb push empy.cfg /sdcard/Android/data/com.reneweller.empyrean.quest/files/empy.cfg
```

| Message | To | What |
|---|---|---|
| `/hands` | the headset port (9105) | the family's six floats: both hands' heights, pinches, tracked -- the same grammar as every generator's app (the desktop's Plugin/Frame.h, Headset): the pinches play and stop the conductor, step it, bring Brain 2 in and out, both hands choose the next preset; the heights move Macro 2 and Macro 1 |
| `/pose` | the port plus 100 (9205) | twelve floats, sixty times a second: the head's turn (yaw, radians, positive to the right) and pitch, each hand's x, y, z in metres from the head in the room as it stood when the app began (x right, y up, z ahead), the pinches, whether each hand is seen (Plugin/Pose.h) |

The room's forward is where you faced when the app began, and again whenever the headset is recentred. On the desktop
the Rooms page's Headset section decides what the pose does: Head Tracking keeps the binaural cloud and the thunder
where they are as you turn, Hands puts your hands into the wind (the chimes and the leaves feel their wake), onto the
halo (its rim goes round them), into the wave room (an open hand absorbs, a pinching one is where the voices sound) and
into the slime mould's trail (it feeds at the pitch of the hand's height).

## In the headset

A sky of slow points round you (a turn in ten minutes), a faint ring at arm's reach where the hands act, your hands'
joints (warmer as they pinch), and a panel low in front: where it sends, or that it does not.
