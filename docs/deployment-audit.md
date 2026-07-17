# Standalone Workflow Deployment Audit

## Result

- Audited at `2026-07-17T20:01:51+08:00` (`2026-07-17T12:01:51Z`).
- Scope: the standalone workflow repository, the read-only official ComfyUI
  checkout, deployed runtime assets, and the two loopback services.
- Status: **technical deployment passed**. Repository separation, dependency
  verification, workflow deployment, live API execution, alpha handling, and
  smoke generation passed.
- Visual status: **demo-only sword continuity**. The eight-frame result is not
  production-ready action animation.

The standalone implementation commit audited below is
`b8babd3a3f5d9343e5a36f48412a8da5e463134a`. The documentation commit that
contains this report necessarily follows that implementation commit.

## Repositories

| Checkout | Repository | Branch | Audited commit | Remote and parity |
| --- | --- | --- | --- | --- |
| Standalone workflow toolkit | `https://github.com/xl732236362/ComfyUI-GameAsset-Workflows` | `main` | `b8babd3a3f5d9343e5a36f48412a8da5e463134a` | Exact HTTPS `origin`; local `HEAD`, `origin/main`, and GitHub REST ref matched; ahead/behind `0/0` |
| ComfyUI runtime | `https://github.com/Comfy-Org/ComfyUI` | `master` | `71b73e3b2bbdfb420aca342d61bef980b5a04f63` | Only the official HTTPS `origin` for fetch and push; local `HEAD`, `origin/master`, and GitHub REST ref matched; ahead/behind `0/0` |

GitHub metadata reported the standalone repository as public, non-fork, with
default branch `main` and no detected license. Its 14-commit public
implementation history ran from `efc3f24551cde64981a13358e84c67f63c79fabe`
through the audited commit. The local ComfyUI safety branch
`codex/pixel-action-continuity-tuning` remains at
`a6d8206bd3dea8d7d1fa67fed7cae693809999fa`.

## Runtime

Both services were responsive and bound only to `127.0.0.1`. PIDs are an audit
snapshot and will change after restart.

| Service | Listener / launcher PID | Started (Asia/Shanghai) | Evidence |
| --- | ---: | --- | --- |
| ComfyUI `http://127.0.0.1:8188` | `23052` / `36360` | `2026-07-17T19:29:33+08:00` | ComfyUI `0.28.0`; Python `3.13.14`; PyTorch `2.12.1+cu130`; `/system_stats` and `/object_info` HTTP 200; 905 node definitions; running/pending queue `0/0` |
| Game-asset API `http://127.0.0.1:8190` | `5952` / `24072` | `2026-07-17T19:32:30+08:00` | Started as `python -m game_asset_api`; root returned `404: Not Found`; an unknown job returned `{"error": "not found"}` without a filesystem path |

The only redirected ComfyUI log files found predate the current service PIDs.
Their content scan found no `Traceback`, `[ERROR]`, or `Exception` marker, but
they are historical files and are not claimed as current-process logs. Live
HTTP health, queue state, job state, and ComfyUI history are the current
runtime evidence.

## Verification

All commands ran from the standalone checkout with the ComfyUI virtual
environment unless noted.

| Check | Result |
| --- | --- |
| `python -m compileall -q game_asset_api scripts` | Exit `0` |
| `python -m pytest tests/game_asset_api_test -q` | `227 passed` |
| `python scripts/audit_repository.py` | Exit `0`, zero index-policy violations |
| `git diff --check` | Exit `0` |
| Workflow worktree blobs versus audited `HEAD` | All five equal |
| `validate_object_info` against live `/object_info` | Passed with 905 advertised node definitions |
| Live `/userdata?dir=workflows` | Returned exactly the five deployed workflow names |
| `python -m pip check` in the ComfyUI environment | `No broken requirements found` |

The repository audit checks index path policy, blocked runtime/model suffixes,
and blob sizes; it does not inspect ordinary file contents. Separately, a
content-aware pattern scan covered every reachable blob in all 14 audited
implementation commits and found zero matches across private-key headers,
GitHub token forms, AWS access-key forms, and credential-assignment patterns.
A full-history filename scan also found zero secret-key or model-weight names.
These checks reduce risk but are not a mathematical proof that arbitrary text
cannot contain a secret.

## Workflows

Every checked-in workflow is byte-for-byte identical to its deployed copy
under `user/default/workflows`. The deployed copies are ignored runtime files
and are not in the ComfyUI Git index.

| Workflow | Bytes | SHA-256 | Deployment |
| --- | ---: | --- | --- |
| `pixel_character_design_api.json` | `3046` | `279d7d1cb7814747eb10550d3565837b6253c4267c996ba5c206851cd3e3e62c` | Exact source/deployed bytes |
| `pixel_character_action_api.json` | `3624` | `c38c59e8f9d4e1566d5109a37d13a1f403abaf641bbc091723f0a4d88a66e789` | Exact source/deployed bytes |
| `pose_controlled_pixel_action_api.json` | `5106` | `e70f1dc499c8077081696d0146b5a607f9305daa0c40ae5c0e0aff6dffc213f3` | Exact source/deployed bytes |
| `video_wan2_2_5B_ti2v.json` | `14523` | `50390798a133757075d750ab78b26274e3bf3e4a09f46aaa2f08d9559e62750b` | Exact source/deployed bytes |
| `wan2_2_5b_dual_balanced.json` | `14128` | `512857670fc4f89500009516d339d03c5fe9bf4370c9a287e5decf5032ebf09d` | Exact source/deployed bytes |

## Models

All nine loader files were read and freshly verified without downloading or
changing their modification times. `MODEL_SPECS` manages the first six; the
three Wan files are external prerequisites documented by the repository.

| Destination relative to ComfyUI root | Bytes | SHA-256 | Ownership / result |
| --- | ---: | --- | --- |
| `models/checkpoints/sd_xl_base_1.0.safetensors` | `6938078334` | `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b` | Managed; `verify_file` passed |
| `models/loras/pixel-art-xl.safetensors` | `170543052` | `4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7` | Managed; `verify_file` passed |
| `models/background_removal/BiRefNet-general-epoch_244.safetensors` | `444473596` | `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` | Managed; `verify_file` passed |
| `models/controlnet/OpenPoseXL2.safetensors` | `5004167829` | `5a4b928cb1e93748217900cb66d4135bf70d932d2924232f925910fad9e43a92` | Managed; `verify_file` passed |
| `models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` | `847517512` | `3f5062b8400c94b7159665b21ba5c62acdcd7682262743d7f2aefedef00e6581` | Managed; `verify_file` passed |
| `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `2528373448` | `6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030` | Managed; `verify_file` passed |
| `models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | `9999658848` | `456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e` | External preinstall; exact size/hash |
| `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `6735906897` | `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68` | External preinstall; exact size/hash |
| `models/vae/wan2.2_vae.safetensors` | `1409400960` | `e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156` | External preinstall; exact size/hash |

## Custom Nodes

| Installed directory | Upstream | Pinned revision | Fresh local evidence |
| --- | --- | --- | --- |
| `custom_nodes/comfyui_controlnet_aux` | `https://github.com/Fannovel16/comfyui_controlnet_aux` | `e8b689a513c3e6b63edc44066560ca5919c0576e` | Revision marker exact; `requirements.txt` present (268 bytes) |
| `custom_nodes/ComfyUI_IPAdapter_plus` | `https://github.com/cubiq/ComfyUI_IPAdapter_plus` | `a0f451a5113cf9becb0847b92884cb10cbdec0ef` | Revision marker exact; this pinned tree has no `requirements.txt` |

Task 7 compared installed source paths against the fixed-revision official
GitHub trees: ControlNet matched `746/746` paths and IPAdapter matched `34/34`.
That tree comparison is historical Task 7 evidence; the marker and requirements
checks above were repeated during this audit.

Installer commit `f6f56b8ca383c9caa97d6108223cf562db711e50` accepts a verified
installed node without requiring or downloading a cached ZIP. Commit
`b8babd3a3f5d9343e5a36f48412a8da5e463134a` still retries requirements
installation for that verified-node path. The current environment's
`pip check` passed.

## Live API Job

Job `2ec589f4-208e-4d8b-82ac-7252d324949b` completed through the API after
ComfyUI was restored to official `master`. Its character and action prompt IDs
both report `success` and `completed=true` in ComfyUI history.

| Artifact relative to ComfyUI root | Bytes | SHA-256 | Image evidence |
| --- | ---: | --- | --- |
| `output/game_assets/2ec589f4-208e-4d8b-82ac-7252d324949b/character.png` | `235846` | `0a759ad342d4a9a1eb88667c247102a609312c29d412c65f050fb3361f0c800c` | `512x512` RGBA, alpha `0..255`, transparent corners, non-empty foreground |
| `output/game_assets/2ec589f4-208e-4d8b-82ac-7252d324949b/frames/000.png` | `5715` | `b52cf72a423ef2ad9cd5148ac17001c3a0c5324f83ef7030a24347c5050449d4` | `64x64` RGBA, alpha `0..255`, transparent corners, non-empty foreground |
| `output/game_assets/2ec589f4-208e-4d8b-82ac-7252d324949b/frames/001.png` | `6438` | `1ee6a5cc2b92e15b6b24a07afaf710fda31c2f2b3ebfe5df926fde18e542ebab` | `64x64` RGBA, alpha `0..255`, transparent corners, non-empty foreground |
| `output/game_assets/2ec589f4-208e-4d8b-82ac-7252d324949b/spritesheet.png` | `12004` | `28bf9b538c957c7dc4d0b9433264222154a6fb366c04c152f49a9540fd6bde4a` | `128x64` RGBA, alpha `0..255`, transparent corners, non-empty foreground |

## Eight-Frame Pose Audit

The Task 7 run used the following reference and produced the eight final frames
below. All eight frame PNGs are `128x128` RGBA with transparent corners and
non-empty foreground. Alpha is `0..255` except frame `000`, whose maximum is
`254`.

| Artifact relative to ComfyUI root | Bytes | SHA-256 | Evidence |
| --- | ---: | --- | --- |
| `input/example.png` | `8589` | `a8e215ad32a0052fc4190e9c5863428d5ee35cd98b246243842ba8360511b7c4` | `768x768` RGB reference |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/000.png` | `17217` | `67769dc2133fa94aff5e3fd40dc12b1dc93522e5d642b2f2884c63632023cd8f` | RGBA, alpha `0..254` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/001.png` | `17648` | `f2c2000bd3f22abebda3f16d11cc6e7687c5d452463c88f090dbcb586ad163ed` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/002.png` | `16215` | `e39c5941c0f522518a421d799deb20df4b4eb634eeefd5c3c56bfb5d7ba30b4e` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/003.png` | `16456` | `87f1b605792ae17dd9c2d38bfe25c56823538d5d11e9565f1894ca2e8ba368ff` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/004.png` | `20022` | `182faa75aeda19df7bc41b34c5924f5f6ffad40ca90b9f51683c49fab795315b` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/005.png` | `18806` | `8f23b3a018fd9470a82966f50692b55b2185774305937237bace10538bf6022b` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/006.png` | `19499` | `b723c2de8c876aeaaf976d3d17631fa0a912e2064191924da7001cbdb1d3f971` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/frames/007.png` | `17680` | `b7474de1faddc55bb661ca4fdb007c5077247dc9ce10f9cf5afcf2c7ed5fdcf4` | RGBA, alpha `0..255` |
| `output/game_assets/standalone-repo-live-audit/pose_action/spritesheet.png` | `142500` | `9f2475a12d3b00525d9bf5b6282876625fb650793743ec3dba3e3d42b6855c6e` | `384x384` RGBA, alpha `0..255`, transparent corners |
| `output/game_assets/standalone-repo-live-audit/pose_action/standalone-repo-live-audit-preview.gif` | `23554` | `195fe3c03dfeaa9466d4d4c3d25ad793a7f2587fce79f2f594b52043f78b4313` | `ffprobe`: GIF, `128x128`, `8` read frames, `1.010000` seconds |

The body poses form a readable anticipation-to-recovery sequence, but the
requested wide horizontal sword arc is unclear: the blade remains mostly
vertical, and sword angle/length, hand contact, face, and silhouette drift
between independently generated frames. OpenPose constrains body joints, not
weapon geometry. The visual result is suitable for workflow demonstration and
iteration, not for production animation without weapon-specific control or
manual cleanup.

## Official-Checkout Deployment Smoke

The latest smoke artifacts were generated after the official ComfyUI branch
switch. Both frames and the sheet are RGBA with alpha `0..255`, transparent
corners, and non-empty foreground.

| Artifact relative to ComfyUI root | Bytes | SHA-256 | Dimensions |
| --- | ---: | --- | --- |
| `output/game_assets/deployment-smoke/pose_action/frames/000.png` | `5723` | `2ee824e804a56e9f77544ecf79700b396950b555953830469436237079adefd8` | `64x64` |
| `output/game_assets/deployment-smoke/pose_action/frames/001.png` | `5332` | `0a37cb13134fab01dee4d462a41d452c8efd33669b5f49e21694e5e5b7f61cd8` | `64x64` |
| `output/game_assets/deployment-smoke/pose_action/spritesheet.png` | `10825` | `1ce0f1f2e7fe9c461159e0142a05c6b41c06ce0f6d9202d5cb4e63f3097dcc9b` | `128x64` |

## Preservation And Security

- The ComfyUI tracked worktree and index were clean. Eight pre-existing
  top-level untracked entries remain untouched across runtime state, local
  planning state, a status marker, two diagnostic/launcher scripts, a
  historical log, local documentation, and a wheel cache.
- The two-file local migration backup set remains outside the standalone Git
  index, and the ComfyUI safety branch remains available. No runtime debris or
  user file was removed.
- The public standalone index contains workflow JSON, source, tests, scripts,
  and documentation only. It contains no model weights, downloaded custom-node
  source, input/output assets, virtual environment, caches, logs, private-key
  filenames, or license file.
- Model and node licenses remain upstream concerns. This repository has no
  `LICENSE` file and grants no license merely by being public.
- Direct GitHub transport was unreliable during the audit. Any push/fetch
  workaround is command-scoped to the existing loopback proxy; no persistent
  repository or global proxy configuration is set. No OAuth/GCM token or
  private-key value was exposed by these audit commands. This report contains
  no device code or credential value.

## Residual Issues

- The local `INSTALL_STATUS.txt` describes ComfyUI `0.26.2` and an old DLL
  blocker. It is stale relative to the running `0.28.0` service and was
  deliberately preserved as local untracked data.
- Local runtime/cache debris remains by design. It is ignored or untracked and
  is excluded from the public repository.
- `docs/superpowers/2026-07-17-pose-controlled-pixel-animation-deployment-audit.md`
  records an older `130 passed` result. This report's fresh `227 passed` result
  supersedes that historical test count; the older record remains for history.
- The README requires `curl.exe` but does not state the `>= 7.71` version needed
  by the node installer's `--retry-all-errors` option. The audited host has curl
  `8.14.1`, so this is a documentation portability issue, not a current
  deployment failure.
- The README audit/export PowerShell example checks the native exit code only
  after later commands rather than immediately after each exporter. A failed
  exporter should be treated as fatal even if the subsequent diff is clean.
- Current service output is not redirected to a current log file. Live endpoint
  and history checks passed, but persistent runtime log capture would improve
  later incident diagnosis.
- Sword/hand/face continuity remains the material visual limitation. It does
  not invalidate the technical deployment, but it prevents a production-ready
  action-continuity claim.

## Reproduce Key Checks

```powershell
Set-Location 'E:\ComfyUI-GameAsset-Workflows'
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m compileall -q game_asset_api scripts
& 'E:\ComfyUI\.venv\Scripts\python.exe' -m pytest tests\game_asset_api_test -q
& 'E:\ComfyUI\.venv\Scripts\python.exe' scripts\audit_repository.py
git diff --check
git status --short --branch
```

Model hashes must be checked against `game_asset_api/model_manifest.py` and the
three Wan values in `README.md`. Live discovery validation requires the local
ComfyUI service at `http://127.0.0.1:8188`; it does not require exposing either
service beyond loopback.
