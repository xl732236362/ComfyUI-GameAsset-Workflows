# Pose-Controlled Pixel Animation Deployment Audit

## Deployment

- Audited: `2026-07-17T09:54:56+08:00`
- ComfyUI root: `E:\ComfyUI`
- Runtime URL: `http://127.0.0.1:8188`
- Workflow: `E:\ComfyUI\user\default\workflows\pose_controlled_pixel_action_api.json`
- Preferred model source: `https://hf-mirror.com`
- IP-Adapter default: weight `0.65`, weight type `style transfer`
- OpenPose ControlNet default: strength `0.9`

The runtime workflow and checked artifact had matching SHA-256
`e70f1dc499c8077081696d0146b5a607f9305daa0c40ae5c0e0aff6dffc213f3`.
ComfyUI `/userdata` returned the workflow, and `/object_info` discovered every
node class and loader filename used by the graph.

## Verified Dependencies

All six model files matched the sizes and SHA-256 values pinned in the design.
The installed custom-node revisions were:

- `comfyui_controlnet_aux`: `e8b689a513c3e6b63edc44066560ca5919c0576e`
- `ComfyUI_IPAdapter_plus`: `a0f451a5113cf9becb0847b92884cb10cbdec0ef`

## Live Result

- Job: `pose-xianxia-sword-attack-v1-style`
- Output: `E:\ComfyUI\output\game_assets\pose-xianxia-sword-attack-v1-style\pose_action`
- Frames: eight chronological `128x128` RGBA PNGs
- Sprite sheet: `384x384` RGBA PNG
- Preview: eight-frame `512x512` GIF, approximately `1.01` seconds
- Alpha: every frame had transparent corners and non-empty foreground alpha

OpenPose produced recognizable anticipation, wind-up, contact, follow-through,
and recovery body poses. IP-Adapter retained the white/azure costume, long dark
hair, palette, and general silhouette. `style transfer` improved pose freedom
over `linear`, which kept the reference sword almost vertical.

Weapon continuity is not production-exact. OpenPose contains body joints but no
weapon geometry, so the blade can change direction or length between independently
sampled frames. A phase-specific text-prompt experiment did not fix the contact
direction and was reverted. Exact sword arcs require an authored lineart/canny or
segmentation control channel, or a temporal animation workflow; those additions
remain outside this deployment's approved scope.

## Verification

- Unit regression: `130 passed`
- Python compilation: passed for `game_asset_api` and `scripts`
- Git whitespace check: passed
- Independent code review: no Critical findings; Important findings resolved
- Runtime discovery, model hashes, node revisions, workflow API access, RGBA
  dimensions, alpha ranges, sheet layout, and GIF frame metadata: passed

Models, node installations, caches, inputs, and generated outputs remain runtime
assets and are not part of the Git commit.
