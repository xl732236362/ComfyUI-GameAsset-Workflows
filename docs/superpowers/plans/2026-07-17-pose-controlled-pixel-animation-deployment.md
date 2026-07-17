# Pose-Controlled Pixel Animation Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install and validate a local SDXL OpenPose + IP-Adapter Plus workflow that produces transparent, pose-ordered pixel character action frames on the existing 16 GB ComfyUI installation.

**Architecture:** Reuse SDXL Base, Pixel Art XL, and BiRefNet. Install pinned official node archives and three hash-pinned models into `E:\ComfyUI`, then submit one 512-square pose-controlled API graph per frame and compose the verified RGBA outputs into a sprite sheet.

**Tech Stack:** Python 3.13, ComfyUI 0.26.2, SDXL, ControlNet OpenPoseXL2, ComfyUI IPAdapter Plus, Pillow, pytest, curl, GitHub API archives, Hugging Face mirror.

---

## File Structure

- Modify: `game_asset_api/model_manifest.py` - add the three pose-workflow model specifications.
- Modify: `scripts/install_game_asset_models.py` - accept an explicit deployment root.
- Modify: `tests-unit/game_asset_api_test/test_model_manifest.py` - pin filenames, bytes, hashes, and installer root behavior.
- Create: `game_asset_api/node_manifest.py` - pin and safely install official node source archives.
- Create: `scripts/install_pose_workflow_nodes.py` - install node archives and their requirements into a selected ComfyUI root.
- Create: `tests-unit/game_asset_api_test/test_node_manifest.py` - verify archive layout and safe publication.
- Create: `game_asset_api/pose_workflow.py` - build one pose-controlled SDXL/IP-Adapter API graph.
- Create: `scripts/export_pose_controlled_workflow.py` - export the representative graph artifact.
- Create: `scripts/create_sword_attack_pose_sequence.py` - write deterministic OpenPose-style attack control images.
- Create: `scripts/run_pose_controlled_action.py` - submit the pose sequence, collect frames, and create the sprite sheet.
- Create: `tests-unit/game_asset_api_test/test_pose_workflow.py` - validate graph wiring, deterministic poses, and output composition.
- Create: `user/default/workflows/pose_controlled_pixel_action_api.json` - importable API workflow artifact.

### Task 1: Extend The Verified Model Installer

- [ ] Add failing assertions for `OpenPoseXL2.safetensors`, `ip-adapter-plus_sdxl_vit-h.safetensors`, and `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` with the exact sizes and SHA-256 values from the approved design.
- [ ] Run `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit\game_asset_api_test\test_model_manifest.py -q` and confirm the new filenames fail lookup.
- [ ] Add the three `ModelSpec` entries using `https://hf-mirror.com/.../resolve/main/...` URLs and add `--root` parsing to `scripts/install_game_asset_models.py`.
- [ ] Re-run the focused test and confirm it passes.
- [ ] Commit with `feat: add pose workflow model manifest`.

### Task 2: Install Pinned Node Archives Safely

- [ ] Write tests that create an in-memory ZIP with one top-level source directory, verify extraction into `custom_nodes/<name>`, reject path traversal, and preserve an already matching `.codex-source-revision` marker.
- [ ] Run `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit\game_asset_api_test\test_node_manifest.py -q` and confirm import failure.
- [ ] Implement `NodeSpec(name, archive_url, revision)`, safe ZIP-member validation, atomic directory publication, and a revision marker. Pin `comfyui_controlnet_aux` to `e8b689a513c3e6b63edc44066560ca5919c0576e` and `ComfyUI_IPAdapter_plus` to `a0f451a5113cf9becb0847b92884cb10cbdec0ef`.
- [ ] Add a CLI that accepts `--root`, downloads official GitHub API archives, installs both repositories, and runs the selected Python interpreter with `-m pip install -r requirements.txt` when present.
- [ ] Run focused tests and commit with `feat: add pinned pose node installer`.

### Task 3: Download And Verify Deployment Assets

- [ ] Run the node installer against `E:\ComfyUI` using `E:\ComfyUI\.venv\Scripts\python.exe`.
- [ ] Run the model installer against `E:\ComfyUI`; preserve `.part` files across transient network failures.
- [ ] Verify all six game-asset model specs with `verify_file` and record installed byte counts and SHA-256 values.
- [ ] Confirm the two custom-node revision marker files match their pinned commits.

### Task 4: Build The Pose-Controlled API Graph Under TDD

- [ ] After node installation, inspect the installed node mappings and running `/object_info` schemas for the IP-Adapter loader/application nodes and core ControlNet nodes.
- [ ] Write a failing graph test requiring SDXL + Pixel Art XL, pose `LoadImage`, `ControlNetLoader(OpenPoseXL2)`, `ControlNetApplyAdvanced`, IP-Adapter Plus with ViT-H clip vision, KSampler, BiRefNet, `InvertMask`, nearest-exact scaling, and `SaveImage`.
- [ ] Run `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit\game_asset_api_test\test_pose_workflow.py -q` and confirm import failure.
- [ ] Implement the smallest graph matching the installed schemas and export `user/default/workflows/pose_controlled_pixel_action_api.json`.
- [ ] Validate every class type against `/object_info` and every loader filename against its advertised options.
- [ ] Run focused tests and commit with `feat: add pose-controlled pixel action workflow`.

### Task 5: Create And Run Deterministic Pose Sequences

- [ ] Write failing tests for two-frame smoke and eight-frame sword-attack pose sequences, requiring unique chronological PNGs at 512 by 512.
- [ ] Implement deterministic black-background OpenPose-style control images with anticipation, wind-up, contact, follow-through, and recovery phases.
- [ ] Write failing tests for sequential graph submission and row-major RGBA sheet composition.
- [ ] Implement the runner with fixed seeds, one GPU job at a time, exact output count, alpha validation, and 64/128-pixel output selection.
- [ ] Run focused tests and commit with `feat: run pose-controlled sprite actions`.

### Task 6: Restart And Validate ComfyUI Discovery

- [ ] Stop only the process listening on port 8188, preserving its current command line.
- [ ] Restart ComfyUI from `E:\ComfyUI` with `HF_ENDPOINT=https://hf-mirror.com` and the existing loopback/port arguments.
- [ ] Poll `/system_stats` until healthy, then query `/object_info` for DWPose/OpenPose, IP-Adapter, ControlNet, BiRefNet, and compositor nodes.
- [ ] Verify the three new model filenames appear in the relevant loader option lists.

### Task 7: Live Smoke And Full Sword-Attack Validation

- [ ] Generate a two-pose 64-pixel smoke sequence and verify two RGBA frames, alpha foreground/background, and a 128 by 64 sheet.
- [ ] Generate an eight-pose 128-pixel xianxia sword attack and verify eight RGBA frames, chronological pose mapping, and a 384 by 384 sheet.
- [ ] Build animated GIF previews and inspect the reference, control poses, output strip, silhouette stability, weapon continuity, camera lock, and alpha edges.
- [ ] Tune only ControlNet strength, IP-Adapter weight, sampler seed/steps, or prompts one variable at a time; retain the best evidence-backed result.

### Task 8: Final Regression And Deployment Audit

- [ ] Run `E:\ComfyUI\.venv\Scripts\python.exe -m pytest tests-unit\game_asset_api_test -q`.
- [ ] Verify node revisions, model byte counts and SHA-256 values, workflow JSON parseability, `/object_info` discovery, and live output paths.
- [ ] Confirm no model, cache, input, or output artifact is staged in Git.
- [ ] Record the working workflow path, model locations, node revisions, final test job, visual verdict, and remaining operational limits.

## Plan Self-Review

- Spec coverage: Tasks 1-3 cover pinned sources and deployment; Tasks 4-5 create the reusable workflow and action runner; Tasks 6-7 prove runtime discovery and visual output; Task 8 audits every requested deliverable.
- Placeholder scan: all sources, revisions, filenames, commands, outputs, and validation gates are explicit; no deferred implementation markers remain.
- Type consistency: `ModelSpec`, `NodeSpec`, pose graph filenames, output sizes, and node/model identifiers are used consistently across installation, export, execution, and validation.
