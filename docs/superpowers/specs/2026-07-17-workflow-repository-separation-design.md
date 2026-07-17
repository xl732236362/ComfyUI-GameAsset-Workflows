# ComfyUI Workflow Repository Separation Design

## Goal

Separate the game-asset workflow toolkit from the ComfyUI source checkout so
official ComfyUI updates and custom workflow development have independent Git
histories. The public standalone repository will be named
`ComfyUI-GameAsset-Workflows`.

## Repository Boundaries

`E:\ComfyUI` remains the runtime installation and tracks only the official
ComfyUI repository:

```text
origin = https://github.com/Comfy-Org/ComfyUI.git
```

`E:\ComfyUI-GameAsset-Workflows` becomes the sole source of truth for the
custom workflow toolkit and points to the new public, non-fork GitHub
repository. The existing `ComfyUI-CustomWorkFlow` fork remains unchanged.

The standalone repository contains:

```text
workflows/             Importable ComfyUI workflow JSON files
game_asset_api/        Workflow builders, API, jobs, and post-processing
scripts/               Install, export, run, and pose-authoring commands
tests/                 Unit and deployment tests
docs/                  Model, node, deployment, and tuning documentation
deploy.ps1             Idempotent deployment entry point
pyproject.toml         Python dependencies and test configuration
README.md              Setup, deployment, and usage instructions
.gitignore             Runtime and secret exclusions
```

The repository never contains model weights, downloaded custom-node source,
private keys, virtual environments, caches, user input, generated output, or
ComfyUI source files. Model and node dependencies are represented only by
pinned manifests containing source URLs, revisions, sizes, and SHA-256 hashes.

The initial workflow set is explicit:

- `pixel_character_design_api.json`
- `pixel_character_action_api.json`
- `pose_controlled_pixel_action_api.json`
- `video_wan2_2_5B_ti2v.json`
- `wan2_2_5b_dual_balanced.json`

## Deployment Data Flow

The supported entry point is:

```powershell
.\deploy.ps1 -ComfyRoot E:\ComfyUI
```

The deployment performs these steps in order:

1. Validate the ComfyUI root, Python environment, and expected directories.
2. Export and validate the workflow JSON files from the standalone source.
3. Atomically publish workflows to `user\default\workflows`.
4. Install pinned custom-node archives without replacing unmanaged folders.
5. Download missing models from the preferred mirror, resume partial files,
   and publish each model only after byte-count and SHA-256 verification.
6. Query the local ComfyUI `/object_info` endpoint for every required node,
   loader filename, and tuning option.
7. Run a low-cost pose smoke test and write a local deployment report.

Deployment uses file copies rather than symlinks so the runtime works without
Windows developer mode and remains portable. Re-running deployment with the
same source and dependency versions produces no material changes.

## Migration Sequence

Migration is deliberately ordered to avoid losing the currently verified
toolkit:

1. Create the standalone local repository and its restrictive `.gitignore`.
2. Copy only the approved workflow-toolkit files from `E:\ComfyUI`.
3. Adapt paths so every installer and runner accepts an explicit ComfyUI root.
4. Add deployment tests and run the complete game-asset regression suite.
5. Deploy from the standalone repository back into `E:\ComfyUI` and compare
   workflow hashes with the current runtime copies.
6. Create and push the public GitHub repository.
7. Only after the pushed repository and deployment are verified, restore the
   ComfyUI `origin` to the official URL and switch its checkout to an official
   branch or release.

Before changing the ComfyUI checkout, preserve its tracked `README.md` change
and inventory all untracked runtime files. No model, cache, input, output,
wheelhouse, log, or launcher file is deleted by this migration. The existing
custom branch remains available locally until the standalone repository has
been verified and the user explicitly approves later cleanup.

## Failure Handling

- A standalone-repository or test failure leaves the ComfyUI Git configuration
  unchanged.
- Workflow publication writes a temporary file and replaces the destination
  only after JSON validation.
- A failed model hash retains the resumable partial file but never replaces a
  valid destination.
- An existing custom-node directory without the expected revision marker stops
  deployment instead of being overwritten.
- A missing or unhealthy ComfyUI server fails discovery and smoke validation
  with an actionable error.
- GitHub creation, authentication, or push failure leaves the complete local
  standalone repository intact and does not trigger ComfyUI cleanup.

## Verification

The migration is complete only when all of these checks pass:

- The standalone Git index contains only the approved toolkit paths.
- Existing `game_asset_api` unit tests pass from the standalone repository.
- Deployment tests prove path validation, atomic workflow publication,
  idempotency, and exclusion of runtime artifacts.
- Two consecutive deployments produce identical tracked workflow hashes.
- ComfyUI discovers all five workflow JSON files, pinned custom nodes, and all
  manifest-listed models.
- A two-frame RGBA smoke action and an eight-frame 128-pixel xianxia sword
  attack produce the expected frames, sprite sheet, and GIF preview.
- The ComfyUI remote resolves to the official repository.
- The standalone remote resolves to the new public non-fork repository.

## Scope Boundaries

This migration does not delete the existing GitHub fork, publish model files,
rewrite the official ComfyUI project history, redesign the generation graphs,
or add new animation models. It changes repository ownership and deployment
packaging while preserving the currently verified runtime behavior.
