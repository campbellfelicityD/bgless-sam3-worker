# bgless-sam3-worker

RunPod Serverless worker for video background removal with **SAM 3** (concept-prompt
video segmentation) + **MatAnyone** (alpha matting refinement).

The image is built by `.github/workflows/build.yml` and pushed to
`ghcr.io/<owner>/bgless-sam3:{latest,sha}`.

Backed by the bgless monorepo `transcoder/runpod_sam3.mjs` dispatcher.

See `docs/api-contract.md` for the input/output JSON schema.
