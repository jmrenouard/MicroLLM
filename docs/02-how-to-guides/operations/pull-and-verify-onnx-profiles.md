# Pull and Verify ONNX Profiles

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that package `micro-jev` is installed in your active Python environment.
2. If you download ONNX files from `Hugging Face Hub`, install `huggingface_hub` and `onnxruntime`.
3. Make sure that you have write access to `MICROLLM_CACHE_DIR` or `~/.cache/microllm`.

## Safety Instructions

**CAUTION:** Make a backup copy of custom files in `MICROLLM_CACHE_DIR` before you pull a profile.
A new pull replaces `profile_manifest.json` and `model_quantized_int8.npz`.

## Procedure

1. Show available hardware profiles and current status:

   ```bash
   microllm list
   ```

2. If you use a custom cache directory, set `MICROLLM_CACHE_DIR` first:

   ```bash
   export MICROLLM_CACHE_DIR=/var/cache/microllm
   ```

3. Pull profile `fast-cpu-int8`:

   ```bash
   microllm pull fast-cpu-int8
   ```

4. Pull profile `multilingual`:

   ```bash
   microllm pull multilingual
   ```

5. Pull profile `guardrail`:

   ```bash
   microllm pull guardrail
   ```

6. Pull profile `rlcd2-decision-head`:

   ```bash
   microllm pull rlcd2-decision-head
   ```

7. If network access is permitted, pull full ONNX weights from `Hugging Face Hub`:

   ```bash
   microllm pull fast-cpu-int8 --from-hub
   microllm pull multilingual --from-hub
   microllm pull guardrail --from-hub
   ```

**NOTE:** Flag `--download` gets only `hf_config.json`, while `--from-hub` gets `onnx/model_quantized.onnx`.

## Verification Check

1. Show installed profiles and make sure that `installed` is `true` for each profile:

   ```bash
   microllm list
   ```

2. Do a SHA-256 checksum check on all 4 installed profiles with `verify_profile`:

   ```bash
   python -c "import json; from microllm.registry import verify_profile; print(json.dumps([verify_profile(p) for p in ('fast-cpu-int8','multilingual','guardrail','rlcd2-decision-head')], indent=2))"
   ```

3. Make sure that each output dictionary shows `"verified": true` and `"sha256_match": true`.

4. If `microllm serve` is active on port `8000`, query `/v1/profiles`:

   ```bash
   curl -fsS http://127.0.0.1:8000/v1/profiles
   ```

## Rollback Procedure

If a checksum check fails or a profile manifest is damaged, do these steps:

1. Remove the damaged profile folder in `MICROLLM_CACHE_DIR`:

   ```bash
   rm -rf "${MICROLLM_CACHE_DIR:-$HOME/.cache/microllm}/fast-cpu-int8"
   ```

2. If a backup is available, put the previous profile folder back in `MICROLLM_CACHE_DIR`.

3. If no backup is available, pull the profile offline again:

   ```bash
   microllm pull fast-cpu-int8
   ```

4. Do a `verify_profile` check and make sure that `"verified"` is `true`.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
