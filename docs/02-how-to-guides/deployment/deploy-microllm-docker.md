# Deploy MicroLLM with Docker

Refer to the
[Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
reference before you do this procedure.

## Preconditions

1. Make sure that Docker Engine is active on the host.
2. Make sure that your terminal is in the MicroLLM repository root directory.
3. Make sure that port `8000` is free on the host.
4. Prepare an API key value for `MICROLLM_API_KEY` before you deploy on a network.

## Safety Instructions

**CAUTION:** Set `MICROLLM_API_KEY` before you expose port `8000` to a network.
Unauthenticated access permits remote clients to start heavy tasks.

## Procedure

1. Make the Docker container image from `Dockerfile`:

   ```bash
   docker build -t microllm:0.4.1 .
   ```

2. Make a Docker volume for `MICROLLM_CACHE_DIR` at `/app/.cache`:

   ```bash
   docker volume create microllm-cache
   ```

3. Start container `microllm-server` on port `8000` with `MICROLLM_API_KEY`:

   ```bash
   docker run -d \
     --name microllm-server \
     --restart unless-stopped \
     -p 8000:8000 \
     -e HOST=0.0.0.0 \
     -e PORT=8000 \
     -e MICROLLM_CACHE_DIR=/app/.cache \
     -e MICROLLM_API_KEY="replace-with-strong-key" \
     -e MICROLLM_MULTILINGUAL_STRATEGY=auto \
     -e MICROLLM_NEURAL_MODEL=minilm \
     -v microllm-cache:/app/.cache \
     microllm:0.4.1
   ```

4. If port `8000` is in use, connect an alternative host port to container port `8000`:

   ```bash
   docker run -d \
     --name microllm-server \
     -p 8080:8000 \
     -e MICROLLM_API_KEY="replace-with-strong-key" \
     -v microllm-cache:/app/.cache \
     microllm:0.4.1
   ```

5. If you must have the Gradio interface on port `7860`, install `micro-jev[gradio]` and start `microllm studio`:

   ```bash
   microllm studio --port 7860
   ```

**NOTE:** The container `Dockerfile` sets `USER appuser` and binds `microllm serve` to `0.0.0.0:8000`.

## Verification Check

1. Examine container status with Docker:

   ```bash
   docker ps --filter name=microllm-server
   ```

2. Make sure that `/healthz` outputs `ok`:

   ```bash
   curl -fsS http://127.0.0.1:8000/healthz
   ```

3. Make sure that `/v1/models` outputs HTTP `200`:

   ```bash
   curl -fsS http://127.0.0.1:8000/v1/models
   ```

4. Send a test request to `/v1/systemone` with your Bearer token:

   ```bash
   curl -fsS -X POST http://127.0.0.1:8000/v1/systemone \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer replace-with-strong-key" \
     -d '{"state":"Payment charged twice","questions":{"q1":{"type":"noul","instructions":"Billing issue?"}}}'
   ```

## Rollback Procedure

If container `microllm-server` fails health checks, do these steps:

1. Stop container `microllm-server`:

   ```bash
   docker stop microllm-server
   ```

2. Remove failed container `microllm-server`:

   ```bash
   docker rm microllm-server
   ```

3. Start the previous container image tag with volume `microllm-cache` on port `8000`:

   ```bash
   docker run -d \
     --name microllm-server \
     -p 8000:8000 \
     -e MICROLLM_CACHE_DIR=/app/.cache \
     -e MICROLLM_API_KEY="replace-with-strong-key" \
     -v microllm-cache:/app/.cache \
     microllm:previous
   ```

4. Query `/healthz` and make sure that the previous container outputs `ok`.

## Related Documentation

- Reference: [Environment and Model Profiles](../../03-reference/configuration/environment-and-model-profiles.md)
- Reference: [CLI Commands and Exit Codes](../../03-reference/cli-and-scripts/cli-commands-and-exit-codes.md)
- Reference: [Error and Alert Matrix](../../03-reference/troubleshooting-codes/error-and-alert-matrix.md)
- Explanation: [Dual-Gear System-1 Topology](../../04-explanation/architecture-overview/dual-gear-system1-topology.md)
