# Integration Notes — StyleSense

## How Frontend Talks to Backend

1. User uploads content + style image via React UI
2. Frontend sends POST /api/recommend with user scenario constraints
3. Backend returns recommended method (fast / optimization / hybrid)
4. User clicks "Run" or "Compare Both"
5. Frontend sends POST /api/stylize with method + images
6. Backend runs NST engine(s), returns stylized image + metrics JSON
7. Frontend displays side-by-side results + benchmark data

## Shared Data Contracts

### /api/stylize Request
```json
{
  "method": "fast | optimization | both",
  "content_image": "<base64 or file path>",
  "style_image": "<base64 or file path>",
  "resolution": 512
}
```

### /api/stylize Response
```json
{
  "output_image": "<base64>",
  "runtime_ms": 120,
  "style_loss": 0.045,
  "content_loss": 0.012
}
```

### /api/recommend Request
```json
{
  "scenario": "real-time | quality-first | batch",
  "hardware": "gpu | cpu",
  "time_constraint_ms": 150
}
```

### /api/recommend Response
```json
{
  "recommended_method": "fast",
  "reason": "Real-time constraint < 150ms suits Fast NST",
  "expected_runtime_ms": 120
}
```
