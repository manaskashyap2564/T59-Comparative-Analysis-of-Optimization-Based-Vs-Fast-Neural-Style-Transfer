# StyleSense Frontend (React)

## Setup (Day 4 — Structure only)

```bash
npx create-react-app stylesense-ui
cd stylesense-ui
npm install axios react-dropzone recharts
```

## Planned Components (Week 6)
- `ImageUploader` — drag & drop content + style images
- `MethodSelector` — optimization / fast / both
- `ResultViewer` — side by side output + metrics
- `RecommendationPanel` — scenario input → suggestion
- `BenchmarkChart` — recharts integration

## API Endpoints to Connect
- POST /api/recommend
- POST /api/stylize
- POST /api/benchmark
- GET  /api/status
