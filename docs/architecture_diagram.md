# Architecture Diagram — StyleSense

## System Overview (Mermaid)

```mermaid
graph TD
    A[User Browser] -->|Upload Content + Style Image| B[Web UI - React]
    B -->|Select Scenario| C[Recommendation Engine]
    C -->|Suggested Method| B
    B -->|Run NST / Compare Both| D[Backend API - Flask/Node]
    D -->|Stylize Request| E[Optimization-Based NST Engine]
    D -->|Stylize Request| F[Fast NST Generator]
    E -->|Uses| G[Custom VGG-like Extractor]
    F -->|Uses| G
    D -->|Log Metrics| H[Evaluation & Benchmark Module]
    H -->|CSV/JSON + Plots| B
    B -->|Display Results + Metrics| A
```

## Module Descriptions

| Module | Tech | Owner |
|---|---|---|
| Web UI | React (MERN) | Manas |
| Backend API | Flask / Node.js | Shubhansh |
| Custom Feature Extractor | PyTorch CNN | Shubhansh |
| Optimization-Based NST | PyTorch | Shubhansh |
| Fast NST Generator | PyTorch | Shubhansh |
| Evaluation & Benchmark | Python (pandas, matplotlib) | Both |
| Recommendation Engine | Python (rules-based) | Manas |

## API Endpoints (planned)

| Endpoint | Method | Description |
|---|---|---|
| /api/stylize | POST | Run NST (method: opt/fast/both) |
| /api/benchmark | POST | Run benchmark on test set |
| /api/recommend | POST | Get method recommendation |
| /api/status | GET | Server health check |
