# High-Level Architecture — StyleSense

## Modules
1. **Custom Feature Extractor** — VGG-like CNN, trained from scratch
2. **Optimization-Based NST Engine** — iterative pixel optimization
3. **Fast NST Generator** — feed-forward conv network
4. **Evaluation & Benchmark Module** — metrics CSV/JSON + plots
5. **Recommendation Engine** — rules-based, uses benchmark + user constraints
6. **Web UI (Frontend)** — MERN stack, upload/compare/recommend
7. **Backend API Layer** — endpoints: /stylize, /benchmark, /recommend

## Data Flow
User Upload → Scenario Select → Recommendation → NST Run(s) → Metrics → Display

## Module Owners
| Module                    | Owner     |
|---------------------------|-----------|
| Custom Feature Extractor  | Shubhansh |
| Optimization-Based NST    | Shubhansh |
| Fast NST Generator        | Shubhansh |
| Evaluation & Benchmark    | Both      |
| Recommendation Engine     | Manas     |
| Web UI / Frontend         | Manas     |
| Backend API               | Shubhansh |
