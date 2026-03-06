# Project Requirements — StyleSense (T59)

## Problem Statement
Optimization-based NST is slow; Fast NST is less flexible.
No systematic comparison exists with use-case-driven recommendations.

## Goals
- Train custom VGG-like CNN from scratch (no pretrained VGG)
- Implement both NST pipelines using same extractor
- Unified evaluation framework (quality vs speed vs resources)
- Interactive platform with recommendation engine

## Non-Goals
- No commercial app deployment
- No real-time 4K video
- No GAN/diffusion-based NST
- No large-scale user studies

## Acceptance Criteria
- [ ] Both pipelines run without crash on 100+ content-style pairs
- [ ] Custom extractor used for ALL perceptual losses (no VGG16/19)
- [ ] Benchmarks at >= 3 resolutions generated
- [ ] Recommendation engine supports >= 3 scenario types
