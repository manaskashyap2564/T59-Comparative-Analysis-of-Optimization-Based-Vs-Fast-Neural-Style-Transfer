# 🎓 B.Tech : CSE (AI/ML and IoT) (III YEAR – VI SEM) (2025-2026)

## 🏛️ DEPARTMENT OF COMPUTER ENGINEERING & APPLICATIONS

**🌟 GLA University**  
17km Stone, NH-19, Mathura-Delhi Road  
P.O. Chaumuhan, Mathura – 281406  
(Uttar Pradesh) India

---

## 📋 Project Title : Comparative Analysis of Optimization-Based vs Fast Neural Style Transfer

**👨‍💼 Team Lead** : Manas Kashyap  
**🆔 UR** : 2315510114

**👥 Team Member** : Shubhansh Gupta  
**🆔 UR** : 23155101204

**🧑‍🏫 Mentor Name** : Dr. Ram Manohar Nisarg NS  
**✍️ Signature** :

---

## 📝 Project Synopsis : Comparative Analysis of Optimization-Based vs Fast Neural Style Transfer

A dual-pipeline neural style transfer system that compares optimization-based and fast feed-forward methods on quality versus speed. An interactive platform that uses a custom-trained CNN backbone to benchmark both approaches and recommend the best method for each use-case scenario.

---

## 0. 📑 Cover

- **📌 Project Title** : Comparative Analysis of Optimization-Based vs Fast Neural Style Transfer
- **🏷️ Team Name & ID** : StyleSense (T59)
- **🎓 Institute / Course** : GLA University / B.Tech : CSE (AI/ML and IoT)
- **📦 Version** : v0.1
- **📅 Date** : 06 Feb, 2026
- **📜 Revision History** :

| Version | Date | Author | Change |
|---------|------|--------|--------|
| v0.1 | 06 Feb, 2026 | Manas Kashyap | Initial Draft |
| v0.5 | 1 Mar, 2026 | StyleSense | Half Project Done |
| v1.0 | 5 Apr, 2026 | StyleSense | Full Project Done |

---

## 1. 🔍 Overview

### ⚠️ Problem Statement

Optimization-based Neural Style Transfer produces high-quality and flexible stylized images but is very slow and computationally expensive. Meanwhile, fast/feed-forward NST gives real-time results but is less flexible, often needs style-specific training, and can produce less consistent quality. This creates a practical gap where most solutions depend on pretrained VGG16/VGG19 and there is no systematic end-to-end comparison with clear, use-case-driven recommendations for real-world deployment.

### 🎯 Goal

Develop a complete Neural Style Transfer system that trains a custom VGG-like CNN feature extractor from scratch (no pretrained VGG), implements both optimization-based and fast/feed-forward NST pipelines using the same extractor, compares them under a unified evaluation framework (quality vs speed vs resources), and deploys an interactive platform where users can visually compare results, view performance metrics, and get recommended methods according to their use-case scenarios and also proper documentations for proper use of different use case scenarios.

### 🚫 Non-Goals

Develop a full-scale commercial photo-editing suite or social media app. Support production-scale real-time video stylization. Implement every advanced NST variant from the literature. Conduct large-scale user studies with hundreds of participants. Train on extremely large proprietary datasets.

### 💎 Value Proposition

The proposed system will make Neural Style Transfer deployment decisions more practical and reliable. By providing a controlled comparison of optimization-based vs fast NST using a custom-trained feature extractor, it will help engineers and researchers clearly understand the quality–speed trade-off through benchmarks and side-by-side results. This enables scenario-based recommendations (e.g., real-time mobile filters vs offline high-quality rendering) so users can confidently choose the right method for their specific use case.

---

## 2. 🎯 Scope and Control

### 2.1 ✅ In-Scope

- Design and training of a custom VGG-like CNN feature extractor from scratch (classification-based pretraining).
- Implementation of an optimization-based NST pipeline using the custom extractor for content and style losses.
- Implementation of a fast / feed-forward NST generator trained using the same extractor's perceptual losses.
- Creation of a unified evaluation suite:
  - ➢ Runtime, memory usage, style/content loss metrics, basic perceptual measures.
- Development of an interactive web-based platform to:
  - ➢ Upload/select content and style images.
  - ➢ Run and compare both NST methods.
  - ➢ Display metrics and visual outputs.
- Implementation of a recommendation engine that, based on user-specified constraints and scenarios, suggests the most suitable NST method (or hybrid strategy).
- High-level documentation, tutorials, and reports integrated into the platform.

### 2.2 ❌ Out-of-scope

- Production-grade mobile app deployment on app stores (only prototype-level or demo recommendations).
- Real-time high-resolution video stylization at 4K/60FPS (only limited demos at lower resolutions).
- Integration with external commercial platforms (e.g., Instagram, Snapchat, etc.).
- Extremely advanced NST variants (e.g., GAN-based style transfer, diffusion-based stylization), beyond a core representative fast NST baseline.
- Handling of 3D content or cross-modal style transfer (e.g., text-to-style).

### 2.3 📋 Assumptions

- Public image datasets (e.g., subsets of ImageNet/COCO/WikiArt) are accessible for training and evaluation.
- Laboratory hardware (at least one GPU-enabled machine) is available for training and benchmarking.
- Python deep learning frameorks (PyTorch/TensorFlow) and web frameworks (e.g. React) work as expected.
- Team members can contribute ~10-15 hours per week consistently.
- Target users (developers, students, mentors) can interact with a browser-based interface in English.

### 2.4 ⚡ Constraints

- **⏱️ Time** : Approx. 8-week active project duration
- **💻 Resources** : Limited GPU resources (no large multi-GPU clusters). Training and experiments must be optimized.
- **👥 Team Size** : Small team (2 core developers + 1 mentor). Tasks need clear prioritization.
- **🔧 Scope** : Focus on core NST comparison and recommendation; UI and non-critical features must stay lightweight.

### 2.5 🔗 Dependencies

- **💿 Software** :
  - ➢ PyTorch or TensorFlow for model implementation.
  - ➢ Image processing libraries (Pillow/OpenCV).
  - ➢ Web framework (MERN).
- **📊 Data** : Public image datasets (content + style images).
- **🖥️ Hardware** : GPU-enabled system for model training and benchmarks.
- **🤝 Stakeholder input** : Guidance and periodic review from mentor (Dr. Ram Manohar Nisarg NS).

### 2.6 ✔️ Acceptance Criteria (Signoff Scenarios)

- **🔄 DUAL PIPELINE FUNCTIONALITY** : GIVEN a content and style image are uploaded, WHEN the user selects "Compare Both", THEN the system must generate outputs from both optimization-based NST and fast NST methods without errors and display them side-by-side.

- **🧠 CUSTOM EXTRACTOR USAGE** : GIVEN the NST pipelines are running, WHEN feature extraction is invoked during style/content loss computation, THEN the system uses the custom trained VGG-like model, not any pretrained VGG16/19.

- **📊 BENCHMARK GENERATION** : GIVEN a predefined benchmark set of content and style images, WHEN the experiment runner is launched, THEN it should produce a metrics report (runtime, losses) comparing both methods across at least two resolutions.

- **💡 RECOMMENDATION ENGINE** : GIVEN a user specifies constraints (e.g., "real-time mobile filter"), WHEN the "Get Recommendation" action is triggered, THEN the platform must suggest a method (fast/optimization/hybrid) with a short explanation.

- **🔒 INTERACTIVE DEMO STABILITY** : GIVEN typical usage (multiple stylizations and comparisons), WHEN the platform runs for at least 30 minutes, THEN no critical crashes should occur, and logs/outputs should be captured correctly.

---

## 3. 👥 Stakeholders and RACI

### 🤝 Stakeholders

StyleSense team members (see Section 4), Project Mentor (faculty advisor), and University administration. This solution ultimately serves ML students, developers, researchers, designers testing NST methods.

### 📊 RACI Matrix (Key Activities)

| Activity | Responsible (R) | Accountable (A) | Consulted (C) | Informed (I) |
|----------|----------------|-----------------|---------------|--------------|
| Requirements & planning | Manas Kashyap | Manas Kashyap | Mentor | Team |
| Custom CNN design & training | Shubhansh Gupta | Shubhansh Gupta | Mentor | Team |
| Optimization-based NST implementation | Shubhansh Gupta | Shubhansh Gupta | Manas, Mentor | Team |
| Fast NST model design & training | Shubhansh Gupta | Shubhansh Gupta | Manas, Mentor | Team |
| Evaluation framework & benchmarking | Manas & Shubhansh | Manas Kashyap | Mentor | Department (summary) |
| Interactive UI & recommendation engine | Manas Kashyap | Manas Kashyap | Shubhansh, Mentor | Team |
| Backend/API & integration | Shubhansh Gupta | Manas Kashyap | Mentor | Team |
| Testing & validation | StyleSense Team | Manas Kashyap | Mentor | Department (final state) |
| Final review, demo & documentation | StyleSense Team | StyleSense Team | Mentor | Department/University |

---

## 4. 👨‍💻 Team and Roles

| Member | Role | Responsibilities | Key Skills | Availability (hr / week) | Contact |
|--------|------|-----------------|------------|--------------------------|---------|
| Manas Kashyap | Team Lead, AI/ML Engineer, Android Developer | Lead project planning and coordination; co-design NST architectures and evaluation protocol; implement/coordinate fast NST training pipeline and optimization-based NST integration; design and implement the interactive platform (frontend + integration hooks); implement recommendation logic based on benchmarks and rules | Python, PyTorch/TensorFlow, Android, APIs, basic frontend | 15 | manas.kashyap_cs.aiml23@gla.ac.in |
| Shubhansh Gupta | AI/ML Engineer, Full Stack Developer | Design and train a custom VGG-like feature extractor, implement content/style/TV losses and both NST pipelines, and build backend APIs to serve the models to the UI. | Python, PyTorch/TensorFlow, backend (Node.js/Python), web basics, ML | 15 | shubhansh.gupta_cs.aiml23@gla.ac.in |

---

## 5. 📅 Weekwise Plan and Assignments

### 🗓️ Week 1 (Feb 9 – Feb 15) – Requirements & Architecture

- ➢ Finalize problem statement, goals, and non-goals for NST comparison.
- ➢ Define architecture for: custom feature extractor, both NST pipelines, evaluation module, and interactive platform.
- ➢ **📦 Deliverables**: Requirement spec, high-level architecture diagram, task breakdown.

### 🗓️ Week 2 (Feb16 – Feb 22) – Custom Feature Extractor Implementation & Dataset Setup

- ➢ Implement VGG-like CNN architecture.
- ➢ Prepare dataset pipeline (content images for classifier pretraining).
- ➢ Start training the custom CNN as an image classifier.
- ➢ **📦 Deliverables**: Training script, initial training logs, preliminary model checkpoints.

### 🗓️ Week 3 (Feb 23 – Mar 1) - Feature Extractor Finalization & Optimization-Based NST Prototype

- ➢ Refine CNN training, choose best checkpoint.
- ➢ Freeze backbone and expose feature maps for selected layers.
- ➢ Implement style/content/TV losses and iterative optimization loop.
- ➢ Test NST on a small set of content/style pairs.
- ➢ **📦 Deliverables**: Working optimization-based NST prototype, sample results.

### 🗓️ Week 4 (Mar 2 – Mar 8) - Fast NST Model Design & Early Training

- ➢ Design feed-forward generator architecture.
- ➢ Integrate loss functions using custom extractor.
- ➢ Begin training single-style fast NST model on content dataset.
- ➢ **📦 Deliverables**: Generator training script, early stylization samples, initial comparison with optimization NST.

### 🗓️ Week 5 (Mar 9 – Mar 15) - Evaluation Framework & Extended Experiments

- ➢ Implement automated evaluation runner (looping over content/style sets).
- ➢ Collect runtime, loss, and basic perceptual metrics for both methods at different resolutions.
- ➢ Start drafting plots/tables.
- ➢ **📦 Deliverables**: CSV/JSON metrics, early benchmark plots, notes on trade-offs.

### 🗓️ Week 6 (Mar 16 – Mar 22) - Interactive Platform & Recommendation Engine

- ➢ Build core UI (upload content/style, choose method, display outputs).
- ➢ Integrate backend inference APIs for both NST methods.
- ➢ Implement basic recommendation rules based on user constraints (real-time vs quality, hardware, etc.).
- ➢ **📦 Deliverables**: Running local web demo, first version of recommendation logic.

### 🗓️ Week 7 (Mar 23 – Mar 29) - Testing, Optimization & Documentation

- ➢ Optimize model inference (batching, lightweight preprocessing) and application performance.
- ➢ Conduct functional, integration, and basic performance testing.
- ➢ Refine documentation: user guide, technical notes, scenario-based examples.
- ➢ **📦 Deliverables**: Test report, improved UI/UX, polished recommendations.

### 🗓️ Week 8 (Mar 30 – Apr 5) - Finalization & Demo

- ➢ Final benchmark runs and result consolidation.
- ➢ Prepare slides, final synopsis, and live demo scenario.
- ➢ Conduct full end-to-end demo for mentor (content+style → outputs + recommendation).
- ➢ **📦 Deliverables**: Final codebase, slide deck, final synopsis v1.0.

---

## 6. 👤 Users and UX

### 6.1 🎭 Personas

- **🔬 ML Researcher / Student**: Wants to study NST quality vs speed trade-offs and architecture effects.
- **📱 App Developer (Mobile/Web)**: Needs to pick the right NST method for a product (e.g., photo filter, AR app).
- **🎨 Designer / Digital Artist**: Wants high-quality stylized images; offline, time is less critical.
- **👨‍🏫 Educator / Mentor**: Wants to demonstrate NST concepts and trade-offs to students interactively.

### 6.2 🗺️ Top User Journeys

#### **🛤️ Journey 1 – Developer Choosing NST for Real-Time Filter**

- ➢ Developer opens the platform, selects "Mobile Real-Time Photo Filter" scenario.
- ➢ Uploads example content and style images.
- ➢ Clicks "Compare Both" to see outputs and runtimes.
- ➢ Platform recommends Fast NST, explains speed benefits vs small quality loss.
- ➢ Developer exports summary and uses guidance to implement in their app.

#### **🛤️ Journey 2 – Artist Creating High-Quality Posters**

- ➢ Designer uploads high-resolution content & style images.
- ➢ Selects "High-Quality Offline Artwork" as use case.
- ➢ Platform recommends optimization-based NST (possibly hybrid: fast preview + optimization final).
- ➢ Designer runs optimization-based method with more iterations, checks visual quality, and downloads final image.

#### **🛤️ Journey 3 – Student Learning NST Concepts**

- ➢ Student uploads simple images.
- ➢ Uses sliders to change style/content weights and optimization iterations.
- ➢ Observes how quality and runtime change.
- ➢ Reads documentation tab that explains differences and sees benchmark plots.

### 6.3 📖 User Stories

#### **📱 US1 – Real-Time Constraint**

"As a mobile app developer, I want the platform to recommend an NST method that can run under 100 ms per image, so that my photo filter feels responsive."

- ➢ GIVEN the user selects a "real-time mobile" constraint, WHEN they request a recommendation, THEN the platform suggests fast NST and provides approximate runtime and memory usage.

#### **🎨 US2 – Quality-First Workflow**

"As a digital artist, I want the most visually rich result, even if it is slower, so that my final artwork looks professional."

- ➢ GIVEN the user picks "maximum quality" with offline processing, WHEN they compare methods, THEN the platform highlights optimization-based NST as primary, with tuned iteration counts and sample results.

#### **📚 US3 – Learning and Comparison**

"As a student, I want to visually compare the outputs and metrics of both methods, so I can understand their trade-offs."

- ➢ GIVEN the student uploads content and style images, WHEN they click "Compare Both", THEN the platform shows side-by-side results, runtimes, and basic metrics along with explanatory notes.

---

## 7. 📊 Market and Competitors

### 7.1 🏢 Competitors

#### **🔧 Single-Method NST Demos/Libraries**

- ➢ Many GitHub repositories and tutorials implement either optimization-based NST or fast NST, but rarely both, and usually without systematic comparison.

#### **📲 Style Transfer Apps (e.g., Prisma-like apps)**

- ➢ Provide fast stylization with fixed styles, but do not expose underlying methods, trade-offs, or research-level control.

#### **🎓 Academic Implementations of Arbitrary-Style Transfer**

- ➢ Papers and code (e.g., AdaIN, WCT) show methods but focus on algorithmic novelty rather than decision-support for deployment.

### 7.2 🎯 Positioning

#### **✨ Our Differentiator**

- ➢ Uses a custom, self-trained VGG-like feature extractor instead of relying on standard pretrained VGG16/19.
- ➢ Provides two full NST pipelines under one framework (optimization-based + fast) with shared feature space.
- ➢ Integrates quantitative benchmarks, qualitative comparisons, and an interactive recommendation system into a single platform.
- ➢ Targets not only research but practical decision-making for deployment scenarios (mobile, desktop, offline batch, etc.).

---

## 8. 🎯 Objectives and Success Metrics

### **📌 O1: Implement Robust Dual NST Pipelines**

- ➢ **Target**: Both optimization-based and fast NST pipelines functional and stable for a variety of images.
- ➢ **📈 KPI**: ≥ 95% success rate (no crashes) across a test set of at least 100 content-style pairs.

### **📌 O2: Train a Custom Feature Extractor and Use It Consistently**

- ➢ **Target**: Classifier training achieves reasonable accuracy (e.g., >70% on validation subset), and the extractor is used for all perceptual losses.
- ➢ **📈 KPI**: No direct dependency on pretrained VGG; code-level verification; model accuracy thresholds met.

### **📌 O3: Provide Clear Quality vs Speed Benchmarks**

- ➢ **Target**: Generate benchmark tables/plots comparing runtime and losses at multiple resolutions.
- ➢ **📈 KPI**: At least 3 standard resolutions evaluated; metrics logged for both methods; benchmark report generated.

### **📌 O4: Implement a Working Interactive Platform with Recommendations**

- ➢ **Target**: Users can upload images, run both methods, view outputs, and receive method recommendations.
- ➢ **📈 KPI**: At least 3 distinct scenario types supported (real-time, offline quality-first, batch processing), with consistent recommendation logic.

### **📌 O5: User Understanding and Satisfaction**

- ➢ **Target**: Small internal user group (team + mentor + peers) finds the explanations and comparisons useful.
- ➢ **📈 KPI**: Qualitative feedback indicating improved understanding of NST trade-offs.

---

## 9. ⚙️ Key Features

### **🔴 Adaptive Dual NST Pipelines (Must)**

- ➢ Core functionality providing both optimization-based and fast NST, using custom extractor.
- ➢ **✅ Acceptance**: GIVEN valid content/style images, WHEN either method is selected, THEN a stylized image is produced without errors.

### **🔴 Unified Evaluation & Benchmarking (Must)**

- ➢ Automated scripts for quality/speed benchmarking across both methods.
- ➢ **✅ Acceptance**: GIVEN a predefined test set, WHEN the benchmark runner is executed, THEN a metrics file and plots are produced.

### **🔴 Interactive Visual Comparison (Must)**

- ➢ Web UI to display side-by-side outputs with metrics.
- ➢ **✅ Acceptance**: GIVEN completion of both stylizations, THEN the platform shows content, style, both results, and runtimes on a single screen.

### **🔴 Recommendation Engine (Must)**

- ➢ Rules-based engine suggesting NST method(s) per use case.
- ➢ **✅ Acceptance**: GIVEN user constraints (real-time, quality, hardware), WHEN they request recommendation, THEN an appropriate method is suggested with reasoning.

### **🟡 Documentation & Scenario Guides (Should)**

- ➢ Built-in tabs/pages explaining trade-offs and sample scenarios.
- ➢ **✅ Acceptance**: Users can access a "Documentation"/"Learn" section with method comparison and use-case examples.

### **🟢 Exportable Reports (Could)**

- ➢ Option to export brief comparison summaries as PDF or text.
- ➢ **✅ Acceptance**: GIVEN a completed comparison session, WHEN "Export Report" is clicked, THEN a short report is generated.

---

## 10. 🏗️ Architecture

The system follows a modular architecture:

### **🧠 Custom Feature Extractor Module**

- ➢ VGG-like CNN trained from scratch.
- ➢ Provides intermediate feature maps for content/style loss computation.

### **⚙️ Optimization-Based NST Engine**

- ➢ Implements iterative optimization over image pixels.
- ➢ Uses feature extractor outputs for loss calculations.

### **⚡ Fast NST Generator Module**

- ➢ Feed-forward convolutional network.
- ➢ Trained using the same loss definitions based on the custom extractor.

### **📊 Evaluation & Benchmark Module**

- ➢ Runs stylization on datasets, records metrics (runtime, losses, memory).
- ➢ Outputs CSV/JSON and plots.

### **💡 Recommendation Engine**

- ➢ Consumes benchmark data + user constraints.
- ➢ Outputs method recommendation, reasoning, and expected performance.

### **🖥️ Web UI / Frontend**

- ➢ Provides image upload, controls, view for outputs & metrics, documentation pages.

### **🔌 Backend / API Layer**

- ➢ Exposes endpoints for:
  - ▪ Stylization with method selection.
  - ▪ Running small benchmarks.
  - ▪ Generating recommendations.

### **🔄 Workflow (Example)**

- ➢ User uploads content & style → selects scenario → platform fetches recommendation → user runs either single method or "Compare Both" → backend calls NST engine(s) → outputs and metrics returned → UI displays results and explanation.

---

## 11. 💾 Data Design

We will log key data items:

### **🎨 Stylization Runs**

- ➢ {run_id, timestamp, method, content_id, style_id, resolution, runtime_ms, style_loss, content_loss, notes}

### **🖼️ Images Metadata**

- ➢ {image_id, type (content/style), source (dataset/local upload), resolution}

### **📈 Benchmark Summaries**

- ➢ {benchmark_id, date, resolution_set, hardware_info, aggregated_metrics}

### **👤 User Scenario Inputs (non-personal)**

- ➢ {scenario_id, time_constraint, hardware_type, quality_priority, flexibility_requirement}

No personal or sensitive data is required; only technical metadata and non-identifying usage patterns may be stored.

---

## 12. 📐 Technical Workflow Diagrams

### 12.1 🔄 State Transition Diagram

### 12.2 🔁 Sequence Diagram

### 12.3 👤 Use Case Diagram

### 12.4 📊 Data Flow Diagram

### 12.5 🗂️ ER Diagram

### 12.6 ⚙️ Technical Workflow Diagram

### 12.7 🏛️ Work Architecture Diagram

---

## 13. ✅ Quality (Non Functional Requirements and Testing)

### 13.1 📋 Non-Functional Requirements (NFRs)

| Metric | SLI / Target | Measurement |
|--------|--------------|-------------|
| Availabilty | 99% uptime during demo sessions | Basic uptime monitoring/logs |
| Stylization Latency (Fast NST) | ≤ 150 ms (512×512, GPU) per image | Time measurements |
| Stylization Latency (Opt. NST) | Configurable; reported, not bounded | Time measurements |
| Reliability | 0 critical crashes during standard usage | Test runs & logs |
| Storage Integrity | 100% metric/log persistence per benchmark run | Log completeness checks |

### 13.2 🧪 Test Plan

#### **🔬 Unit Tests**

- ➢ Loss functions, feature extraction layers, configuration parsers.

#### **🔗 Integration Tests**

- ➢ End-to-end stylization (input → output), verification that custom extractor is used.

#### **🖥️ System Tests**

- ➢ Full workflow: upload → recommendation → stylization → metrics display.

#### **⚡ Performance Tests**

- ➢ Stylization runtime on sample resolutions for both methods.

### 13.3 🌍 Environments

#### **💻 Development**

- ➢ Local machines with small datasets and debug configs.

#### **🚀 Staging**

- ➢ Shared lab GPU machine for integrated tests & benchmarks.

#### **🎬 Demo/Final**

- ➢ Stable environment (lab PC or cloud VM) for final presentation and evaluation.

---

## 14. 🔒 Security and Compliance

### 14.1 ⚠️ Threat Model (High Level)

| Asset | Threat Category | Mitigation |
|-------|-----------------|------------|
| Model & Codebase | Unauthorized modification E | Private repo, access control |
| Web Interface | Misuse or DoS during Demo D | Limited access, manual monitoring |
| Uploaded Images | Misuse of User Content I / T | Local-only demo, no public hosting |
| Logs & Metrics | Accidental Data Loss D / T | Regular backups, version control |

### 14.2 🔐 AuthN / AuthZ

- Only team members and mentor access code repository.
- Demo platform run in controlled environment (lab/localhost).

### 14.3 📝 Audit and Logging

- Log successful and failed stylization runs with timestamps.
- No personal identifying information stored.

### 14.4 ⚖️ Compliance

- Academic project; aligns with university IT policies.
- Uses open datasets and open-source tools with proper citation.

---

## 15. 🚀 Delivery and Operations

### 15.1 📦 Release Plan

#### **🔵 Alpha (Week 3–4)**

- ➢ Custom extractor trained; optimization-based NST prototype.

#### **🟢 Beta (Week 6)**

- ➢ Fast NST model added; basic UI + evaluation scripts.

#### **🔴 v1.0 (Week 8)**

- ➢ Full dual pipeline, benchmarks, recommendation engine, documentation, and demo-ready system.

### 15.2 🔄 CI / CD and Rollback

- Git-based version control.
- Basic CI (lint + unit tests) on push.
- Manual rollback via tags/branches.

### 15.3 📡 Monitoring and Alerting

- For demo, simple log checks and manual monitoring.
- If repeated errors or crashes are observed, fallback scripts and reduced functionality mode.

### 15.4 📢 Communication Plan

- Internal sync 2–3 times per week (short standups).
- Weekly progress summary to mentor.
- Milestone reviews at end of Week 4 and Week 8.

---

## 16. ⚠️ Risks and Mitigations

### **🔴 R1: Custom Extractor Underperforms → Poor NST Quality**

- ➢ **Mitigation**: Use adequate dataset and training epochs; if needed, simplify network or use smaller, well-regularized architecture.

### **🔴 R2: Fast NST Training Instability / Poor Style Fidelity**

- ➢ **Mitigation**: Start with a single style, tune loss weights, and refer to known successful architectures from literature.

### **🔴 R3: Time Overrun (Models + UI + Benchmarks Too Much)**

- ➢ **Mitigation**: Prioritize core features (custom extractor, both NST pipelines, minimal UI); add advanced UI/export features only if time allows.

### **🔴 R4: Hardware Limitations**

- ➢ **Mitigation**: Reduce resolution for training, use smaller datasets or subsampling; focus on clear relative comparison rather than huge scale.

### **🔴 R5: Integration Bugs Between Backend and UI**

- ➢ **Mitigation**: Early stub-based integration tests; incremental wiring; regular end-to-end tests.

---

## 17. 📊 Evaluation Strategy and Success Metrics

### **📈 Baseline Comparison**

- ➢ Compare optimization-based NST and fast NST using the same custom feature extractor on a curated test set.
- ➢ Metrics: runtime, memory footprint, style/content loss values, and simple perceptual metrics.

### **🎯 Scenario-Based Evaluation**

- ➢ Define scenarios (real-time mobile-like, offline quality-first, batch processing) and assess how each method performs relative to scenario demands.

### **💪 Stress Tests**

- ➢ Higher resolutions, multiple runs, varied styles to ensure stability.

### **✅ Success**

- ➢ Objectives in Section 8 met (dual pipelines, benchmarks, platform + recommendations, internal user satisfaction).
- ➢ Clear, empirically backed guidance on "when to use which NST method".

---

## 18. 📚 Appendices

### 18.1 📖 Glossary

- **NST (Neural Style Transfer)**: Technique that blends content of one image with style of another using deep neural features.
- **Optimization-Based NST**: Method where the output image is iteratively optimized by minimizing style and content loss.
- **Fast / Feed-Forward NST**: Method where a neural network is trained to directly output stylized images in one forward pass.
- **Perceptual Loss**: Loss computed in feature space of a CNN rather than pixel space.
- **VGG-like CNN**: Convolutional neural network architecture with stacked 3×3 conv layers and pooling layers, inspired by VGG16/19.
- **Style Loss**: Loss term measuring difference in style (often via Gram matrices).
- **Content Loss**: Loss term measuring difference in content features.

### 18.2 📄 References (for concept inspiration; not copied content)

- Gatys et al., "A Neural Algorithm of Artistic Style."
- Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution."
- Huang & Belongie, "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization."
- PyTorch and TensorFlow official documentation for style transfer tutorials.
- Relevant academic and tutorial material on feature extractors and perceptual losses.

---

## 🎉 End of Synopsis

**✨ StyleSense Team (T59)**  
**🏛️ GLA University**  
**📅 2025-2026**
