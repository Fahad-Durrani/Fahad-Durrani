## **Senior AI Engineer | Architecting & Building Production AI 🚀 | GenAI · Multimodal RAG · Multi-Agent · Recommenders · On-Device AI | MLOps · GCP · Azure · AWS | 5 yrs | MS Data Science (AI) – FAST NUCES**

📍 Islamabad, Pakistan  
🔗 [LinkedIn](https://www.linkedin.com/in/fahaddurrani/)

## 💡 About Me

Senior AI Engineer with nearly 5 years leading design and delivery of production AI systems across **GenAI, multi-agent frameworks, RAG, recommenders, and on-device AI**. Shipped products serving users across the **US, UAE, and Pakistan**, driving cost, latency, and quality trade-offs end-to-end from architecture through deployment on **Azure, AWS, and GCP**.

---

## 💼 Experience

### 🏢 Senior AI Engineer — 9D Technologies
📍 Hybrid (Islamabad, Pakistan) · 📅 February 2026 – Present

**Product:** [XeNo AI ↗](https://play.google.com/store/apps/details?id=filemanager.ai.personnel.assistant) · **AI File Manager for Android & iOS** · **Client:** Darwin Technology L.L.C (UAE)

- Cut multimodal **RAG cost by ~60%** by delivering the on-device inference stack (**ONNX / CoreML**) with quantized embedding models and on-device OCR (**Google ML Kit / Apple Intelligence**) across Android and iOS, powering search, chat, and Q&A over documents and images while keeping all data local.
- **Designed and led the Ask AI pipeline architecture** with context resolution, intent classification, and hybrid retrieval fusing semantic, keyword, metadata, and on-device structured-data queries (Room Database with text-to-SQL on Android, CoreData Predicates on iOS) via **Reciprocal Rank Fusion (RRF)** into a single cited response; enabled voice queries and optional Web Search via opt-in cloud APIs.
- Shipped **AI Collections and Proactive Insights** pipelines that auto-categorize files into semantic Collections, surface critical documents on Home, and generate Key Insights, FAQs, and follow-ups backed by a memory layer.
- Replaced random chunk selection with **content-aware retrieval** combining **KNN semantic search, BM25 keyword matching, and MMR for diversity-aware ranking** for summarization and cross-file RAG Q&A, cutting token cost and improving answer quality.
- Own the **GCP backend** on Google Cloud Run, mitigating **cold-start latency** via warm-instance strategies and auto-scaling, and managing logs, analytics, and routing optimization.
- **Lead the AI team** and cross-platform delivery across Android and iOS, driving intent routing, retrieval fusion, on-device inference strategy, and **project-wide cost optimization (LLM API + cloud spend)**; own sprint planning via Notion and Asana and contribute to AI UI/UX.

**🛠 Tech Stack:** Room Database, CoreData, SQLite, ONNX, CoreML, ML Kit, Apple Intelligence, Text & Image Embedding Models, Model Quantization (FP16, INT8), Hybrid Retrieval (RRF), Vertex AI, Google Cloud Run, Firebase, Notion, Asana

---

### 🏢 AI Engineer / Data Scientist — TechGenies
📍 Remote (Texas, USA) · 📅 Nov 2025 – Jan 2026

- **Delivered a natural-language analytics platform** replacing manual SQL requests for the internal analytics team, unblocking non-technical stakeholders to query NPS and customer feedback data directly via NL-to-SQL.
- **Reduced query latency by 30%** through Star Schema data modeling and schema-aware prompt design on **Azure OpenAI**; added sentiment and product classification over open-ended NPS responses to surface trends without manual tagging.
- Shipped as a **containerized REST service** on **Azure Container Instances** with **automated ETL** (Blob, Functions, SQL) and full observability (Monitor, App Insights), production-ready handoff to the client's internal team.

**🛠 Tech Stack:** Azure OpenAI, Azure Functions, Azure Container Instances, Azure SQL, Azure Monitor, App Insights

---

### 🏢 Machine Learning Engineer — ASLASE, Human Capital Management System
📍 Remote (Dubai, UAE) · 📅 Feb 2024 – Nov 2025

#### 📝 Product: [Takafo ↗](https://www.mubadala.com/) · AI-Powered Hiring Platform · **Client:** Mubadala (UAE)

*Enterprise platform combining AI-driven candidate-job matching, a role-based agentic layer for conversational access, and a document OCR pipeline spanning the full hiring and onboarding lifecycle.*

##### 🎯 Applicant Tracking System (ATS) — AI Matching & Scoring Engine

- Co-led development of an end-to-end ATS supporting **experienced, graduate, and intern hiring workflows**; authored the **Business Requirements Document (BRD)** defining scope, features, and stakeholder expectations, incorporating recruiter feedback throughout.
- Designed a full **CV/JD parsing and matching pipeline** covering parsing, skill extraction and standardization, functional experience mapping, career-level assignment, and **education verification** (degree/major relevance, CGPA, university ranking).
- Built a **multi-dimensional candidate–job scoring system** producing separate scores for overall match, skills match, career-level match, and functional-experience match with **AI-generated candidate summaries and skill-gap analysis** surfaced directly in the recruiter UI.
- Applied **generative AI (Azure OpenAI)** alongside classical NLP techniques — similarity measures, n-grams, pattern matching, Jaccard similarity, and embedding-based approaches — across every pipeline stage.
- Engineered **unbiased scoring formulas and ranking systems** grounded in domain-specific business logic for accurate extraction, verification, and field mapping.
- Deployed a **RAG pipeline on Milvus** for embedding storage and candidate matching; served embedding model via **PyTorch Serve**, optimizing generative-model costs.
- **Cut inference costs by ~40%** by designing first-level and second-level candidate-job matching tiers that reduced reliance on expensive generative models.
- Managed data with **PostgreSQL** and shipped the full ATS as **containerized microservices on Azure** with scalability, reliability, and secure remote access.

<details>
  <summary><b><u>📸 Click to view ATS Screenshots</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1ufscR7UVbz3blAkkRwlF8qvT6je-o5HE" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1hJrqEuhPYbyFUVFo9DbS_gqVE9_WmiwR" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1trElOIgg4UJ43SYfFD7QiHXMC_QU7OE0" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1yLg57K29MLAO1k4ncamNmHJwXtxibD__" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1bYS-7KYDoFpB_3qUUPf7uGz8e5EeYQEn" width="500" />
  </p>
</details>

##### 🤖 Multi-Agent Layer (LangGraph + MCP)

- Built a **role-based agentic layer over the entire platform** — a **LangGraph MCP framework with 8 role-based agents** (recruiter, hiring manager, candidate & more), each scoped to that user role's UI and data permissions.
- Designed a **hybrid tool-routing architecture** where each agent chose intelligently between two access modes — **frontend APIs** (for operational actions like candidate screening, approvals, status changes, and role definition, reusing its permission and audit layer) and **direct database access via a LangChain NL-to-SQL agent** (for complex analytics queries not covered by pre-built dashboards).
- Handled both **operational actions and analytical queries** through the same conversational UX, from *"approve this candidate's status change"* to *"how many requisitions are pending and for how long?"*, mirroring dashboards and workflows via chat instead of requiring users to navigate the product manually.
- Enforced **human-in-the-loop approvals** for state-changing actions and applied role-based database access controls so each agent only queried and acted on data its user role was authorized to see.
- Integrated **action execution, process analytics, and conversational memory** with full observability, logging, and traceability; deployed on Azure as a microservice.

<details>
  <summary><b><u>📸 Click to view Multi-Agent Screenshot</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1-LHZ82axYW38mpG8Z45hU2cJP2CGbc7q" width="500" />
  </p>
</details>

##### 📄 Document OCR Pipeline (Application Intake + Onboarding)

- **Owned the document ingestion layer** spanning application intake (CVs, experience letters, educational and equivalency certificates) and onboarding (identity, compliance, and life-event documents) — **15+ distinct document types including UAE-specific formats like Emirates IDs and Thiqa cards**, tailored to UAE regulatory requirements.
- **Fine-tuned custom Azure models** for structured extraction and object detection — Azure Document Intelligence Custom Models for key–value fields (issue/expiry dates, IDs, attestation status, salary, nationality) and Azure Custom Vision Models for stamps, logos, signatures, and letterhead detection on experience letters and legal documents.
- Combined **Azure Document Intelligence Prebuilt Models** (Read, Layout, Identity Document) for common document types with fine-tuned custom models for domain-specific ones; curated and labeled diverse datasets spanning multiple regions and languages to achieve high precision and recall on real-world documents.
- Shipped the **end-to-end OCR pipeline** as a containerized microservice on Azure — integrating prebuilt and custom models via the Azure SDK with image validation, format standardization, and text preprocessing that **cut GenAI cost by ~20%** in production.

<details>
  <summary><b><u>📸 Click to view OCR Screenshots</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=18MfxTNKJZwY2_doGdhWo83BZgcRlYKeD" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1FcntuHASPcEV7TclqGavzTXe0QB4ZEkT" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1c4ISwCFwdGg2op_1YqBoWkgkVOAw6EF-" width="600" />
  </p>
</details>

**🛠 Tech Stack (Takafo):** Azure OpenAI, Azure Document Intelligence, Azure Custom Vision, Azure SDK, Azure ML (model monitoring), Azure Monitor, Azure Blob Storage, Azure Key Vault, Milvus, PostgreSQL, Hugging Face Transformers, PyTorch Serve, spaCy, NLTK, LangGraph, MCP, LangChain (NL-to-SQL), FastAPI, WebSockets, Docker, pdf2image, PyMuPDF, Pillow, OpenCV, Pydantic, Python

---

#### 🍽️ Product: [RestHero ↗](https://www.resthero.io/) · Multi-brand food recommendation SaaS platform · **Client:** RestHero (UAE)

- Architected a **multi-brand content-based recommender** serving restaurant catalogues across the platform, with an **ETL pipeline** ingesting item metadata (title, description, ingredients) from **MongoDB** and **cron-driven refreshes** for menu updates and new item onboarding.
- Solved the **item cold-start problem** using multilingual sentence embeddings, making new dishes and restaurants recommendable from metadata alone.
- Instrumented the recommender with **CTR, Coverage, Diversity, Novelty, Churn, and Responsiveness** as first-class metrics, moving evaluation beyond accuracy to catch filter-bubble and staleness regressions before release.
- Built a **Power BI analytics pipeline** monitoring restaurant churn, user behavior, and system health, giving business stakeholders self-serve visibility into recommender performance.
- Shipped a **WhatsApp AI catalogue assistant** that captured cravings in natural language via **NER (Hugging Face transformer)**, pulled candidates from the recommender, and ranked nearby restaurants using **Haversine distance, Google Distance Matrix, and Geocoding APIs**.
- Deployed end-to-end on **AWS (EC2, S3)** with automated model refresh workflows and served embedding models via **PyTorch Serve**.

**🛠 Tech Stack (RestHero):** Python, MongoDB, AWS S3, AWS EC2, Hugging Face Transformers, PyTorch Serve, Pandas, Power BI, Google Maps APIs

<details>
  <summary><b><u>📸 Click to view RestHero Screenshots</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=12pHrsdx9YhJk9XKXfgpDsMB5SzS6qPl5" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1BHOAWJJ9DcuoAzkPz-5BiYuPt4jz3ThH" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=10T7pVQ8Dbgs9SE7Q_7OfdzVXETZvVoNl" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1jcYDXVpGnuKOP1Q8VosgQjcKs6ZTqP8U" width="400" />
  </p>

</details>

---

### 🏢 Machine Learning Engineer — Artificial Intelligence Diagnostic Lab
📍 On-Site (Islamabad, Pakistan) · 📅 Feb 2022 – Jan 2024

**Client:** [AKUH ↗](https://hospitals.aku.edu/Pages/default.aspx) · Medical Imaging & Clinical NLP

- Designed a **compact 2D architecture achieving accuracy comparable to 3D models** on the **BRATS 2021** brain tumor segmentation benchmark, fused multiple attention mechanisms (**CBAM with kernel factorization**, coordinate attention aligned with Connected Component Analysis) into a **3.5M-parameter gradient-flow network**, validated via cross-validation across diverse medical datasets.
- Automated tumor annotation on **hospital-provided imaging data** via Connected Component Analysis and a pre-trained **YOLO** for tumor-slice classification, significantly reducing manual labeling effort.
- Prototyped an **early RAG-based clinical NLP system** using OCR,
