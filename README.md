## **Senior AI Engineer | Architecting & Building Production AI 🚀 | GenAI · Multimodal RAG · Multi-Agent · Recommendation Engines · On-Device AI | MLOps · GCP · Azure · AWS | 5 yrs | MS Data Science (AI) – FAST NUCES**

📍 Islamabad, Pakistan  
🔗 [LinkedIn](https://www.linkedin.com/in/fahaddurrani/)

## 💡 About Me

Senior AI Engineer with nearly 5 years leading design and delivery of production AI systems across **GenAI, multi-agent frameworks, RAG, recommenders, and on-device AI**. Shipped products serving users across the **US, UAE, and Pakistan**, driving cost, latency, and quality trade-offs end-to-end from architecture through deployment on **Azure, AWS, and GCP**.

## 🏆 Successfully Delivered Projects

### 📱 XeNo AI — Privacy-First On-Device AI File Manager (Android & iOS)
[Live Website Link](https://play.google.com/store/apps/details?id=filemanager.ai.personnel.assistant)
- Cut multimodal **RAG cost by ~60%** by delivering the on-device inference stack (**ONNX / CoreML**) with quantized embedding models and on-device OCR (**Google ML Kit / Apple Intelligence**) across Android and iOS, powering search, chat, and Q&A over documents and images while keeping all data local.
- **Designed and led the Ask AI pipeline architecture** with context resolution, intent classification, and hybrid retrieval fusing semantic, keyword, metadata, and on-device structured-data queries (Room Database with text-to-SQL on Android, CoreData Predicates on iOS) via **Reciprocal Rank Fusion (RRF)** into a single cited response.
- Shipped **AI Collections and Proactive Insights** pipelines that auto-categorize files into semantic Collections, surface critical documents on Home, and generate Key Insights, FAQs, and follow-ups backed by a memory layer.
- Replaced random chunk selection with **content-aware retrieval** combining **KNN semantic search, BM25 keyword matching, and MMR for diversity-aware ranking** for summarization and cross-file RAG Q&A.
- **Lead the AI team** and cross-platform delivery across Android and iOS, driving intent routing, retrieval fusion, on-device inference strategy, and **project-wide cost optimization (LLM API + cloud spend)**.

**🛠 Tech Stack:** Room Database, CoreData, SQLite, ONNX, CoreML, ML Kit, Apple Intelligence, Text & Image Embedding Models, Model Quantization (FP16, INT8), Hybrid Retrieval (RRF), Vertex AI, Google Cloud Run, Firebase, Notion, Asana

---

### 🤖 Multi-Agent (LangGraph) Hiring & Onboarding System
[Live Website Link](https://www.mubadala.com/)
- Built a **LangGraph MCP framework** with **8 role-based agents** (recruiter, hiring manager, candidate & more), each scoped to that user role's UI and data permissions.
- Designed a **hybrid tool-routing architecture** where each agent chose intelligently between **frontend APIs** (for operational actions) and **direct database access via a LangChain NL-to-SQL agent** (for complex analytics queries).
- Enforced **human-in-the-loop approvals** for state-changing actions and applied role-based database access controls so each agent only queried and acted on data its user role was authorized to see.
- Integrated **action execution, process analytics, and conversational memory** with full observability, logging, and traceability; deployed on Azure as a microservice.

**🛠 Tech Stack:** LangGraph, MCP, LangChain (NL-to-SQL), Backend APIs, Azure Blob Storage, WebSockets, Docker

<details>
  <summary><b><u>📸 Click to view Screenshot</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1-LHZ82axYW38mpG8Z45hU2cJP2CGbc7q" width="500" />
  </p>
</details>

---

### 📝 Applicant Tracking System (ATS) — AI Matching & Scoring Engine
[Live Website Link](https://www.mubadala.com/)
- Co-led development of an end-to-end ATS supporting **Experienced, Graduate, and Intern hiring workflows**; authored the **Business Requirements Document (BRD)** defining scope, features, and stakeholder expectations, incorporating recruiter feedback throughout.
- Designed a full **CV/JD parsing and matching pipeline** covering parsing, skill extraction and standardization, functional experience mapping, career-level assignment, and **education verification** (degree/major relevance, CGPA, university ranking).
- Built a **multi-dimensional candidate–job scoring system** producing separate scores for overall match, skills match, career-level match, and functional-experience match with **AI-generated candidate summaries and skill-gap analysis** (job-required vs. additional skills) surfaced directly in the recruiter UI.
- Applied **generative AI (Azure OpenAI)** alongside classical NLP techniques — similarity measures, n-grams, pattern matching, Jaccard similarity, and embedding-based approaches — across every pipeline stage.
- Engineered **unbiased scoring formulas and ranking systems** grounded in domain-specific business logic for accurate extraction, verification, and field mapping.
- Deployed a **RAG pipeline on Milvus** for embedding storage and candidate matching; served embedding model via **PyTorch Serve**, optimizing generative-model costs.
- **Cut inference costs by ~40%** by designing first-level and second-level candidate-job matching tiers that reduced reliance on expensive generative models.
- Shipped the full ATS as **containerized microservices on Azure** with scalability, reliability, and secure remote access.

**🛠 Tech Stack:** Azure OpenAI, Azure Document Intelligence, Hugging Face Transformers, Milvus, Azure Blob Storage, Azure Key Vault, Azure ML (model monitoring), Azure Monitor, Python, PyTorch Serve, PostgreSQL, spaCy, NLTK, Docker, FastAPI, WebSockets, Pydantic

<details>
  <summary><b><u>📸 Click to view Screenshots</u></b></summary>

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

---

### 📄 Document OCR Pipeline (Application Intake + Onboarding)
[Live Website Link](https://www.mubadala.com/)
- **Owned the document ingestion layer** spanning application intake (CVs, experience letters, educational and equivalency certificates) and onboarding (identity, compliance, and life-event documents) — **15+ distinct document types including UAE-specific formats like Emirates IDs and Thiqa cards**, tailored to UAE regulatory requirements.
- **Fine-tuned custom Azure models** for structured extraction and object detection — **Azure Document Intelligence Custom Models** for key–value fields (issue/expiry dates, IDs, attestation status, salary, nationality) and **Azure Custom Vision Models** for stamps, logos, signatures, and letterhead detection on experience letters and legal documents.
- Combined **Azure Document Intelligence Prebuilt Models** (Read, Layout, Identity Document) for common document types with fine-tuned custom models for domain-specific ones; curated and labeled diverse datasets spanning multiple regions and languages to achieve high precision and recall.
- Shipped the **end-to-end OCR pipeline** as a containerized microservice on Azure — integrating prebuilt and custom models via the Azure SDK with image validation, format standardization, and text preprocessing that **cut GenAI cost by ~20%** in production.

**🛠 Tech Stack:** Azure Document Intelligence, Azure Custom Vision, Azure SDK, Microsoft Azure, Python, Azure Blob Storage, PostgreSQL, pdf2image, PyMuPDF, Pillow, OpenCV, FastAPI, Pydantic

<details>
  <summary><b><u>📸 Click to view Screenshots</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=18MfxTNKJZwY2_doGdhWo83BZgcRlYKeD" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1FcntuHASPcEV7TclqGavzTXe0QB4ZEkT" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1c4ISwCFwdGg2op_1YqBoWkgkVOAw6EF-" width="600" />
  </p>
</details>

---

## 🍽️ Food Recommendation System (RestHero)
[Live Website Link](https://www.resthero.io/)
- Architected a **multi-brand content-based recommender** serving restaurant catalogues across the platform, with an **ETL pipeline** ingesting item metadata (title, description, ingredients) from MongoDB and **cron-driven refreshes** for menu updates and new item onboarding.
- Solved the **item cold-start problem** using multilingual sentence embeddings — new dishes and newly-onboarded restaurants became recommendable immediately from metadata alone, with no dependency on historical interaction data.
- Instrumented the recommender with **CTR, Coverage, Diversity, Novelty, Churn, and Responsiveness** as first-class metrics, moving evaluation beyond accuracy to catch filter-bubble and staleness regressions before release.
- Built a **Power BI analytics pipeline** monitoring restaurant churn, user behavior, and system health, giving business stakeholders self-serve visibility into recommender performance.
- Shipped a **WhatsApp AI catalogue assistant** that captured cravings in natural language via **NER (Hugging Face transformer)**, pulled candidates from the recommender, and ranked nearby restaurants using **Haversine distance, Google Distance Matrix, and Geocoding APIs**.
- Deployed end-to-end on **AWS (EC2, S3)** with automated model refresh workflows and served embedding models via **PyTorch Serve**.

**🛠 Tech Stack:** Python, MongoDB, AWS S3, AWS EC2, Hugging Face Transformers, PyTorch Serve, Pandas, Power BI, Google Maps APIs

<details>
  <summary><b><u>📸 Click to view Screenshots</u></b></summary>

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

### 🤖 Multi-Agent Chatbot Framework
[Live Website Link](https://d2a5948llkb7uv.cloudfront.net/auth/login)
- Designed, built, and deployed a complete **Multi-agent workflow** POC using **CrewAI with WebSocket** support for real-time interactions, leveraging OpenAI's GPT models for NLP tasks, and hosted the solution on AWS.
- Developed a **LangChain-based SQL Agent** to translate natural language queries into SQL statements and retrieve structured skill demand and supply data from relational databases.
- Implemented a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, fetching relevant documents from a vector store and generating context-aware responses via OpenAI's LLMs.
- Integrated custom **CrewAI tools** to generate real-time visualizations — such as bar charts and pie charts — embedded directly within the chatbot interface.

**🛠 Tech Stack:** AWS EC2, CrewAI, LangChain SQL Agent, LangChain RAG, OpenAI, Plotly, WebSockets, FastAPI, Milvus, S3

<details>
  <summary><b>📸 Click to view Screenshots</b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1ykAZ9svPSZ9OmoFz4yNP3nWuaV_Hb4TO" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1hOjh_oJDAqQCvVAW6w9RY5G0wMgQqtmr" width="400" />
  </p>

</details>

---

### 🧠 Healthcare Diagnostics & Medical Imaging (AKUH)
[Live Website Link](https://hospitals.aku.edu/Pages/default.aspx)
- Designed a **compact 2D architecture achieving accuracy comparable to 3D models** on the **BRATS 2021** brain tumor segmentation benchmark — fused multiple attention mechanisms (**CBAM with kernel factorization**, coordinate attention aligned with Connected Component Analysis) into a **3.5M-parameter gradient-flow network**, validated via cross-validation across diverse medical datasets.
- Automated tumor annotation on **hospital-provided imaging data** via Connected Component Analysis and a pre-trained **YOLO** for tumor-slice classification, significantly reducing manual labeling effort.
- Prototyped an **early RAG-based clinical NLP system** using OCR, **ChromaDB, LlamaIndex, and OpenAI LLMs over MySQL** for natural-language querying of prescription data.
- Predicted **patient survival** by combining radiomic features with deep learning and ensemble models (**XGBoost, Random Forest, Decision Trees**); visualized results in Power BI, Pandas, and Plotly.

**🛠 Tech Stack:** Python, PyTorch, TensorFlow, YOLO, ChromaDB, LlamaIndex, OpenAI, XGBoost, Random Forest, MySQL, Weights & Biases, Pandas, NumPy, Plotly, Power BI

<details>
  <summary><b><u>📸 Click to view Screenshots</u></b></summary>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1yqkYULvCnDcxCV0NZQrd9wuKJjTD16NF" width="400" />
    <img src="https://drive.google.com/uc?export=view&id=1v3DA-qwi9c_X__raOsscKzSzLPC8jE2R" width="400" />
  </p>

  <p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1xLlRnm5LUKTal2WpGbcS9sVG67_tkF30" width="400" />
  </p>

</details>

---

## 🛠️ Skills

| Category | Skills |
|----------|--------|
| **GenAI & Agents** | LLMs (OpenAI, Google Gemini), RAG, LangChain, LlamaIndex, LangGraph, CrewAI, MCP, text-to-SQL, PEFT (LoRA, Quantization), Fine-tuning |
| **ML, DL & On-Device AI** | PyTorch, TensorFlow, Hugging Face Transformers, spaCy, NLTK, XGBoost, YOLO, Recommender Systems, NER, OCR — *On-device:* ONNX (Android), CoreML (iOS), ML Kit, Apple Intelligence, Model Quantization (FP16, INT8), Hybrid Retrieval (RRF) |
| **Cloud & MLOps** | Azure (OpenAI, Doc Intelligence, Custom Vision, ML, Functions, ACI, Blob, Key Vault, Monitor), AWS (SageMaker, Bedrock, Lambda, EC2, S3), GCP (Cloud Run, Vertex AI, Firebase), Docker, GitHub Actions, MLflow, PyTorch Serve, CI/CD |
| **Data & Backend** | Python, SQL, PostgreSQL, MongoDB, MySQL, Milvus, ChromaDB, FastAPI, WebSockets, Pydantic, Power BI |
| **Tools** | Claude Code, Cursor, Git, Jira, Notion, Asana |

---

## 🎓 Education

| Degree | Institution | CGPA | Duration |
|--------|-------------|------|----------|
| **MS Data Science** | National University of Computer and Emerging Sciences (FAST), Islamabad | 3.96 | 2021–2024 |
| **BS Electronics Engineering** | COMSATS University, Abbottabad Campus | 3.09 | 2013–2017 |

---

## 📑 Research Papers

- **Paper 1:** [SCADA & PLC based fully automated pneumatic cutting machine: A test bench for industry and laboratory](https://ieeexplore.ieee.org/document/8338634) — IEEE
- **Paper 2:** [Design and Implementation of Smart Fault Detection System for Industrial Power House using PLC and SCADA](https://www.academia.edu/33228236/Design_and_Implementation_of_Smart_Fault_Detection_System_for_Industrial_Power_House_using_PLC_and_SCADA) — Academia.edu

---

## 📜 Certificates

| Certificate | Issued By | Platform |
|-------------|-----------|----------|
| **Claude Code in Action** | Anthropic | Coursera |
| **LLMOps** | Duke University | Coursera |
| **MLOps** | DeepLearning.AI | Coursera |
| **NLP Specialization** | DeepLearning.AI | Coursera |
| **Generative AI with LLMs** | DeepLearning.AI | Coursera |
| **Deep Learning Specialization** | DeepLearning.AI | Coursera |
| **Machine Learning Specialization** | Stanford & DeepLearning.AI | Coursera |
| **Recommender Systems** | — | Udemy |
| **Data Analysis with R** | Duke University | Coursera |
| **DeepLearning.AI TensorFlow Developer** | DeepLearning.AI | Coursera |
| **Introduction to LangGraph** | LangChain | LangChain Academy |

---

✨ Always open to collaborations on **AI, GenAI, RAG, Multi-Agent Systems, On-Device AI, and MLOps projects**.
