---
title: "No-Code ML Platform"
description: "Enterprise-grade no-code ML platform with time series forecasting module, reducing baseline model development time by 80% for forecasting workflows."
tech: [Python, PySpark, TensorFlow, Scikit-learn, XGBoost, Docker, Kubernetes, AWS EKS, ArgoCD, Crossplane, Kafka, Grafana, Prometheus, PostgreSQL, Qdrant]
featured: true
---

## Overview

Co-developed an enterprise no-code ML platform with a cross-functional team of 10–15 engineers. The platform enables non-technical users to ingest data from 500+ enterprise sources, build analytics models through a visual workflow interface, and deploy to production — all without writing code. I owned the **time series forecasting module** end-to-end, which automated 7+ manual preprocessing steps and reduced baseline model development time by 80%.

The platform is deployed across Fortune 500 organizations in healthcare, retail, banking, and insurance verticals, achieving 97.52% cumulative accuracy on retail demand forecasting benchmarks with segmented forecasting by product, region, and store. Customer onboarding is fully automated through an Infrastructure-as-Code pipeline using Argo Workflows, Crossplane, and ArgoCD, provisioning isolated tenant environments in under one hour.

---

## The Problem

Traditional enterprise analytics projects spend **90% of their effort on infrastructure** — building pipelines, ETL, data warehouses — before a single model gets trained. Data scientists burn weeks on data wrangling. Business users wait months for insights. And once a model is built, getting it to production is another battle entirely.

We wanted to flip this: a platform where **100% of the focus is on business outcomes**. Connect your data, build your model, deploy it — no code required, no infrastructure to manage. The key personas we designed for were **Business Analysts** who understand the domain but can't write Python, **Data Scientists** who want to iterate faster without infrastructure overhead, and **IT Administrators** who need governance, security, and compliance at scale.

---

## Platform Architecture

The platform is built as a **modular, cloud-native SaaS application** composed of several integrated components. Each module handles a distinct concern in the ML lifecycle, and they communicate through well-defined APIs and event-driven messaging.

| Module | Responsibility |
| ------ | ------------- |
| **DataSpace** | Data integration and consolidation across enterprise sources |
| **ModelSpace** | No-code model building, training, and experimentation |
| **ModelSpace Library** | Reusable components — actions, widgets, functions |
| **SolutionSpace** | Interactive dashboards and production applications |
| **ProductionSpace** | Model deployment, monitoring, and pipeline management |
| **Conversational AI** | Natural language interface for querying data and triggering workflows |
| **CloudSpace** | Cloud resource observability, cost tracking, and autoscaling |
| **Insights API** | REST API for integrating platform outputs into external systems |

The entire platform runs on **Amazon Web Services (AWS)**, with multi-cloud support (GCP, Azure) on the roadmap.

---

## DataSpace: The Data Integration Layer

DataSpace is the **foundational data integration platform** that consolidates enterprise data across mainframes, on-premises systems, and cloud platforms. It ships with over **500 pre-built connectors** to enterprise systems including ERP systems like SAP and Oracle, CRM platforms like Salesforce and HubSpot, warehouse management and inventory systems, marketing platforms like Google Ads and Facebook Ads, digital ad spend and payment processors, and cloud storage services like S3, Azure Blob, and GCS.

### Ingestion Pipeline

The connector layer is powered by **Airbyte** under the hood. When a user configures a data source through the UI, the DataSpace UI sends a trigger ingestion request to the API Gateway, which publishes the ingestion request to **Kafka** as an event. Kafka forwards the event to the Trigger Ingestion module, which communicates with the **Manager API** to instruct Airbyte to extract data from the designated external connectors via an Ingestion Worker. The extracted data flows through Airbyte into the **Data Lake** backed by S3.

Upon successful ingestion, Kafka notifies the Interface API, which generates metadata for datasets and data profiles. If the ingested data includes nested or array structures, the system triggers a normalization process that flattens complex fields into separate tables for better accessibility. Finally, the DataSpace backend saves the final dataset while the DataService backend logs ingestion metadata including the database name, batch ID, and schedule/frequency details.

DataSpace creates a **digital data twin** of source systems and performs automated profiling to assess **data quality** and detect **data events**. These insights feed directly into ModelSpace workflows.

### Data Quality & Analysis

Every ingested dataset gets **automatically profiled** with column-level statistics covering nulls, distributions, and cardinality. The system runs anomaly detection on incoming batches, monitors for schema drift across ingestion runs, and produces an overall data quality score. This was critical for our enterprise clients — bad data in, bad models out. Automated profiling catches issues before they propagate downstream.

---

## ModelSpace: No-Code Model Building

ModelSpace is the **no-code modeling environment** where users build end-to-end analytic workflows. It supports the full spectrum of data preparation and modeling tasks: data shaping (filtering, pivoting, reshaping), data cleaning (handling missing values, outliers, duplicates), automated feature engineering (extraction, encoding, scaling), joins and appends for combining multiple datasets, model training across classification, regression, clustering, time series forecasting, and anomaly detection, model ensembling for combining multiple models, and automated hyperparameter tuning.

### The No-Code Workflow Engine

Users build models by dragging and dropping **Actions** — reusable logic components — onto a visual canvas. Each Action represents an operation: load data, clean columns, engineer features, train a model, evaluate results. Under the hood, each Action maps to a Python execution unit. The platform supports **140+ open-source ML/LLM libraries** including Scikit-learn for classical ML (regression, classification, clustering), TensorFlow and Keras for deep learning, XGBoost for gradient boosting, PySpark for large-scale data processing, and NLTK/SpaCy for NLP tasks.

The **Hyperscaling Execution Service** dynamically allocates compute resources for large workloads. When a user kicks off a training job, the platform analyzes the dataset size and model complexity, provisions appropriate compute by scaling EKS pods horizontally, executes the workflow, streams real-time execution logs back to the UI, and stores model artifacts in the centralized **Artifact Store**.

### Collaborative Workspaces

ModelSpace uses **Workspaces** — isolated, collaborative environments where teams can experiment without risking disruption to source systems or production models. Each workspace has its own data access permissions, model version history, execution environment, and user role assignments. This was designed so that a junior analyst can freely experiment in a sandbox without accidentally overwriting a production forecasting model.

---

## My Module: Time Series Forecasting

I owned the **time series forecasting module** end-to-end. This was the most technically challenging module because time series data has unique requirements that don't map neatly to standard tabular ML workflows.

### The Problem with Manual Time Series Workflows

Before our module, building a time series forecast required a data scientist to manually identify and parse datetime columns, handle irregular time intervals and missing timestamps, set the correct frequency (hourly, daily, weekly, monthly), create lag features (t-1, t-2, ..., t-n), engineer rolling statistics like moving averages and exponential smoothing, handle seasonality decomposition, select and tune an appropriate model, validate using proper time-series cross-validation without data leakage, and generate forecasts at the right horizon. Each of these steps is error-prone, and getting any one wrong invalidates the entire forecast.

### What We Automated

Our module automated **7+ manual preprocessing steps** into a single, configurable workflow.

**Automated DateTime Detection & Parsing** — The system analyzes all columns and identifies datetime fields automatically. It infers the frequency (daily, weekly, monthly) from the data distribution and fills gaps with appropriate imputation strategies.

**Intelligent Feature Engineering** — Based on the detected frequency, the module automatically generates lag features with configurable lag windows (e.g., t-1 through t-7 for daily data), rolling statistics including moving averages, rolling standard deviations, and exponential weighted means, calendar features like day of week, month, quarter, and holiday flags, trend components through linear and polynomial trend decomposition, and seasonal decomposition using STL (Seasonal and Trend decomposition using Loess).

**Automated Model Selection** — The module runs a suite of models in parallel and selects the best performer based on validation metrics. It evaluates statistical models like ARIMA, SARIMA, and Exponential Smoothing (Holt-Winters), ML models like XGBoost, LightGBM, and Random Forest with lagged features, and deep learning approaches using LSTM networks for complex sequential patterns. We used **walk-forward validation** to prevent data leakage — the model only ever trains on past data and validates on future windows.

**Explainability** — Every forecast comes with feature importance rankings showing which lag and calendar features matter most, confidence intervals at configurable levels (80%, 95%), actual vs. predicted overlays for visual validation, and residual analysis plots for diagnosing model behavior.

### Results

Deployed across Fortune 500 organizations, the module reduced baseline development time from weeks to **hours** (an 80% reduction), enabled non-technical users to build production-grade forecasts without writing a single line of code, achieved **97.52% cumulative accuracy** on retail demand forecasting benchmarks, and supported segmented forecasting across product, region, and store dimensions.

---

## ProductionSpace: From Experiment to Production

ProductionSpace manages the **deployment, scaling, and monitoring** of models and applications in production.

### One-Click Productionization

The key design principle was to **promote workspaces from experimentation to production with one click**. When a user productionizes a ModelSpace workspace, the system snapshots the current workflow state, packages model artifacts including serialized models, feature pipelines, and configs, deploys them as **REST endpoints** on the production EKS cluster, sets up automated production data pipelines that feed fresh data, and enables scheduled re-training at configurable intervals.

### Model Governance & Lineage

Every production workspace includes **full data lineage** — tracking source data, derived features, model inputs, and outputs across the entire pipeline. The system versions every model artifact, compares expected vs. actual results over time, and enables model retraining and updates without disrupting production applications. This was non-negotiable for our enterprise clients in regulated industries like healthcare, banking, and insurance. They needed to demonstrate exactly how a prediction was generated, which data it used, and when the model was last trained.

### Model Auditability & Explainability

The platform follows **CRISP-DM** methodology and integrates MLOps pipelines via **ArgoCD** for continuous model delivery. Evaluation metrics including accuracy, precision, recall, and F1-score are tracked per model version. For bias and explainability, the platform uses SHAP for feature attribution and RAG scoping to limit model behavior to enterprise-only content. Models are retrained on a quarterly cadence and monitored continuously via Grafana OnCall for performance degradation.

---

## Conversational AI: Natural Language Interface

The platform includes a **conversational AI interface** that lets users interact with the entire platform using natural language. Business users can ask questions about their data like "What were last quarter's sales by region?", generate insights without writing SQL, and trigger analytics workflows through chat — bridging the gap between raw data and business decision-making.

### Gen AI Architecture

The conversational AI layer uses a sophisticated multi-stage pipeline. User inputs are first routed through a custom client layer for prompt engineering and contextual metadata enrichment. **Titan Embedding** then transforms customer-specific data into vector embeddings, which are stored in **Qdrant Vector DB** to enable semantic search for retrieval-augmented generation (RAG). **Claude 3.5** generates SQL or code using contextual prompts, while **AWS Bedrock** orchestrates LLM inference with LLAMA 70B as a fallback model. Results are returned to the user with interactive visualizations.

The RAG strategy is carefully designed: **no PII or contract data is vectorized**. Only metadata about dataset tables, relations, and methods is stored in Qdrant. Documents are uploaded by clients only if they explicitly opt in. Hallucination management uses **Bedrock Guardrails**, blocking 85% of harmful content and filtering 75% of hallucinated responses for RAG queries.

---

## Cloud Infrastructure & Deployment

### Master-Child Account Architecture

The platform uses a **Master-Child AWS account structure**. The Master (Backoffice) Account hosts centralized management services including Account Management, CloudSpace, Crossplane, ModelSpace Library, TrainingSpace, and the full observability stack with Grafana, Prometheus, Mimir, Loki, and Tempo. Each customer gets a **Child Account** — an isolated AWS account with a dedicated EKS cluster, its own VPC, IAM roles, KMS encryption, and all core platform applications. This provides **strong tenant isolation** where each customer's data stays in their own AWS account, with their own encryption keys and network boundaries.

### Automated Customer Onboarding Pipeline

Provisioning a new customer environment is **fully automated** and completes within one hour. The pipeline uses Infrastructure-as-Code (IaC) with **Argo Workflows** for orchestrating the multi-step provisioning pipeline, **Crossplane** for declaratively provisioning cloud infrastructure including VPCs, EKS clusters, and IAM roles, **ArgoCD** for GitOps-based application deployment, and **RabbitMQ** for event-driven communication between pipeline stages.

The process begins when Account Management triggers the customer creation flow and publishes a message to a RabbitMQ topic. Argo Events detects this message and triggers an Argo Workflow that provisions cloud resources including VPCs, IAM roles, and EKS clusters. Crossplane generates a ChildAccount manifest and streams status updates back to RabbitMQ. A watcher process monitors resource readiness, and once resources are confirmed ready, the workflow templates deployment manifests with the customer's account ID and KMS key, pushes them to the Deployments Repository, and ArgoCD syncs and deploys all applications into the new customer cluster. A final service readiness check verifies all health endpoints before publishing the deployment-ready status. This ensures **consistent, repeatable, secure** onboarding at scale.

### EKS Cluster Architecture

Each customer VPC runs a dedicated **Amazon EKS cluster** with purpose-built node groups: Airbyte nodes (t4g.2xlarge) handle data ingestion workloads, data-tooling nodes (m6g.medium to m6a.2xlarge) run processing pipelines, database nodes (c6g.large, r6g.xlarge) host PostgreSQL, services nodes (r6a.4xlarge, c6a.large) run platform applications, dripper nodes (t4g.2xlarge) manage data streaming, and operations nodes (c6g.medium) handle cluster management. Node provisioning uses **Karpenter** for intelligent autoscaling — right-sizing instances based on actual workload demands and leveraging **Spot instances** (up to 62% discount) for non-critical workloads.

### Observability Stack

The platform provides **comprehensive observability** across all tenants using a full Grafana stack. **Mimir** stores and queries metrics from all services, monitoring CPU, memory, and resource utilization. **Loki** provides centralized log aggregation from all containers, services, and applications. **Tempo** delivers distributed tracing for end-to-end visibility of application requests. **Grafana** ties it all together with unified dashboards for metrics, logs, and traces, while **Grafana OnCall** handles alert routing and incident management with configurable escalation policies. **Prometheus** collects metrics from EKS nodes and pods, forwarding them to Mimir for long-term storage. Each child account has its own Mimir, Loki, and Tempo data sources, ensuring tenant data separation while enabling centralized monitoring from the Backoffice Grafana.

---

## Security & Compliance

Security was architected as a **first-class concern**, not an afterthought.

Every customer runs in an **isolated AWS account** with a dedicated VPC, ensuring complete infrastructure isolation. Network boundaries are enforced through Security Groups and Network Firewalls, with a **Web Application Firewall (WAF)** protecting all public-facing endpoints. All data in transit is encrypted with **TLS**, data at rest uses **S3 server-side encryption**, and each tenant gets their own **KMS** encryption keys.

For identity and access management, the platform integrates with customer identity providers through **OIDC-based Single Sign-On (SSO)** and supports **Multi-Factor Authentication (MFA)**. Role-based access control is enforced at the workspace, dataset, and application levels, giving administrators fine-grained control over who can access what.

The platform is designed for **SOC 2, HIPAA, and FedRAMP** alignment, with full audit logging via **CloudTrail**, vulnerability management using CVSS-based severity assessment, and **GuardDuty** for continuous threat detection across all accounts.

---

## Industry Use Cases

The platform is deployed across multiple verticals.

### Healthcare

In healthcare, the platform enables real-time monitoring of Central Line-Associated Bloodstream Infections (**CLABSI**) using predictive models on EHR data, delivering dashboards and alerts to infection prevention teams. It powers AI-driven **demand planning** for staffing, appointment volumes, and resource allocation across facilities. For **clinical research**, it accelerates cohort discovery through AI-assisted mining of clinical and claims data, leveraging LLMs for literature synthesis and eligibility criteria matching. The platform also automates **HIPAA and CMS compliance monitoring**, using LLMs to gather evidence, track regulatory updates, generate audit reports, and flag policy violations.

### CPG / Retail

For CPG and retail organizations, the platform delivers accurate, AI-driven **demand planning** forecasts aligned with seasonal trends and real-time market signals. It enables dynamic **supply chain modeling** of the entire logistics network for cost optimization, groups stores through **store clustering** based on performance, demographics, and buying patterns, and uses predictive analytics for **markdown optimization** to time price reductions that maximize sell-through while minimizing revenue loss.

### Banking & Capital Markets

In banking and capital markets, the platform powers real-time **fraud detection** through predictive modeling on transaction data, AI-driven **risk management** with automated risk scoring and portfolio optimization, and automated **client segmentation** with personalized financial planning recommendations based on clustering analysis.

### Insurance

For insurance carriers, the platform enhances **underwriting** accuracy through AI-driven risk assessments that improve consistency and speed of decision-making, automates **claims management** workflows with intelligent processing and orchestration, and supports dynamic **pricing** strategies using historical data analysis and market trend modeling.

---

## Tech Stack Summary

| Layer | Technologies |
| ----- | ------------ |
| **ML/Deep Learning** | PyTorch, TensorFlow, Keras, Scikit-learn, XGBoost |
| **Data Processing** | PySpark, Pandas, NumPy, Airbyte, Kafka |
| **Infrastructure** | AWS (EKS, EC2, S3, EBS, VPC, IAM, KMS), Docker |
| **Orchestration** | Kubernetes, Karpenter, ArgoCD, Argo Workflows, Crossplane |
| **Observability** | Grafana, Prometheus, Mimir, Loki, Tempo, Grafana OnCall |
| **Messaging** | RabbitMQ, Kafka |
| **Databases** | PostgreSQL, Redis, Qdrant Vector DB |
| **Gen AI** | AWS Bedrock, Claude 3.5, LLAMA 70B, Titan Embeddings |
| **Security** | WAF, GuardDuty, CloudTrail, OIDC/SSO, MFA |
| **IaC** | Crossplane, Argo Workflows |

---

## Lessons Learned

### 1. Automate Everything or Users Won't Adopt It

Non-technical users have zero tolerance for manual configuration. If the time series module required users to "set the frequency" or "configure lag features," adoption would have been zero. Every step that could be automated, was.

### 2. Tenant Isolation is Non-Negotiable at Enterprise Scale

Shared-tenancy architectures don't fly in regulated industries. Our Master-Child account structure was more expensive to build and operate, but it unlocked healthcare, banking, and insurance verticals that would have been impossible otherwise.

### 3. Observability Pays for Itself

Investing in Grafana + Mimir + Loki + Tempo from day one meant we caught issues before customers reported them. In a multi-tenant SaaS platform, a monitoring gap is a customer churn risk.

### 4. Model Governance is a Feature, Not a Burden

Enterprise clients don't just want predictions — they want to know **why** a prediction was made, **when** the model was trained, and **what data** it used. SHAP values, data lineage, and version tracking aren't nice-to-haves — they're requirements.

### 5. No-Code Doesn't Mean No Complexity

Building a no-code interface that hides complexity while preserving flexibility is harder than building the underlying ML pipeline. The visual workflow builder took more engineering effort than the actual model training infrastructure.
