# PharmaSales Analytics Chatbot System

## 📑 Project Overview  
This project implements an **intelligent pharmacy sales analytics chatbot** using Agentic RAG architecture and multi-agent collaboration. The system integrates dual data analysis pipelines (text sentiment analysis & sales sequence prediction) with natural language interaction capabilities. Core components include:

1. **Dual-Modal Data Analysis**:
   - Text sentiment analysis on pharmaceutical reviews
   - Time-series forecasting for drug sales using LSTM
2. **Agentic RAG Framework**:
   - Multi-agent collaboration for pharmaceutical data Q&A
   - Context-aware retrieval and dynamic response generation
3. **Interactive Web Interface**:
   - Natural language query interface
   - Visual analytics dashboard


## 🛠 Technical Architecture  
### 1. Data Processing Pipeline
**Data Sources**:
- Drug Review: https://www.kaggle.com/datasets/mohamedabdelwahabali/drugreview
- Pharmacy Sales:

**Pre-processing**:

**Clustering**:

**Reviews Analyse**:

**Sales Analyse**:

### 2. Multi-Agent System
**Agentic RAG Architecture**:  
![rag_architecture.png](media\rag_architecture.png)

### 3. Frontend Implementation


## ⚙️ Installation & Deployment
### Prerequisites
1. Create your environment configuration
```sh
cp .env.example .env
```

2. Create a virtual environment and install dependencies in your python:
```commandline
pip install uv
uv sync --frozen
```

3. run server and streamlit
enter the virtual environment
```commandline
cd .venv/Scripts && activate.bat
```
run service:
```commandline
python src/run_service.py
```

run streamlit:
```commandline
streamlit run src/streamlit_app.py
```

### Configuration
If you need to run at local environment, please change `.env`
```.env
HOST=localhost
```

Please first add any API key before running
```.env
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
```

## 🗺 Implementation Roadmap
做完了这里可以画个甘特图
```
gantt
    title Project Milestones
    dateFormat  YYYY-MM-DD
    section Data Layer
    Text Data Cleaning       :
    Time-series DB Deployment :
    
    section Model Layer
    Sentiment Model Training  :
    Hybrid Forecast Tuning    :
    -
    section Interaction Layer
    Dialog Logic Development  :
    Report Template Design     :
```

## 📚 References
### 1. In Using
- framework:https://github.com/JoshuaC215/agent-service-toolkit
preparing to use:
- RAG agent:https://github.com/NVIDIA/workbench-example-agentic-rag
- RAG agent2:https://github.com/SciPhi-AI/R2R/blob/main/py/core/agent/rag.py

### 2. Only for reference
- Basic Knowledge:
  - RAG Intro: https://www.zhihu.com/question/638503601/answer/3384081209
  - RAG Framework: https://zhuanlan.zhihu.com/p/19229901774
- UI: 
  - Chatbot: https://github.com/rag-web-ui/rag-web-ui
  - Chatbot: https://github.com/karthik-codex/Autogen_GraphRAG_Ollama
  - Dashboard: https://github.com/ColorlibHQ/gentelella
- Multi Agent: 
  - Overview: https://github.com/DSXiangLi/DecryptPrompt
  - InsightLens: https://arxiv.org/abs/2404.01644
  - Agentic RAG Framework: https://github.com/asinghcsu/AgenticRAG-Survey