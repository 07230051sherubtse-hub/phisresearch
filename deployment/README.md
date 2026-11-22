# PhishResearch Assistant - Deployment Guide

## 1. Local Deployment

### Requirements
- Python 3.10 or later
- Git
- Internet connection

### Steps

1. Clone the GitHub repository.
2. Open the project folder.
3. Run `deployment\run_local.bat`.

Alternatively, use these commands:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run phishresearch_app/app.py

## 2. Online Deployment

The application is intended for deployment using Streamlit Community Cloud.

### Steps

1. Push the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `phishresearch_app/app.py` as the application entry point.
4. Configure the required dependencies and model/retrieval artifact access.

### Important Note

The Qwen model weights, research corpus, and precomputed retrieval artifacts are not included in the GitHub repository. These resources must be made available separately for the complete application to function.
