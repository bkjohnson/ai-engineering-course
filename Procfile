api: uvicorn main:app --reload --port ${API_PORT:-8000}
ui: API_BASE_URL=http://127.0.0.1:${API_PORT:-8000} streamlit run streamlit_app.py --server.port ${UI_PORT:-8501} --server.headless true
