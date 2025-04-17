import streamlit as st

def apply_global_styles():
    st.markdown("""
    <style>
    /* 全局样式 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 主容器样式 */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* 卡片样式 */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* 标题样式 */
    h1 {
        color: #2c3e50;
        margin-bottom: 20px;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True) 