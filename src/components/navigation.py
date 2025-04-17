import streamlit as st

def create_navigation():
    st.markdown("""
    <style>
    .nav-container {
        display: flex;
        justify-content: space-around;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .nav-item {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .nav-item:hover {
        background-color: #e1e4e8;
    }
    .nav-item.active {
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 获取当前页面
    current_page = st.query_params.get("page", ["dashboard"])[0]
    
    # 创建导航栏
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 数据分析仪表板", key="nav_dashboard"):
            st.query_params["page"] = "dashboard"
            st.rerun()
    with col2:
        if st.button("🤖 智能助手", key="nav_chatbot"):
            st.query_params["page"] = "chatbot"
            st.rerun()
    
    # 添加样式类
    st.markdown(f"""
    <script>
    document.querySelector('[data-testid="stButton"][aria-label="nav_{current_page}"]').parentElement.classList.add('active');
    </script>
    """, unsafe_allow_html=True) 