import streamlit as st
from components.navigation import create_navigation
from components.styles import apply_global_styles

def main():
    # 设置页面配置必须是第一个 Streamlit 命令
    st.set_page_config(
        page_title="智能助手系统",
        page_icon="🤖",
        layout="wide"
    )
    
    # 应用全局样式
    apply_global_styles()
    
    # 创建导航
    create_navigation()
    
    # 获取当前页面
    current_page = st.query_params.get("page", ["dashboard"])[0]
    
    # 根据当前页面显示内容
    if current_page == "dashboard":
        st.title("📊 数据分析仪表板")
        st.write("数据分析功能开发中...")
    elif current_page == "chatbot":
        st.title("🤖 智能助手")
        st.write("正在加载智能助手...")

if __name__ == "__main__":
    main()
