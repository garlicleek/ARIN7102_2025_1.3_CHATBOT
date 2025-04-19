import streamlit as st
from components.navigation import create_navigation
from components.styles import apply_global_styles

def main():
    # 设置页面配置必须是第一个 Streamlit 命令
    st.set_page_config(
        page_title="数据分析仪表板",
        page_icon="📊",
        layout="wide"
    )
    
    # 应用全局样式
    apply_global_styles()
    
    # 创建导航
    create_navigation()
    
    # 页面标题
    st.title("📊 数据分析仪表板")
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    # 第一列：数据概览
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("数据概览")
        st.metric("总数据量", "1,234", "+123")
        st.metric("平均响应时间", "0.45s", "-0.1s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第二列：图表展示
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("性能趋势")
        # 这里可以添加图表
        st.line_chart([1, 2, 3, 4, 5])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第三列：操作面板
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("操作面板")
        st.button("刷新数据")
        st.button("导出报告")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 底部数据表格
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("详细数据")
    # 这里可以添加数据表格
    st.dataframe({
        '日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
        '访问量': [100, 200, 300],
        '转化率': [0.1, 0.2, 0.3]
    })
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 