import streamlit as st
from components.navigation import create_navigation
from components.styles import apply_global_styles
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def main():
    # 设置页面配置必须是第一个 Streamlit 命令
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # 应用全局样式
    apply_global_styles()
    
    # 创建导航
    # create_navigation()

    # 页面标题
    st.title("📊 Data Analysis Dashboard")

    
    # # 创建三列布局
    # col1, col2, col3 = st.columns(3)
    #
    # # 第一列：数据概览
    # with col1:
    #     st.markdown('<div class="card">', unsafe_allow_html=True)
    #     st.subheader("数据概览")
    #     st.metric("总数据量", "1,234", "+123")
    #     st.metric("平均响应时间", "0.45s", "-0.1s")
    #     st.markdown('</div>', unsafe_allow_html=True)
    #
    # # 第二列：图表展示
    # with col2:
    #     st.markdown('<div class="card">', unsafe_allow_html=True)
    #     st.subheader("性能趋势")
    #     # 这里可以添加图表
    #     st.line_chart([1, 2, 3, 4, 5])
    #     st.markdown('</div>', unsafe_allow_html=True)
    #
    # # 第三列：操作面板
    # with col3:
    #     st.markdown('<div class="card">', unsafe_allow_html=True)
    #     st.subheader("操作面板")
    #     st.button("刷新数据")
    #     st.button("导出报告")
    #     st.markdown('</div>', unsafe_allow_html=True)
    #
    # # 底部数据表格
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    # st.subheader("详细数据")
    # # 这里可以添加数据表格
    # st.dataframe({
    #     '日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
    #     '访问量': [100, 200, 300],
    #     '转化率': [0.1, 0.2, 0.3]
    # })
    # st.markdown('</div>', unsafe_allow_html=True)
    # 展示药品评论聚类结果
    st.header("Analysis of Clustering in Drug Reviews")
    st.subheader("Clustering result")
    root_path = os.getcwd()
    cluster_img_path = os.path.join(root_path, "Deep Embedded Cluster", "cluster_result.png")
    st.image(cluster_img_path, width=1000)
    # 展示销量预测误差图片
    st.header("Analysis of Drug Sales Prediction")
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "LSTM")
    col_img, col_desc = st.columns([6, 4])
    # 左侧图片列
    with col_img:
        st.subheader("Overall error")
        image_path = os.path.join(data_path, "Predict_error_img.png")
        # if image_path.exists():
        #     st.image(image_path, use_column_width=True)
        # else:
        #     st.error(f"Image not found, please check the path: {image_path}")
        st.image(image_path, use_container_width=True)
    # 右侧说明列
    with col_desc:
        st.subheader("Parameters")
        st.markdown("""
        **Based on comparative multi-round experiments and error analysis,**
        **we can roughly determine the optimal parameter settings for LSTM as follows:**
        """)

        # 使用HTML/CSS美化表格
        st.markdown("""
        <style>
        .param-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .param-table th, .param-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .param-table th {
            background-color: #f2f2f2;
        }
        .param-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        </style>

        <table class="param-table">
            <tr>
                <th>Parameter Name</th>
                <th>Parameter Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>epochs</td>
                <td>20</td>
                <td>Number of training iterations</td>
            </tr>
            <tr>
                <td>batch_size</td>
                <td>8</td>
                <td>Number of training samples per batch</td>
            </tr>
            <tr>
                <td>lookback_steps</td>
                <td>12</td>
                <td>Lookback time steps</td>
            </tr>
            <tr>
                <td>forecast_steps</td>
                <td>12</td>
                <td>Forecast time steps</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)


    # 绘制药品销量曲线


    def load_sales_data():
        """加载销售数据（自动识别后12个月为预测值）"""
        result_path = os.path.join(data_path, "LSTM_result.csv")
        try:
            df = pd.read_csv(result_path, parse_dates=['ds'])
            df['period'] = 'history_value'
            df.iloc[-12:, df.columns.get_loc('period')] = 'predict_value'
            return df
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None

    def create_comparison_plot(selected_drugs, df):
        """创建对比图表（Matplotlib实现）"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 颜色和样式配置
        colors = plt.cm.tab10.colors  # 使用tab10调色板
        styles = {
            'history_value': {'linestyle': '-', 'linewidth': 2, 'marker': 'o'},
            'predict_value': {'linestyle': '--', 'linewidth': 1.5, 'marker': 'x'}
        }

        for i, drug in enumerate(selected_drugs):
            for period in ['history_value', 'predict_value']:
                data = df[df['period'] == period]
                ax.plot(
                    data['ds'],
                    data[drug],
                    label=f"{drug} ({period})",
                    color=colors[i % len(colors)],
                    **styles[period]
                )

        # 标记预测起始点
        if 'predict_value' in df['period'].values:
            forecast_start = df[df['period'] == 'predict_value']['ds'].iloc[0]
            ax.axvline(forecast_start, color='red', linestyle=':', alpha=0.7)
            ax.text(
                forecast_start,
                ax.get_ylim()[1] * 0.95,
                'Forecast Start',
                color='red',
                ha='center',
                bbox=dict(facecolor='white', alpha=0.8)
            )

        # 图表美化
        ax.set_title("Drug Sales Comparison Analysis", fontsize=16, pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sales Volume", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 日期格式优化
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        # 图例位置优化
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def drug_comparison_app():
        st.header("💊 Drug Sales Comparison Analysis")

        # 加载数据
        df = load_sales_data()
        if df is None:
            return

        # 药品选择器
        drug_list = [col for col in df.columns if col not in ('ds', 'period')]
        cols = st.columns(3)
        selected_drugs = []

        with cols[0]:
            drug1 = st.selectbox("Drug 1", drug_list, key="drug1")
            selected_drugs.append(drug1)

        with cols[1]:
            if len(drug_list) > 1:
                drug2 = st.selectbox("Drug 2 (Optional)", [""] + [d for d in drug_list if d != drug1], key="drug2")
                if drug2: selected_drugs.append(drug2)

        with cols[2]:
            if len(drug_list) > 2:
                remaining = [d for d in drug_list if d not in selected_drugs]
                drug3 = st.selectbox("Drug 3 (Optional)", [""] + remaining, key="drug3")
                if drug3: selected_drugs.append(drug3)

        if not selected_drugs:
            st.warning("Please select at least one drug")
            return

        # 创建图表
        st.subheader("Sales Trend Comparison")
        fig = create_comparison_plot(selected_drugs, df)
        st.pyplot(fig)

        # 添加图例说明
        st.markdown("""
            <div style="display: flex; gap: 20px; margin-top: -15px; font-size: 14px;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 0; border-top: 2px solid #1f77b4; margin-right: 5px;"></div>
                    <span>Solid line=history_value</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 0; border-top: 2px dashed #ff7f0e; margin-right: 5px;"></div>
                    <span>Dashed line=predict_value</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 20px; height: 0; border-top: 1px dotted red; margin-right: 5px;"></div>
                    <span>Vertical line=Forecast Start</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    drug_comparison_app()

if __name__ == "__main__":
    main() 