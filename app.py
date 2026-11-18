import streamlit as st
import pandas as pd
import numpy as np

# 设置页面宽屏
st.set_page_config(page_title="成都二手房决策支持系统", layout="wide")

st.title("🏠 成都二手房价格影响因素分析与购房决策支持系统")
st.write("本系统基于特征价格模型与机器学习，为您提供市场概览与单套性价比判断。（v0.1 初代框架）")

# 侧边栏：用户交互筛选
st.sidebar.header("🔍 市场筛选条件")
district = st.sidebar.selectbox("选择行政区域", ["全成都", "锦江区", "高新区", "武侯区", "郫都区"])

# 模拟一部分你的“已落盘数据集（D1）”数据
# 在真实项目中，这里会替换为 pd.read_csv('你的数据.csv')
@st.cache_data
def load_mock_data():
    data = {
        "小区": ["万科城市花园", "中海城南一号", "保利心语", "华润凤凰城", "绿地锦天府"],
        "区域": ["锦江区", "高新区", "武侯区", "高新区", "锦江区"],
        "面积(㎡)": [89, 120, 75, 140, 105],
        "挂牌价(万)": [210, 360, 150, 420, 300],
        "成交价(万)": [195, 330, 138, 390, 270],
        "楼层": ["中楼层", "高楼层", "低楼层", "中楼层", "高楼层"]
    }
    df = pd.DataFrame(data)
    df["挂牌单价(元/㎡)"] = (df["挂牌价(万)"] * 10000 / df["面积(㎡)"]).astype(int)
    df["成交单价(元/㎡)"] = (df["成交价(万)"] * 10000 / df["面积(㎡)"]).astype(int)
    df["折价率(%)"] = (df["成交价(万)"] / df["挂牌价(万)"] * 100).round(2)
    return df

df = load_mock_data()

# 市场概览（F2 功能）
st.subheader("📊 市场概览看板")
col1, col2, col3 = st.columns(3)
col1.metric("当前展示房源数", f"{len(df)} 套")
col2.metric("平均成交折价率", f"{df['折价率(%)'].mean():.2f}%", delta="-10.31% (相比挂牌价)")
col3.metric("平均成交单价", f"{df['成交单价(元/㎡)'].mean():,.0f} 元/㎡")

st.markdown("---")

# 挂牌价与成交价对比（特征价格模型的基础实证）
st.subheader("📉 挂牌价与成交价系统性偏差分析")
chart_data = df[["小区", "挂牌单价(元/㎡)", "成交单价(元/㎡)"]].set_index("小区")
st.bar_chart(chart_data)

st.markdown("---")

# 房源明细（F4 模块雏形）
st.subheader("🏘️ 房源明细数据")
if district != "全成都":
    filtered_df = df[df["区域"] == district]
else:
    filtered_df = df
st.dataframe(filtered_df, use_container_width=True)

st.info
