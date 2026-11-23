import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 设置页面宽屏
st.set_page_config(page_title="成都二手房决策支持系统", layout="wide", page_icon="🏠")

# ================= 全局变量 =================
# 提到全局，确保估价计算器也能读取到
district_base_price = {'锦江区': 25000, '高新区': 23000, '武侯区': 18000, '青羊区': 20000, '成华区': 17000, '郫都区': 12000}

# ================= 模拟数据生成 =================
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500  # 模拟500套房源
    districts = list(district_base_price.keys())
    
    df = pd.DataFrame({
        '小区': [f"小区_{i}" for i in range(n)],
        '行政区域': np.random.choice(districts, n),
        '建筑面积(㎡)': np.random.randint(50, 200, n),
        '楼层': np.random.choice(['低楼层', '中楼层', '高楼层'], n),
        '房龄(年)': np.random.randint(0, 25, n),
        '装修': np.random.choice(['毛坯', '简装', '精装'], n, p=[0.2, 0.5, 0.3])
    })
    
    # 根据区域、面积、房龄等计算模拟单价和总价
    base = df['行政区域'].map(district_base_price)
    df['挂牌单价(元/㎡)'] = (base * (1 - df['房龄(年)'] * 0.01) * np.random.uniform(0.9, 1.1, n)).astype(int)
    df['挂牌总价(万)'] = (df['挂牌单价(元/㎡)'] * df['建筑面积(㎡)'] / 10000).round(1)
    
    # 模拟成交价：基础折价率 90%-97% 之间，加上随机扰动
    discount = np.random.uniform(0.90, 0.97, n)
    df['成交总价(万)'] = (df['挂牌总价(万)'] * discount).round(1)
    df['成交单价(元/㎡)'] = (df['成交总价(万)'] * 10000 / df['建筑面积(㎡)']).astype(int)
    df['折价率(%)'] = (df['成交总价(万)'] / df['挂牌总价(万)'] * 100).round(2)
    
    return df

df = load_data()

# ================= 侧边栏筛选 =================
st.sidebar.title("🔍 数据筛选")
selected_districts = st.sidebar.multiselect("选择行政区域", df['行政区域'].unique(), default=df['行政区域'].unique())
filtered_df = df[df['行政区域'].isin(selected_districts)]

# ================= 主页面 =================
st.title("🏠 成都二手房价格影响因素分析与购房决策支持系统")
st.markdown("本系统基于 **11.8万条** 成都二手房成交记录（模拟集），利用特征价格模型与机器学习，为您提供市场洞察与单套性价比判断。")

# 分页展示
tab1, tab2, tab3 = st.tabs(["📊 市场概览看板", "📈 价格影响洞察", "🤖 智能估价与性价比"])

# --- Tab 1: 市场概览看板 ---
with tab1:
    st.subheader("核心市场指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前筛选房源", f"{len(filtered_df)} 套")
    col2.metric("平均挂牌单价", f"{filtered_df['挂牌单价(元/㎡)'].mean():,.0f} 元/㎡")
    col3.metric("平均成交单价", f"{filtered_df['成交单价(元/㎡)'].mean():,.0f} 元/㎡")
    
    # 平均折价率
    avg_discount = 100 - filtered_df['折价率(%)'].mean()
    col4.metric("平均议价空间", f"{avg_discount:.2f}%", delta="挂牌价高于成交价", delta_color="inverse")

    st.markdown("---")
    st.subheader("各行政区挂牌与成交单价对比")
    # 聚合区域数据
    dist_stats = filtered_df.groupby('行政区域')[['挂牌单价(元/㎡)', '成交单价(元/㎡)']].mean().reset_index()
    
    # Plotly 分组柱状图
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=dist_stats['行政区域'], y=dist_stats['挂牌单价(元/㎡)'], name='挂牌单价', marker_color='#636EFA'))
    fig1.add_trace(go.Bar(x=dist_stats['行政区域'], y=dist_stats['成交单价(元/㎡)'], name='成交单价', marker_color='#EF553B'))
    fig1.update_layout(barmode='group', template='plotly_white', height=400)
    st.plotly_chart(fig1, use_container_width=True)

# --- Tab 2: 价格影响洞察 ---
with tab2:
    st.subheader("特征价格模型 (HPM) 洞察")
    st.write("分析各因素对房价的隐含溢价。以下图表模拟了 SHAP 模型的特征重要性排序：")
    
    col_a, col_b = st.columns(2)
    with col_a:
        # 模拟特征重要性（SHAP）
        features = ['建筑面积', '行政区域', '房龄', '装修状况', '楼层']
        importance = [0.45, 0.25, 0.15, 0.10, 0.05]
        fig2 = px.bar(x=importance, y=features, orientation='h', 
                      title="特征对价格的贡献度 (SHAP值)",
                      labels={'x': '重要性权重', 'y': '特征'},
                      color=importance, color_continuous_scale='Blues')
        fig2.update_layout(template='plotly_white', height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_b:
        # 面积与单价关系散点图
        fig3 = px.scatter(filtered_df, x='建筑面积(㎡)', y='挂牌单价(元/㎡)', 
                          color='行政区域', size='挂牌总价(万)', opacity=0.6,
                          title="面积与单价分布（气泡大小代表总价）")
        fig3.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig3, use_container_width=True)

# --- Tab 3: 智能估价与性价比 ---
with tab3:
    st.subheader("单套估价与性价比计算器")
    st.write("输入房源特征，模型将给出参考价区间，并与同商圈均价对比。")
    
    with st.form("valuation_form"):
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            in_district = st.selectbox("所在区域", df['行政区域'].unique())
            in_area = st.number_input("建筑面积 (㎡)", min_value=30, max_value=300, value=90, step=10)
        with col_y:
            in_age = st.slider("房龄 (年)", 0, 30, 5)
            in_floor = st.selectbox("楼层", ['低楼层', '中楼层', '高楼层'])
        with col_z:
            in_deco = st.selectbox("装修状况", ['毛坯', '简装', '精装'])
            submit = st.form_submit_button("🚀 生成估价报告")
            
    if submit:
        # 简化的估价逻辑（模拟 XGBoost 模型）
        base_price = district_base_price[in_district]  # 现在这里不会报错了
        deco_factor = {'毛坯': 0.95, '简装': 1.0, '精装': 1.05}[in_deco]
        age_factor = max(0.7, 1 - in_age * 0.01)
        floor_factor = {'低楼层': 0.98, '中楼层': 1.02, '高楼层': 1.0}[in_floor]
        
        est_unit_price = base_price * deco_factor * age_factor * floor_factor
        est_total = est_unit_price * in_area / 10000
        
        st.success(f"### 估价结果：本套房源参考总价约为 **{est_total:.1f} 万元**")
        st.info(f"**单价参考：** {est_unit_price:,.0f} 元/㎡。\n\n**模型提示：** 在此区域、房龄和装修条件下，建议挂牌价可设在 {est_total * 1.05:.1f} 万元左右，预留约 5%-8% 的议价空间。")
        
        # 性价比对比
        st.markdown("#### 📊 性价比对比")
        district_median = filtered_df[filtered_df['行政区域'] == in_district]['挂牌单价(元/㎡)'].median()
        diff_pct = (est_unit_price - district_median) / district_median * 100
        
        if diff_pct < -5:
            st.write(f"✅ **高性价比**：该房源预估单价比该区域中位数低 **{-diff_pct:.1f}%**。如果实际挂牌价在这个区间，强烈建议关注。")
        elif diff_pct > 5:
            st.write(f"⚠️ **存在溢价**：该房源预估单价比该区域中位数高 **{diff_pct:.1f}%**。请仔细考量楼层或装修是否有特殊优势。")
        else:
            st.write("⚖️ **价格中规中矩**：该房源预估单价与该区域中位数基本持平。")
