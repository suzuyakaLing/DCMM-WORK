import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# 设置页面标题和布局
st.set_page_config(
    page_title="数据质量",
    layout="wide"
)

# 添加标题
st.markdown("<h1 style='text-align: center;'>数据质量  淘宝 vs 沃尔玛</h1>", unsafe_allow_html=True)

# 在侧边栏创建导航目录
st.sidebar.title("导航目录")
page = st.sidebar.radio(
    "请选择要查看的内容：",
    ["项目概述", "数据质量需求","数据质量检查"]
)

# 根据选择显示不同的内容
if page == "项目概述":
    st.header("项目背景与目标")
    st.markdown("""
    <div style='font-size: 20px;'>
    基于数据质量项目的性质，认为一个案例不能很好地体现其重要性与作用，所以找了国内案例淘宝与国外案例沃尔玛，对于它们双方的关于它们电商平台的数据集的数据质量进行对比分析评价指标，发现各自优劣，并对双方的数据集各自进行优化提升并展示优化效果，最后能够互相学习借鉴数据质量管理方法。不足之处在于所查找到的双方的数据集所在的业务范围并不是完全一致，这点可能会导致后续一些判断或者是对比时产生不可抗力的原因导致说服力并不是很有效。
    </div>
    """, unsafe_allow_html=True)

    st.header("数据集简要介绍")
    st.subheader("淘宝数据集")
    st.markdown("""
    <div style='font-size: 20px;'>
    数据集来源：<a href='https://tianchi.aliyun.com/dataset/56' target='_blank'>阿里天池淘宝用户行为数据集</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("沃尔玛数据集")
    st.markdown("""
    <div style='font-size: 20px;'>
    数据集来源：<a href='https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data?select=features.csv.zip' target='_blank'>Walmart Recruiting - Store Sales Forecasting | Kaggle</a>
    </div>
    """, unsafe_allow_html=True)

    st.header("使用工具")
    st.markdown("""
    <div style='font-size: 20px;'>
    - Trae：汇报演示与模型展示
    - FineBI：数据质量检查
    - SQLServer：数据质量检查中关联性视图绘制
    </div>
    """, unsafe_allow_html=True)

elif page == "数据质量需求":
    st.header("数据质量需求")
    
    # 创建选项卡
    tab1, tab2 = st.tabs(["评估指标定义", "数据质量规则"])
    
    with tab1:
        # 创建表格数据
        data = {
            "质量维度": ["规范性", "准确性", "准确性", "准确性", "唯一性", "唯一性",
                      "完整性", "完整性", "一致性", "一致性", "一致性", "关联性"],
            "质量评估指标": ["字段类型", "数值型字段范围", "文本型字段范围", "日期型字段范围",
                        "字段数据唯一率", "字段含义与作用唯一", "字段数据完整率", "字段含义明确率",
                        "字段命名一致率", "字段格式一致率", "字段含义一致率", "主键约束/外键约束"],
            "指标计算方法": ["", "(数值范围异常行数/实际行数)*100%", "(格式或字符长度异常行数/实际行数)*100%",
             "(格式或日期范围异常行数/实际行数)*100%", "(实际唯一的编码数量/编码数量)*100%",
              "", "（非空行数/实际行数）*100%", "(含义明确的字段数/表内实际字段数)*100%", "(表共有字段命名一致组数/实际组数)*100%", 
              "(表共有字段格式一致组数/实际组数)*100%", "(表共有字段含义一致组数/实际组数)*100%", ""],
            "参考阈值": ["", "所有表的所有字段异常率=0%", "所有表的所有字段异常率=0%", "所有表的所有字段异常率=0%",
             "所有表的类主键字段唯一率=100%", "表内的所有字段含义与作用唯一", "重要字段=100%,较重要字段=95%,次重要字段=90%,一般重要字段=80%",
              "所有表明确率=100%", "所有表共有字段命名一致率=100%", "所有表共有字段格式一致率=100%", "所有表共有字段含义一致率=100%", ""],
            "改善措施": ["按逻辑和业务所需更改字段类型",
                      "对异常数据进行清洗",
                      "对异常数据进行清洗",
                      "对异常数据进行清洗",
                      "删除冗余信息",
                      "按业务需求合并字段或删除冗余字段",
                      "按业务需求填充/清洗缺失数据",
                      "按逻辑和相关资料推断和补全含义",
                      "以父表为依据修改字段命名",
                      "以父表为依据修改字段格式",
                      "以父表为依据修改字段含义",
                      "建立主外键约束关系"]
        }
        
        # 使用 Pandas DataFrame 创建表格
        import pandas as pd
        df = pd.DataFrame(data)
        
        # 创建正六边形图
        import plotly.graph_objects as go
        import numpy as np
        
        # 定义六边形的顶点
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6个点，去掉最后一个重复点
        r = [1]*6  # 统一半径
        categories = ['规范性', '准确性', '唯一性', '完整性', '一致性', '关联性']
        
        # 创建图形
        fig = go.Figure()
        
        # 添加六边形
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=categories,
            fill='toself',
            line=dict(color='blue'),
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        # 设置图形样式
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=False,  # 隐藏径向轴
                    range=[0, 1]
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=60,
                    tickfont=dict(size=20, color='black', family='Arial Bold')  # 增大字体
                )
            ),
            showlegend=False,
            width=400,  # 减小图形宽度
            height=400,  # 减小图形高度
            margin=dict(l=50, r=50, t=30, b=30)  # 调整边距
        )
        
        # 创建三列布局，中间放置图形
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # 显示图表
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示表格
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        # 创建数据集选择器
        dataset = st.selectbox(
            "请选择要查看的数据集：",
            ["淘宝数据集", "沃尔玛数据集"]
        )
        
        if dataset == "淘宝数据集":
            st.subheader("淘宝数据集质量规则")
            # 创建淘宝数据集的表选项卡
            taobao_tabs = st.tabs(["用户基本信息表", "广告基本信息表", "用户行为日志表", "原始样本表"])
            
            with taobao_tabs[0]:
                st.write("用户基本信息表(user_profile)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["userid", "cms_segid", "cms_group_id", "final_gender_code", "age_level", 
                     "pvalue_level", "shopping_level", "occupation", "new_user_class_level"]
                )
                
                # 字段信息字典
                fields_info = {
                    "userid": {
                        "字段含义": "用户ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "cms_segid": {
                        "字段含义": "微群ID",
                        "字段类型": "文本型",
                        "字段范围": "0-96",
                        "关联性": "无",
                        "非空约束": "不需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "cms_group_id": {
                        "字段含义": "微群分组ID",
                        "字段类型": "数值型",
                        "字段范围": "0-12",
                        "关联性": "无",
                        "非空约束": "不需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "final_gender_code": {
                        "字段含义": "性别 (1:男,2:女)",
                        "字段类型": "数值型",
                        "字段范围": "1，2",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "age_level": {
                        "字段含义": "年龄层次",
                        "字段类型": "数值型",
                        "字段范围": "0-6",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "pvalue_level": {
                        "字段含义": "消费档次（1:低档，2:中档，3:高档）",
                        "字段类型": "数值型",
                        "字段范围": "1，2，3",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "shopping_level": {
                        "字段含义": "购物深度（1:浅层用户,2:中度用户,3:深度用户）",
                        "字段类型": "数值型",
                        "字段范围": "1，2，3",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "occupation": {
                        "字段含义": "是否大学生（1:是,0:否）",
                        "字段类型": "数值型",
                        "字段范围": "1，0",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "new_user_class_level": {
                        "字段含义": "城市层级",
                        "字段类型": "数值型",
                        "字段范围": "1,2,3,4",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    }
                }
                
                # 创建三列布局展示字段信息
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
            with taobao_tabs[1]:
                st.write("广告基本信息表(ad_feature)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["adgroup_id", "cate_id", "campaign_id", "customer_id", "brand", "price"]
                )
                
                # 字段信息字典
                fields_info = {
                    "adgroup_id": {
                        "字段含义": "广告ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "cate_id": {
                        "字段含义": "商品类目ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "campaign_id": {
                        "字段含义": "广告计划ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "不需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "customer_id": {
                        "字段含义": "广告主ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "不需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "brand": {
                        "字段含义": "品牌ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "price": {
                        "字段含义": "宝贝的价格",
                        "字段类型": "数值型",
                        "字段范围": "price>0",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
            with taobao_tabs[2]:
                st.write("用户行为日志表(behavior_log)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["user", "time_stamp", "btag", "cate", "brand"]
                )
                
                # 字段信息字典
                fields_info = {
                    "user": {
                        "字段含义": "用户ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "主键/外键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "time_stamp": {
                        "字段含义": "时间戳",
                        "字段类型": "日期型",
                        "字段范围": "",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "btag": {
                        "字段含义": "行为类型",
                        "字段类型": "文本型",
                        "字段范围": "pv:浏览,cart:加入购物车,fav:喜欢,buy:购买",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "cate": {
                        "字段含义": "脱敏过的商品类目ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "brand": {
                        "字段含义": "脱敏过的品牌ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
            with taobao_tabs[3]:
                st.write("原始样本表(raw_sample)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["user", "adgroup_id", "time_stamp", "pid", "nonclk", "clk"]
                )
                
                # 字段信息字典
                fields_info = {
                    "user": {
                        "字段含义": "用户ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "主键/外键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "adgroup_id": {
                        "字段含义": "广告ID",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "主键/外键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "time_stamp": {
                        "字段含义": "时间戳",
                        "字段类型": "日期型",
                        "字段范围": "",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "pid": {
                        "字段含义": "资源位",
                        "字段类型": "文本型",
                        "字段范围": "",
                        "关联性": "无",
                        "非空约束": "不需要",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "nonclk": {
                        "字段含义": "没有点击:1；点击:0",
                        "字段类型": "数值型",
                        "字段范围": "1，0",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "clk": {
                        "字段含义": "没有点击:0;点击:1",
                        "字段类型": "数值型",
                        "字段范围": "1，0",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
        else:
            st.subheader("沃尔玛数据集质量规则")
            # 创建沃尔玛数据集的表选项卡
            walmart_tabs = st.tabs(["商店信息表", "部门销售表", "消费环境表"])
            
            with walmart_tabs[0]:
                st.write("商店信息表(stores)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["Store", "Type", "Size"]
                )
                
                # 字段信息字典
                fields_info = {
                    "Store": {
                        "字段含义": "商店编号",
                        "字段类型": "文本型",
                        "字段范围": "1-45",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "Type": {
                        "字段含义": "商店类型",
                        "字段类型": "文本型",
                        "字段范围": "A,B,C",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    },
                    "Size": {
                        "字段含义": "商店规模(平方英尺)",
                        "字段类型": "数值型",
                        "字段范围": "30000-220000",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "不需要",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
            with walmart_tabs[1]:
                st.write("部门销售表(train)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
                )
                
                # 字段信息字典
                fields_info = {
                    "Store": {
                        "字段含义": "商店编号",
                        "字段类型": "文本型",
                        "字段范围": "1-45",
                        "关联性": "主键/外键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "Dept": {
                        "字段含义": "部门编号",
                        "字段类型": "文本型",
                        "字段范围": "1-98",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "Date": {
                        "字段含义": "日期(周)",
                        "字段类型": "日期型",
                        "字段范围": "2010.2.5 -2013.7.26",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "Weekly_Sales": {
                        "字段含义": "该部门在该商店的周销售额",
                        "字段类型": "数值型",
                        "字段范围": ">=0",
                        "关联性": "无",
                        "非空约束": "95%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "IsHoliday": {
                        "字段含义": "是否为特殊节假日周",
                        "字段类型": "文本型",
                        "字段范围": "True,False",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
            with walmart_tabs[2]:
                st.write("消费环境表(features)质量规则")
                
                # 创建字段选择器
                selected_field = st.selectbox(
                    "选择要查看的字段：",
                    ["Store", "Date", "Temperature", "Fuel_Price", "MarkDown", "CPI", "Unemployment", "IsHoliday"]
                )
                
                # 字段信息字典
                fields_info = {
                    "Store": {
                        "字段含义": "商店编号",
                        "字段类型": "文本型",
                        "字段范围": "1-45",
                        "关联性": "主键/外键约束",
                        "非空约束": "需要",
                        "唯一性": "需要",
                        "一致性约束": "需要"
                    },
                    "Date": {
                        "字段含义": "日期(周)",
                        "字段类型": "日期型",
                        "字段范围": "2010.2.5-2013.7.26",
                        "关联性": "主键约束",
                        "非空约束": "需要",
                        "唯一性": "",
                        "一致性约束": "需要"
                    },
                    "Temperature": {
                        "字段含义": "该地区的平均温度(℉)",
                        "字段类型": "数值型",
                        "字段范围": "-20-120",
                        "关联性": "无",
                        "非空约束": "80%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "Fuel_Price": {
                        "字段含义": "该地区的燃油价格",
                        "字段类型": "数值型",
                        "字段范围": "$2.00-$5.00",
                        "关联性": "无",
                        "非空约束": "80%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "MarkDown": {
                        "字段含义": "促销降价(含义不明确)",
                        "字段类型": "数值型",
                        "字段范围": "权重数据",
                        "关联性": "无",
                        "非空约束": "无需",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "CPI": {
                        "字段含义": "消费者价格指数",
                        "字段类型": "数值型",
                        "字段范围": "2%-6%上涨幅度",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "Unemployment": {
                        "字段含义": "失业率(%)",
                        "字段类型": "数值型",
                        "字段范围": "3-14",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    },
                    "IsHoliday": {
                        "字段含义": "是否为特殊节假日周",
                        "字段类型": "文本型",
                        "字段范围": "True,False",
                        "关联性": "无",
                        "非空约束": "90%非空",
                        "唯一性": "无需",
                        "一致性约束": "需要"
                    }
                }
                
                # 复用之前的展示布局代码
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>基本信息</h4>
                        <p><b>字段名：</b> {selected_field}</p>
                        <p><b>字段含义：</b> {fields_info[selected_field]['字段含义']}</p>
                        <p><b>字段类型：</b> {fields_info[selected_field]['字段类型']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>约束条件</h4>
                        <p><b>字段范围：</b> {fields_info[selected_field]['字段范围']}</p>
                        <p><b>关联性：</b> {fields_info[selected_field]['关联性']}</p>
                        <p><b>非空约束：</b> {fields_info[selected_field]['非空约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h4>质量要求</h4>
                        <p><b>唯一性：</b> {fields_info[selected_field]['唯一性']}</p>
                        <p><b>一致性约束：</b> {fields_info[selected_field]['一致性约束']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 添加可视化度量
                quality_metrics = {
                    "规范性": 100 if fields_info[selected_field]['字段类型'] != "" else 0,
                    "完整性": 100 if fields_info[selected_field]['非空约束'] == "需要" else 90,
                    "唯一性": 100 if fields_info[selected_field]['唯一性'] == "需要" else 0,
                    "一致性": 100 if fields_info[selected_field]['一致性约束'] == "需要" else 0
                }
                
                # 创建进度条
                cols = st.columns(len(quality_metrics))
                for col, (metric, value) in zip(cols, quality_metrics.items()):
                    with col:
                        st.metric(metric, f"{value}%")
                        st.progress(value/100)
elif page == "数据质量检查":
    st.header("数据质量检查")

    # 创建两个选项卡
    overview_tab, results_tab = st.tabs(["检查概述", "检查结果"])
    
    with overview_tab:
        # 添加概述文字说明
        st.markdown("""
        ### 检查内容
        1. **规范性约束**：字段类型的检查与修改
        2. **关联性**：主外键约束
        3. **一致性约束**：字段命名、字段格式、字段含义检查
        4. **唯一性约束**：类主键的唯一性检查
        5. **完整性约束**：非空约束检查
        6. **准确性约束**：字段范围异常率检查

        ### 关联性视图
        """)
        st.image("assets/taobao.jpg")
        st.image("assets/walmart.jpg")

        st.markdown("""
        ### 检查过程举例
        以下是FineBI对于唯一性约束和非空约束检查操作视频
        """)
        st.video("assets\\sample.mp4")
        
    with results_tab:
        # 创建雷达图数据
        categories = ['准确性-字段范围异常率', '唯一性-字段数据唯一率', '唯一性-字段含义与作用唯一率', '完整性-字段数据完整率', 
                      '完整性-字段含义明确率', '一致性-字段命名一致率', '一致性-字段格式一致率', '一致性-字段含义一致率']
        taobao_values = [0, 100, 90, 85.71, 100, 60, 100, 100]
        walmart_values = [2.2, 100, 100, 100, 68.75, 100, 100, 100]

        # 创建雷达图
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=taobao_values,
            theta=categories,
            fill='toself',
            name='淘宝'
        ))
        fig.add_trace(go.Scatterpolar(
            r=walmart_values,
            theta=categories,
            fill='toself',
            name='沃尔玛'
        ))

        # 设置图形样式
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                ),
                angularaxis=dict(
                    tickfont=dict(size=15)
                )
            ),
            showlegend=True
        )

        # 显示雷达图
        st.plotly_chart(fig, use_container_width=True)

        # 添加对比分析说明
        st.markdown("""
        ### 数据质量对比分析
        
        #### 1. 共同优势
        两个数据集在以下方面都达到了100%的完美表现：
        - 唯一性-字段数据唯一率
        - 一致性-字段格式一致率
        - 一致性-字段含义一致率
        
        这表明两个数据集在大多数质量指标上都保持较高水平。
        
        #### 2. 各自特点
        
        **淘宝数据集：**
        - ✅ 准确性表现优异，字段范围异常率为0%
        - ✅ 字段含义明确率达到100%
        - ❗ 字段命名一致率较弱，仅为60%
        
        **沃尔玛数据集：**
        - ✅ 字段命名一致率达到100%
        - ❗ 存在2.2%的字段范围异常
        - ❗ 字段含义明确率相对较低，为68.75%
        """)