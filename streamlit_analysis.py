import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, silhouette_score)
from sklearn.cluster import KMeans


# 设置页面
# 在文件开头导入部分添加
from matplotlib import rcParams

# 在设置页面后添加中文字体配置
# 设置页面
st.set_page_config(page_title="淘宝用户行为分析平台", layout="wide")
st.title("淘宝用户行为分析交互式课件")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载与预处理
def load_and_preprocess():
    st.header("1. 数据集介绍")

    # 数据集描述
    with st.expander("数据集描述"):
        st.markdown("""
        ### 数据集组成
        1. **raw_sample.csv**: 原始样本数据
        2. **ad_feature.csv**: 广告特征数据
        3. **user_profile.csv**: 用户画像数据
        4. **behavior_log1.csv**: 用户行为日志数据
        """)

    # 加载原始数据集
    st.subheader("原始数据集预览")
    
    # raw_sample数据集
    # 修改数据文件路径
    raw_sample = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\raw_sample.csv').sample(frac=0.01, random_state=42)
    with st.expander("raw_sample数据集(抽样1%)"):
        st.write("""
        ### 字段说明
        - user_id: 脱敏过的用户ID
        - adgroup_id: 脱敏过的广告单元ID  
        - time_stamp: 时间戳
        - pid: 资源位
        - noclk: 为1代表没有点击；为0代表点击
        - clk: 为0代表没有点击；为1代表点击
        """)
        st.dataframe(raw_sample.head())
        st.write(f"数据维度: {raw_sample.shape}")

    # user_profile数据集
    user_profile = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\user_profile.csv').sample(frac=0.5, random_state=42).dropna(how='any')
    with st.expander("user_profile数据集（50%）"):
        st.write("""
        ### 字段说明
        - userid: 脱敏过的用户ID
        - cms_segid: 微群ID
        - cms_group_id: cms_group_id
        - final_gender_code: 性别 1:男,2:女
        - age_level: 年龄层次
        - pvalue_level: 消费档次 1:低档,2:中档,3:高档
        - shopping_level: 购物深度 1:浅层用户,2:中度用户,3:深度用户
        - occupation: 是否大学生 1:是,0:否
        - new_user_class_level: 城市层级
        """)
        st.dataframe(user_profile.head())
        st.write(f"数据维度: {user_profile.shape}")

    # ad_feature数据集
    ad_feature = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\ad_feature.csv').sample(frac=0.5, random_state=42).dropna(subset=['brand'])
    with st.expander("ad_feature数据集（50%）"):
        st.write("""
        ### 字段说明
        - adgroup_id: 脱敏过的广告ID
        - cate_id: 脱敏过的商品类目ID
        - campaign_id: 脱敏过的广告计划ID
        - customer_id: 脱敏过的广告主ID
        - brand: 脱敏过的品牌ID
        - price: 宝贝的价格
        """)
        st.dataframe(ad_feature.head())
        st.write(f"数据维度: {ad_feature.shape}")

    # behavior数据集
    behavior = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\behavior_log1.csv')
    with st.expander("behavior_log1数据集"):
        st.write("""
        ### 字段说明
        - user: 脱敏过的用户ID
        - time_stamp: 时间戳
        - btag: 行为类型(ipv:浏览, cart:加入购物车, fav:喜欢, buy:购买)
        - cate: 脱敏过的商品类目
        - brand: 脱敏过的品牌词
        """)
        st.dataframe(behavior.head())
        st.write(f"数据维度: {behavior.shape}")

    # 合并数据集
    with st.expander("数据合并详情"):
        st.write("""
        1. 数据集raw_sample, user_profile以关键词'userid'内连接,生成表merged_data1
        2..数据集merged_data1, behavior以关键词'userid'内连接,生成表merged_data2
        3. 数据集merged_data2, ad_feature以关键词'adgroup_id'内连接,生成表merged_data
        4. 最终合并数据集保存为merged_data.csv
      
        """)
 # 上传文件或使用默认路径
    data_path = r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\merged_data.csv'
    data = pd.read_csv(data_path)
       
# 显示合并后的数据
    st.dataframe(data.head())
    st.write(f"合并后数据集大小: {data.shape}")

    

# 2. 未处理数据质量的分析
# 在文件开头添加图表保存路径
import os
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# 修改可视化部分代码
def analysis_without_processing():
    st.header("2. 未处理数据质量的分析")
    
    # 直接从文件加载未处理数据
    data = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\merged_data.csv')
    st.session_state['raw_data'] = data
    
    # 显示数据
    st.dataframe(data.head())
    st.write(f"数据集大小: {data.shape}")

    if 'raw_data' not in st.session_state:
        st.warning("请先加载数据")
        return
    
    data = st.session_state['raw_data'].copy()
    
    # 特征选择
    default_features = [
        'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
        'btag', 'cate', 'brand', 'cate_id', 'campaign_id', 'customer', 'price'
    ]
    available_features = [col for col in default_features if col in data.columns]
    
    features = st.multiselect(
        "选择特征列(未处理数据)",
        options=data.columns,
        default=available_features
    )
    
    target = st.selectbox("选择目标变量(未处理数据)", options=data.columns, index=data.columns.get_loc('clk'))
    
    # 准备数据
    X = data[features].fillna(0)
    y = data[target].fillna(0)
    
    # 分类特征编码
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 模型训练
    model_type = st.selectbox("选择模型类型", ["逻辑回归", "决策树", "随机森林"])
    
    if model_type == "逻辑回归":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "决策树":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.write(f"最佳参数: {grid_search.best_params_}")

    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # 显示结果
    # 可视化部分修改
    st.subheader("评估结果(未处理数据)")
    
    # 预先生成并保存ROC曲线图
    roc_path = os.path.join(CHARTS_DIR, "roc_curve_raw.png")
    if not os.path.exists(roc_path):
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax1.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_proba):.2f})')
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_title('ROC曲线(未处理数据)')
            ax1.legend()
            fig1.savefig(roc_path)
            plt.close(fig1)
    
    # 预先生成并保存混淆矩阵图
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix_raw.png")
    if not os.path.exists(cm_path):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('混淆矩阵(未处理数据)')
        fig2.savefig(cm_path)
        plt.close(fig2)
    
    # 直接显示预生成的图表
    if os.path.exists(roc_path):
        st.image(roc_path)
    else:
        st.warning("ROC曲线图生成失败")
    
    if os.path.exists(cm_path):
        st.image(cm_path)
    else:
        st.warning("混淆矩阵图生成失败")

    # 聚类分析部分也做类似修改
    st.subheader("用户聚类分析(未处理数据)")
    
    # 定义用户特征列
    user_features = ['userid', 'final_gender_code', 'age_level', 'pvalue_level']
    user_features = [col for col in user_features if col in data.columns]
    
    if len(user_features) > 0:
        X_user = data[user_features].fillna(0)
        
        # 分类特征编码
        for col in X_user.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_user[col] = le.fit_transform(X_user[col].astype(str))
        
        # 标准化
        X_user_scaled = StandardScaler().fit_transform(X_user)
        
        # 选择聚类数量
        n_clusters = st.slider("聚类数量", 2, 10, 4, key='user_cluster')
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_user_scaled)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 聚类结果
        if len(user_features) >= 2:
            sns.scatterplot(x=X_user[user_features[1]], y=X_user[user_features[2]], 
                           hue=clusters, palette='viridis', ax=ax1)
            ax1.set_title(f'用户聚类结果 (k={n_clusters})')
        
        # 肘部法则
        inertias = []
        for k in range(1, 11):
            kmeans_elbow = KMeans(n_clusters=k, random_state=42)
            kmeans_elbow.fit(X_user_scaled)
            inertias.append(kmeans_elbow.inertia_)
            
        ax2.plot(range(1, 11), inertias, marker='o')
        ax2.set_title('肘部法则')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('Inertia')
        
        st.pyplot(fig)
        
        # 评估
        st.write(f"轮廓系数: {silhouette_score(X_user_scaled, clusters):.4f}")
    else:
        st.warning("缺少用户特征数据，无法进行聚类分析")


# 3. 处理数据质量后的分析
def analysis_with_processing():
    st.header("3. 处理数据质量后的分析")
    
    # 加载数据
    if 'raw_data' in st.session_state:
        data = st.session_state['raw_data'].copy()
    else:
        data = pd.read_csv(r'c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\data\merged_data.csv')
        st.session_state['raw_data'] = data
    
    # 显示原始数据
    with st.expander("原始数据预览"):
        st.dataframe(data.head())
        st.write(f"原始数据维度: {data.shape}")

    # 数据处理过程描述
    st.subheader("数据处理过程")
    
    # 日期处理
    with st.expander("1. 日期处理"):
        st.write("将时间戳转换为日期格式，并提取日期中的天数")
        data['time_stamp'] = pd.to_datetime(data['time_stamp'], unit='s')
        data['day'] = data['time_stamp'].dt.day
        st.dataframe(data[['time_stamp', 'day']].head())
    
    # 日期编码
    with st.expander("2. 日期编码"):
        st.write("将日期编码为连续数值：6-12日编码为0-6，13日编码为8")
        def encode_day(day):
            if 6 <= day <= 12: return day - 6
            elif day == 13: return 8
            else: return None
        data['encoded_day'] = data['day'].apply(encode_day)
        st.dataframe(data[['day', 'encoded_day']].head())
    
    # 价格分箱
    with st.expander("3. 价格分箱"):
        st.write("基于33%和66%分位数将价格分为low/medium/high三档")
        q1 = data['price'].quantile(0.33)
        q2 = data['price'].quantile(0.66)
        st.write(f"分箱边界值: q1(33%)={q1:.2f}, q2(66%)={q2:.2f}")
        
        def classify_price(price):
            if price < q1: return 'low'
            elif price < q2: return 'medium'
            else: return 'high'
        
        data['price_category'] = data['price'].apply(classify_price)
        st.dataframe(data[['price', 'price_category']].head())
    
    # 独热编码
    with st.expander("4. 独热编码"):
        st.write("将价格分类转换为独热编码格式")
        data = pd.get_dummies(data, columns=['price_category'], prefix='price')
        for col in ['price_low', 'price_medium', 'price_high']:
            if col not in data.columns:
                data[col] = 0
        st.dataframe(data[['price_low', 'price_medium', 'price_high']].head())
    
    # 显示处理后的数据
    st.subheader("处理后的数据")
    st.dataframe(data.head())
    st.write(f"处理后数据维度: {data.shape}")

    # 特征选择
    features = [
        'encoded_day', 'pid', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'occupation',
        'btag', 'price_low', 'price_medium', 'price_high'
    ]
    features = [col for col in features if col in data.columns]
    
    target = 'clk'
    
    # 准备数据
    X = data[features].fillna(0)
    y = data[target].fillna(0)
    
    # 分类特征编码
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 模型训练
    model_type = st.selectbox("选择模型类型", ["随机森林", "决策树"])
    
    if model_type == "决策树":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        
        # 保存交叉验证结果
        cv_results_path = os.path.join(CHARTS_DIR, "cv_results.csv")
        pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)
        st.write(f"最佳参数: {grid_search.best_params_}")
    
    model.fit(X_train, y_train)
    
    # 保存模型
    model_path = os.path.join(CHARTS_DIR, f"{model_type}_model.pkl")
    import joblib
    joblib.dump(model, model_path)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 保存评估结果
    eval_results = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    eval_path = os.path.join(CHARTS_DIR, "eval_results.json")
    import json
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f)
    
    # 评估结果部分修改
    st.subheader("评估结果(处理后数据)")
    
    # 预生成并保存处理后数据的ROC曲线
    roc_path = os.path.join(CHARTS_DIR, "roc_curve_processed.png")
    if not os.path.exists(roc_path):
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax1.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_proba):.2f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title('ROC曲线(处理后数据)')
        ax1.legend()
        fig1.savefig(roc_path)
        plt.close(fig1)
    
    # 预生成并保存处理后数据的混淆矩阵
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix_processed.png")
    if not os.path.exists(cm_path):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('混淆矩阵(处理后数据)')
        fig2.savefig(cm_path)
        plt.close(fig2)
    
    # 直接显示预生成的图表
    if os.path.exists(roc_path):
        st.image(roc_path)
    if os.path.exists(cm_path):
        st.image(cm_path)
    
    # 特征重要性图
    if model_type == "随机森林":
        fi_path = os.path.join(CHARTS_DIR, "feature_importance.png")
        if not os.path.exists(fi_path):
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            ax3.barh(range(len(features)), importances[indices], align='center')
            ax3.set_yticks(range(len(features)), [features[i] for i in indices])
            ax3.set_title('特征重要性(处理后数据)')
            fig3.savefig(fi_path)
            plt.close(fig3)
        st.image(fi_path)
    
    # 聚类分析部分
    st.subheader("聚类分析(处理后数据)")
    
    # 选择用于聚类的特征
    cluster_features = [col for col in features if col in data.columns]
    if len(cluster_features) >= 2:
        feature1 = st.selectbox("选择第一个聚类特征", options=cluster_features, index=0)
        feature2 = st.selectbox("选择第二个聚类特征", options=cluster_features, index=1)
        
        X_cluster = data[[feature1, feature2]].dropna()
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # 选择聚类数量
        n_clusters = st.slider("聚类数量", 2, 10, 4, key='processed_cluster')
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 聚类结果图
        sns.scatterplot(x=X_cluster[feature1], y=X_cluster[feature2], 
                       hue=clusters, palette='viridis', ax=ax1)
        ax1.set_title(f'K-means聚类结果 (k={n_clusters})')
        
        # 添加肘部法则图
        inertias = []
        for k in range(1, 11):
            kmeans_elbow = KMeans(n_clusters=k, random_state=42)
            kmeans_elbow.fit(X_scaled)
            inertias.append(kmeans_elbow.inertia_)
            
        ax2.plot(range(1, 11), inertias, marker='o')
        ax2.set_title('肘部法则')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('Inertia')
        
        st.pyplot(fig)
    
        # 评估
        silhouette = silhouette_score(X_scaled, clusters)
        st.write(f"轮廓系数: {silhouette:.4f}")

# 主程序
def main():
    # 添加侧边栏目录
    with st.sidebar:
        st.title("目录")
        selected = st.radio(
            "导航",
            ["数据加载", "未处理数据分析", "处理后数据分析"]
        )
    
    # 根据选择显示不同内容
    if selected == "数据加载":
        load_and_preprocess()
    elif selected == "未处理数据分析":
        load_and_preprocess()
        analysis_without_processing()
    else:
        # 显示所有内容
        load_and_preprocess()
        analysis_without_processing()
        analysis_with_processing()

    # 移除原来的tab布局
    # tab1, tab2 = st.tabs(["未处理数据质量", "处理数据质量后"])
    # with tab1:
    #     analysis_without_processing()
    # with tab2:
    #     analysis_with_processing()

if __name__ == "__main__":
    main()

# 修改图表保存路径
CHARTS_DIR = r"c:\Users\林旭馨\Desktop\上海应用技术大学\rl\DCMM\charts"