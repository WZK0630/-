import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy.stats import chi2
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. 数据读取与预处理
# ---------------------------
def load_data():
    """读取季度GDP数据和月度指标数据并预处理"""
    # 读取季度GDP数据
    gdp_q = pd.read_excel(
        r"D:\海洋经济大数据监测预警研究\基于多源高维数据的海洋经济景气实时监测研究\数据\海洋GDP季度.xlsx",
        usecols=["quarter", "mgdp"]
    )
    # 将季度格式转换为日期格式
    quarter_to_month = {"1": "03", "2": "06", "3": "09", "4": "12"}
    gdp_q["date"] = pd.to_datetime(gdp_q["quarter"].str[:4] + "-" + 
                                  gdp_q["quarter"].str[-1].map(quarter_to_month))
    gdp_q = gdp_q.set_index("date")["mgdp"].astype(float)
    
    # 读取月度数据
    df_m = pd.read_excel(
        r"D:\海洋经济大数据监测预警研究\基于多源高维数据的海洋经济景气实时监测研究\数据\月度数据整理.xlsx"
    )
    # 转换年月格式为日期
    df_m["年月"] = pd.to_datetime(df_m["年月"].astype(str), format="%Y%m")
    df_m = df_m.set_index("年月")
    
    return gdp_q, df_m

# ---------------------------
# 2. 计算同比增速与标准化
# ---------------------------
def preprocess_data(df_m, gdp_q):
    """计算指标同比增速并标准化"""
    # 处理月度数据
    # 排除累计值列
    non_cum_cols = [col for col in df_m.columns if "累计" not in col]
    df_growth = df_m[non_cum_cols].pct_change(periods=12) * 100
    
    # 处理无穷值和缺失值
    df_growth.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 标准化（跳过全NA列）
    valid_cols = df_growth.columns[df_growth.notna().any()]
    df_std = df_growth[valid_cols].copy()
    df_std = (df_std - df_std.mean()) / df_std.std()
    
    # 处理季度GDP数据
    gdp_growth = gdp_q.pct_change(periods=4) * 100  # 按年计算GDP同比增速
    gdp_growth = (gdp_growth - gdp_growth.mean()) / gdp_growth.std()  # 标准化
    
    return df_std.dropna(how='all', axis=1), gdp_growth

# ---------------------------
# 3. 混频格兰杰因果检验
# ---------------------------
def mixed_freq_granger_test(gdp_q, df_std, maxlag=1, n_boot=500, alpha=0.2):
    """实现三种混频格兰杰因果检验"""
    selected_indicators = {
        'traditional': [],
        'fdwb': [],
        'rdwb': [],
        'all_methods': []
    }
    
    # 将季度数据下采样到月度（向前填充）
    gdp_monthly = gdp_q.resample("MS").ffill().dropna()
    
    for col in df_std.columns:
        try:
            # 对齐时间索引并删除缺失值
            merged = pd.merge(df_std[[col]], gdp_monthly, 
                             left_index=True, right_index=True, 
                             how="inner").dropna()
            
            # 确保样本量足够
            if len(merged) < maxlag * 3:
                print(f"指标 {col} 样本量不足")
                continue
            
            # 检查数据类型并确保是数值型
            for column in merged.columns:
                if not np.issubdtype(merged[column].dtype, np.number):
                    merged[column] = pd.to_numeric(merged[column], errors='coerce')
                    
            # 检查是否有NaN值
            if merged.isna().any().any():
                print(f"指标 {col} 含有NaN值，跳过")
                continue
                
            # 1. 传统Wald检验
            try:
                model = VAR(merged)
                results = model.fit(maxlag)
                
                # 检查系数矩阵条件数
                stacked_coefs = np.hstack(results.coefs)
                if np.linalg.cond(stacked_coefs) > 1e10:
                    print(f"指标 {col} 系数矩阵条件数过大，跳过")
                    continue
                    
                # 检验指标是否Granger导致GDP
                wald_res = results.test_causality(causing=[col], caused="mgdp", kind="wald")
                wald_stat = wald_res.test_statistic
                trad_p_value = 1 - chi2.cdf(wald_stat, maxlag)
            except Exception as e:
                print(f"传统Wald检验出错 (指标 {col}): {str(e)}")
                continue
            
            # 2. FDWB-Wald检验
            try:
                fdwb_p_value = fdwb_wald_test(model, results, wald_stat, maxlag, n_boot)
            except Exception as e:
                print(f"FDWB-Wald检验出错 (指标 {col}): {str(e)}")
                fdwb_p_value = 1.0
            
            # 3. RDWB-Wald检验
            try:
                rdwb_p_value = rdwb_wald_test(merged, col, "mgdp", maxlag, n_boot)
            except Exception as e:
                print(f"RDWB-Wald检验出错 (指标 {col}): {str(e)}")
                rdwb_p_value = 1.0
            
            # 记录各种方法筛选结果
            if trad_p_value < alpha:
                selected_indicators['traditional'].append(col)
            
            if fdwb_p_value < alpha:
                selected_indicators['fdwb'].append(col)
                
            if rdwb_p_value < alpha:
                selected_indicators['rdwb'].append(col)
                
            # 综合三种方法（至少一种方法显著）
            methods_significant = sum([
                trad_p_value < alpha,
                fdwb_p_value < alpha,
                rdwb_p_value < alpha
            ])
            
            if methods_significant >= 1:  # 修改为只要一种方法显著即可
                selected_indicators['all_methods'].append(col)
                
            print(f"指标: {col}, 传统p值: {trad_p_value:.4f}, FDWB p值: {fdwb_p_value:.4f}, RDWB p值: {rdwb_p_value:.4f}")
            
        except Exception as e:
            print(f"处理指标 {col} 时出错: {str(e)}")
            continue
    
    return selected_indicators

def fdwb_wald_test(model, results, wald_stat, maxlag, n_boot=500):
    """频域野生自助法Wald检验"""
    residuals = results.resid.values
    T, k = residuals.shape
    
    boot_stats = []
    merged_values = model.endog
    
    # 限制bootstrap次数，提高效率
    actual_boot = min(n_boot, 500)
    
    for _ in range(actual_boot):
        try:
            # 生成野生误差项
            wild_errors = np.random.normal(size=residuals.shape)
            bootstrap_resid = residuals * wild_errors
            
            # 生成稳定模拟数据
            bootstrap_data = np.zeros_like(merged_values)
            bootstrap_data[:maxlag, :] = merged_values[:maxlag, :]
            
            for t in range(maxlag, T):
                pred = np.zeros(k)
                for j in range(maxlag):
                    pred += results.coefs[j] @ bootstrap_data[t - j - 1, :]
                bootstrap_data[t, :] = pred + bootstrap_resid[t]
            
            boot_model = VAR(bootstrap_data).fit(maxlag)
            boot_res = boot_model.test_causality(causing=0, caused=1, kind="wald")
            boot_stat = boot_res.test_statistic
            boot_stats.append(boot_stat)
        except Exception:
            continue
    
    if len(boot_stats) < actual_boot * 0.1:  # 如果成功率低于10%
        return 1.0  # 返回最大p值
    
    # 计算p值
    fdwb_p = (np.sum(np.array(boot_stats) >= wald_stat) + 1) / (len(boot_stats) + 1)
    return fdwb_p

def rdwb_wald_test(data, x_name, y_name, maxlag, n_boot=500):
    """残差野生自助法Wald检验"""
    # 限制bootstrap次数，提高效率
    actual_boot = min(n_boot, 500)
    
    try:
        # 拟合完整VAR模型
        model_full = VAR(data)
        results_full = model_full.fit(maxlag)
        
        # 拟合约束模型（x不Granger导致y）
        y_data = data[y_name].values.reshape(-1, 1)
        y_lags = []
        
        for i in range(1, maxlag + 1):
            y_lags.append(np.roll(y_data, i, axis=0)[:len(y_data)-maxlag])
        
        y_lags = np.hstack(y_lags)
        y_vals = y_data[maxlag:]
        
        # OLS估计约束模型
        X = np.column_stack([np.ones(len(y_lags)), y_lags])
        beta_r = np.linalg.pinv(X.T @ X) @ X.T @ y_vals
        
        # 计算约束模型残差
        e_r = y_vals - X @ beta_r
        
        # 计算原始Wald统计量
        wald_res = results_full.test_causality(causing=[x_name], caused=y_name, kind="wald")
        wald_stat = wald_res.test_statistic
    except Exception as e:
        print(f"RDWB测试初始化失败: {str(e)}")
        return 1.0
    
    # RDWB自助法
    boot_stats = []
    for _ in range(actual_boot):
        try:
            # 生成野生残差
            wild_factor = np.random.normal(size=len(e_r))
            e_star = e_r * wild_factor
            
            # 生成自助样本
            y_star = X @ beta_r + e_star
            
            # 构建自助数据集
            boot_data = data.copy()
            boot_data.iloc[maxlag:, boot_data.columns.get_loc(y_name)] = y_star.flatten()
            
            # 在自助样本上拟合VAR模型
            boot_model = VAR(boot_data)
            boot_results = boot_model.fit(maxlag)
            
            # 计算自助样本的Wald统计量
            boot_res = boot_results.test_causality(causing=[x_name], caused=y_name, kind="wald")
            boot_stat = boot_res.test_statistic
            boot_stats.append(boot_stat)
        except Exception:
            continue
    
    if len(boot_stats) < actual_boot * 0.1:  # 如果成功率低于10%
        return 1.0
    
    # 计算p值
    rdwb_p = (np.sum(np.array(boot_stats) >= wald_stat) + 1) / (len(boot_stats) + 1)
    return rdwb_p

# ---------------------------
# 4. 混频动态单因子模型
# ---------------------------
class MixedFrequencyDFM:
    def __init__(self, n_factors=1, max_iter=100, tol=1e-6):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, data):
        """拟合混频动态因子模型"""
        # 填充缺失值（前向填充和后向填充）
        data_filled = data.ffill().bfill()
        
        # 初始化因子和载荷（PCA初始化）
        pca = PCA(n_components=self.n_factors)
        factors_init = pca.fit_transform(data_filled)
        loadings_init = pca.components_
        
        # 初始化模型参数
        self.factors = factors_init
        self.loadings = loadings_init
        observation_cov = np.diag(np.var(data_filled - self.factors @ self.loadings, axis=0))
        
        # 设置状态空间模型参数
        transition_matrix = np.eye(self.n_factors)  # 单位矩阵作为转移矩阵
        transition_cov = np.eye(self.n_factors) * 0.01  # 初始状态协方差
        
        prev_loglik = -np.inf
        
        # EM算法迭代
        for i in range(self.max_iter):
            # E步骤：Kalman滤波和平滑
            kf = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=self.loadings,
                transition_covariance=transition_cov,
                observation_covariance=observation_cov,
                initial_state_mean=np.zeros(self.n_factors),
                initial_state_covariance=np.eye(self.n_factors),
                em_vars=['transition_covariance', 'observation_covariance']
            )
            
            # 运行卡尔曼平滑器
            smoothed_state_means, smoothed_state_covs = kf.smooth(data_filled)
            
            # M步骤：更新参数
            # 更新转移矩阵
            sum_tt1 = np.zeros((self.n_factors, self.n_factors))
            sum_t1t1 = np.zeros((self.n_factors, self.n_factors))
            
            for t in range(len(data_filled) - 1):
                sum_tt1 += (smoothed_state_means[t+1] @ smoothed_state_means[t].T)
                sum_t1t1 += (smoothed_state_means[t] @ smoothed_state_means[t].T + smoothed_state_covs[t])
            
            transition_matrix = sum_tt1 @ np.linalg.pinv(sum_t1t1)
            
            # 更新因子载荷
            sum_xy = data_filled.T @ smoothed_state_means
            sum_xx = np.zeros((self.n_factors, self.n_factors))
            
            for t in range(len(data_filled)):
                sum_xx += smoothed_state_means[t].reshape(-1, 1) @ smoothed_state_means[t].reshape(1, -1) + smoothed_state_covs[t]
            
            self.loadings = sum_xy @ np.linalg.pinv(sum_xx)
            
            # 更新因子
            self.factors = smoothed_state_means
            
            # 计算对数似然
            residuals = data_filled - smoothed_state_means @ self.loadings.T
            
            # 更新观测协方差矩阵
            observation_cov = np.diag(np.mean(residuals**2, axis=0))
            
            # 计算新的对数似然
            loglik = -0.5 * np.sum(residuals**2 / np.diag(observation_cov)) - 0.5 * len(data_filled) * np.log(np.prod(np.diag(observation_cov)))
            
            # 检查收敛
            if np.abs(loglik - prev_loglik) < self.tol:
                print(f"EM算法在第{i+1}次迭代后收敛")
                break
                
            prev_loglik = loglik
            
            if i == self.max_iter - 1:
                print(f"达到最大迭代次数{self.max_iter}，EM算法未收敛")
        
        return self.factors[:, 0]  # 返回第一个因子作为先行指数

# ---------------------------
# 5. 评估和可视化
# ---------------------------
def evaluate_leading_index(leading_index, gdp_growth, max_lead=12):
    """评估先行指数与GDP的相关性和先行性"""
    correlations = []
    
    # 将GDP转换为月度
    gdp_monthly = gdp_growth.resample('MS').ffill()
    
    # 计算不同先行期的相关系数
    for lead in range(max_lead + 1):
        if lead == 0:
            corr = leading_index.corr(gdp_monthly)
        else:
            corr = leading_index[:-lead].corr(gdp_monthly[lead:])
        correlations.append(corr)
    
    # 找到最大相关系数及其对应的先行期
    max_corr = max(correlations)
    optimal_lead = correlations.index(max_corr)
    
    return {
        'correlations': correlations,
        'max_correlation': max_corr,
        'optimal_lead': optimal_lead
    }

def plot_results(leading_index, gdp_growth, optimal_lead):
    """绘制先行指数与GDP的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 将GDP转换为月度
    gdp_monthly = gdp_growth.resample('MS').ffill()
    
    # 对齐时间序列
    common_idx = leading_index.index.intersection(gdp_monthly.index)
    aligned_index = leading_index[common_idx]
    aligned_gdp = gdp_monthly[common_idx]
    
    # 绘制对比图
    plt.plot(aligned_index, label='经济合成先行指数')
    plt.plot(aligned_gdp, label='GDP同比增长率')
    
    # 如果有先行期，绘制先行期后的先行指数
    if optimal_lead > 0:
        plt.plot(aligned_index.shift(-optimal_lead), label=f'先行指数（后移{optimal_lead}个月）', linestyle='--')
    
    plt.legend()
    plt.title('经济合成先行指数与GDP同比增长率对比')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('经济景气先行指数评估.png', dpi=300)
    plt.show()

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    try:
        print("1. 加载数据...")
        gdp_q, df_m = load_data()
        
        print("2. 数据预处理...")
        df_std, gdp_growth = preprocess_data(df_m, gdp_q)
        
        if df_std.empty:
            raise ValueError("预处理后数据为空，请检查输入数据")
        
        print("3. 混频格兰杰因果检验...")
        selected_indicators = mixed_freq_granger_test(gdp_q, df_std, maxlag=1, n_boot=500, alpha=0.2)
        
        print("\n各方法筛选结果：")
        print(f"传统Wald检验筛选指标数: {len(selected_indicators['traditional'])}")
        print(f"FDWB-Wald检验筛选指标数: {len(selected_indicators['fdwb'])}")
        print(f"RDWB-Wald检验筛选指标数: {len(selected_indicators['rdwb'])}")
        print(f"综合三种方法筛选指标数: {len(selected_indicators['all_methods'])}")
        
        # 使用综合方法筛选的指标
        final_indicators = selected_indicators['all_methods']
        
        if not final_indicators:
            # 如果综合方法没有选出指标，尝试使用单一方法选出的指标
            if selected_indicators['traditional']:
                final_indicators = selected_indicators['traditional']
                print("使用传统Wald检验筛选的指标")
            elif selected_indicators['fdwb']:
                final_indicators = selected_indicators['fdwb']
                print("使用FDWB-Wald检验筛选的指标")
            elif selected_indicators['rdwb']:
                final_indicators = selected_indicators['rdwb']
                print("使用RDWB-Wald检验筛选的指标")
            else:
                # 如果仍然没有指标，随机选择一些指标
                print("所有检验方法均未筛选出指标，随机选择前10个指标作为备选")
                final_indicators = list(df_std.columns[:min(10, len(df_std.columns))])
            
        print(f"\n最终筛选的指标（{len(final_indicators)}个）:")
        for idx in final_indicators:
            print(f"- {idx}")
        
        print("\n4. 构建混频动态单因子模型...")
        model = MixedFrequencyDFM(n_factors=1)
        leading_index = model.fit(df_std[final_indicators])
        
        # 转换为Series并保存
        leading_series = pd.Series(leading_index, index=df_std.index)
        leading_series.to_csv("经济合成先行指数.csv")
        
        print("\n5. 评估先行指数...")
        eval_results = evaluate_leading_index(leading_series, gdp_growth)
        print(f"最大相关系数: {eval_results['max_correlation']:.4f}")
        print(f"最佳先行期: {eval_results['optimal_lead']}个月")
        
        print("\n6. 绘制结果...")
        plot_results(leading_series, gdp_growth, eval_results['optimal_lead'])
        
        print("\n程序运行完成！先行指数已保存至'经济合成先行指数.csv'")
        
    except Exception as e:
        print(f"程序运行失败: {str(e)}")# -
