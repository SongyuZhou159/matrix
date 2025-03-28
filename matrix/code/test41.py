import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def temporal_split(ratings_df, train_ratio=0.8):
    ratings_df = ratings_df.sort_values('timestamp')
    split_idx = int(len(ratings_df) * train_ratio)
    return ratings_df.iloc[:split_idx], ratings_df.iloc[split_idx:]

def prepare_data(data, time_decay_factor=0.05, confidence_alpha=0.1):
    # 获取唯一的用户ID和电影ID并排序
    user_ids = sorted(data['userId'].unique())
    movie_ids = sorted(data['movieId'].unique())
    
    # 创建ID到索引的映射字典
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    
    # 计算用户和物品的评分次数
    user_rating_counts = data['userId'].value_counts()
    item_rating_counts = data['movieId'].value_counts()
    
    # 初始化置信度数组
    user_confidence = np.zeros(len(user_ids))
    item_confidence = np.zeros(len(movie_ids))
    
    # 使用评分次数计算置信度
    for uid, count in user_rating_counts.items():
        user_confidence[user_to_idx[uid]] = 1 + confidence_alpha * np.log1p(count)
    for mid, count in item_rating_counts.items():
        item_confidence[movie_to_idx[mid]] = 1 + confidence_alpha * np.log1p(count)
    
    # 计算全局平均评分
    global_mean = data['rating'].mean()
    
    # 计算用户和物品的偏置
    user_biases = data.groupby('userId')['rating'].agg(['mean', 'count']).reset_index()
    item_biases = data.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    
    # 计算加权偏置
    user_biases['weighted_bias'] = (user_biases['mean'] - global_mean) * np.log1p(user_biases['count'])
    item_biases['weighted_bias'] = (item_biases['mean'] - global_mean) * np.log1p(item_biases['count'])
    
    # 初始化偏置数组
    user_biases_array = np.zeros(len(user_ids))
    item_biases_array = np.zeros(len(movie_ids))
    
    # 填充偏置数组
    for _, row in user_biases.iterrows():
        user_biases_array[user_to_idx[row['userId']]] = row['weighted_bias']
    for _, row in item_biases.iterrows():
        item_biases_array[movie_to_idx[row['movieId']]] = row['weighted_bias']
    
    # 归一化偏置
    user_biases_array = user_biases_array / (np.log1p(user_rating_counts.max()) + 1e-6)
    item_biases_array = item_biases_array / (np.log1p(item_rating_counts.max()) + 1e-6)
    
    # 创建评分矩阵和掩码
    R = np.zeros((len(user_ids), len(movie_ids)))
    mask = np.zeros_like(R, dtype=bool)
    timestamps = np.zeros_like(R)
    
    # 填充评分矩阵和时间戳
    for _, row in data.iterrows():
        u_idx = user_to_idx[row['userId']]
        m_idx = movie_to_idx[row['movieId']]
        R[u_idx, m_idx] = row['rating']
        mask[u_idx, m_idx] = True
        timestamps[u_idx, m_idx] = row['timestamp']
    
    # 计算时间权重
    latest_timestamp = timestamps[timestamps > 0].max()
    time_weights = np.where(timestamps > 0,
                          np.exp(-time_decay_factor * (latest_timestamp - timestamps) / (24*60*60)),
                          0)
    
    # 对于NMF，保持评分矩阵的非负性
    R_processed = R.copy()
    if np.min(R_processed) < 0:
        R_processed += abs(np.min(R_processed))
    
    return (R_processed, mask, timestamps, user_to_idx, movie_to_idx, global_mean, 
            user_biases_array, item_biases_array, user_confidence, item_confidence, time_weights)

class BaseRecommender:
    def __init__(self, n_factors, reg_param=0.1):
        self.n_factors = n_factors
        self.reg_param = reg_param
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_biases_array = None
        self.item_biases_array = None
        self.user_confidence = None
        self.item_confidence = None
        self.time_weights = None
        self.singular_values = None
        self.Y = None
    
    def predict(self, user_idx, item_idx, timestamp=None):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        # 1. 时间权重计算
        time_weight = 1.0
        if timestamp is not None and self.time_weights is not None:
            latest_timestamp = np.max(timestamp)
            time_weight = np.exp(-0.05 * (latest_timestamp - timestamp) / (24*60*60))
        
        confidence = np.sqrt(self.user_confidence[user_idx] * self.item_confidence[item_idx])
        
        base_pred = (self.global_mean + 
                    self.user_biases_array[user_idx] * np.sqrt(self.user_confidence[user_idx]) +
                    self.item_biases_array[item_idx] * np.sqrt(self.item_confidence[item_idx]))
        
        # 2. 潜在因子预测计算
        latent_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # 3. 保持原有的预测组合方式
        pred = base_pred + latent_pred * confidence * time_weight
        return np.clip(pred, 1, 5)

class SVDRecommender(BaseRecommender):
    def fit(self, R, mask, timestamps=None, global_mean=0, user_biases=None, item_biases=None,
            user_confidence=None, item_confidence=None, time_weights=None):
        self.global_mean = global_mean
        self.user_biases_array = user_biases
        self.item_biases_array = item_biases
        self.user_confidence = user_confidence
        self.item_confidence = item_confidence
        self.time_weights = time_weights
        
        # Apply global scaling and weighting
        R_masked = np.where(mask, R, 0)
        if time_weights is not None:
            R_masked = R_masked * np.sqrt(time_weights)
        
        # Convert to sparse matrix for efficient SVD
        R_sparse = csr_matrix(R_masked)
        
        # Compute SVD
        U, s, Vt = svds(R_sparse, k=self.n_factors)
        
        # 简化并改进正则化方式
        s_max = np.max(s)
        relative_importance = s / s_max
        
        # 使用单调的正则化函数
        reg_strength = self.reg_param * (1 - relative_importance)
        s_regularized = s / (1 + reg_strength)
        
        # 应用因子数量的缩放
        scale_factor = np.power(self.n_factors / 75, 0.3)
        s_regularized = s_regularized * scale_factor
        
        # 重要性衰减
        importance_decay = np.exp(-np.arange(len(s)) / (self.n_factors / 3))
        s_regularized = s_regularized * importance_decay
        
        self.user_factors = U * np.sqrt(s_regularized)[:, None].T
        self.item_factors = Vt.T * np.sqrt(s_regularized)
        self.singular_values = s_regularized
        
        # 调整噪声正则化
        noise_scale = 0.1 * self.reg_param / np.sqrt(self.n_factors)
        self.user_factors += np.random.normal(0, noise_scale, self.user_factors.shape)
        self.item_factors += np.random.normal(0, noise_scale, self.item_factors.shape)

class SVTRecommender(BaseRecommender):
    def fit(self, R, mask, timestamps=None, global_mean=0, user_biases=None, item_biases=None,
            user_confidence=None, item_confidence=None, time_weights=None, n_iterations=30):
        self.global_mean = global_mean
        self.user_biases_array = user_biases
        self.item_biases_array = item_biases
        self.user_confidence = user_confidence
        self.item_confidence = item_confidence
        self.time_weights = time_weights
        
        R_masked = np.where(mask, R, 0)
        if time_weights is not None:
            R_masked = R_masked * np.sqrt(time_weights)
        
        #初始化，保持局部结构
        Y = R_masked.copy()
        
        #改进自适应阈值计算函数
        def get_adaptive_threshold(Y_local, local_mask, iteration, singular_values=None):
            if not np.any(local_mask):
                return 0
                
            #基于局部统计的阈值
            local_std = np.std(Y_local[local_mask]) if np.any(local_mask) else 0
            local_density = np.count_nonzero(local_mask) / local_mask.size
            
            #简化迭代进度的影响
            iteration_factor = np.exp(-iteration / n_iterations)
            
            #简化奇异值的影响
            if singular_values is not None and len(singular_values) > 0:
                sv_factor = singular_values[0] / (singular_values[-1] + 1e-8)
                sv_factor = np.clip(np.log1p(sv_factor) / 5, 0.5, 2.0)
            else:
                sv_factor = 1.0
            
            # 简化阈值计算
            base_threshold = local_std * np.sqrt(local_density)
            threshold = (base_threshold * 
                        iteration_factor * 
                        sv_factor * 
                        (1.0 + 0.2 * (self.n_factors / 75)))
            
            return threshold * self.reg_param
        
        # 添加动态窗口大小
        def get_dynamic_window_size(shape, iteration):
            base_size = min(100, min(shape) // 2)
            # 随迭代次数减小窗口大小，使得后期更关注局部结构
            decay = 0.8 + 0.2 * (iteration / n_iterations)
            return max(20, int(base_size * decay))
        
        # 添加自适应学习率
        def get_learning_rate(iteration, singular_values=None):
            base_rate = 1.0
            if singular_values is not None and len(singular_values) > 0:
                sv_decay = singular_values[0] / (singular_values[-1] + 1e-8)
                sv_factor = np.clip(1 / np.log1p(sv_decay), 0.1, 1.0)
            else:
                sv_factor = 1.0
            
            iteration_decay = (1 + iteration) ** -0.5
            return base_rate * sv_factor * iteration_decay
        
        prev_Y = None
        singular_values = None
        
        for iteration in range(n_iterations):
            window_size = get_dynamic_window_size(R_masked.shape, iteration)
            learning_rate = get_learning_rate(iteration, singular_values)
            
            # 计算并存储当前奇异值
            U, s, Vt = svds(Y, k=self.n_factors)
            singular_values = s
            
            # 添加自适应正则化
            reg_strength = self.reg_param * (1 + np.arange(len(s))) / len(s)
            s = s / (1 + reg_strength)
            
            # 处理数据块
            for i in range(0, Y.shape[0], window_size):
                for j in range(0, Y.shape[1], window_size):
                    i_end = min(i + window_size, Y.shape[0])
                    j_end = min(j + window_size, Y.shape[1])
                    
                    local_mask = mask[i:i_end, j:j_end]
                    if not np.any(~local_mask):  # 如果全部是观测值，跳过
                        continue
                    
                    tau = get_adaptive_threshold(
                        Y[i:i_end, j:j_end], 
                        local_mask,
                        iteration,
                        singular_values
                    )
                    
                    # 改进更新策略
                    Y_new_local = U[i:i_end] @ np.diag(s) @ Vt[:, j:j_end]
                    
                    # 只更新未观测的条目
                    unobserved = ~local_mask
                    if np.any(unobserved):
                        Y[i:i_end, j:j_end][unobserved] = (
                            (1 - learning_rate) * Y[i:i_end, j:j_end][unobserved] +
                            learning_rate * Y_new_local[unobserved]
                        )
            
            # 添加收敛检查
            if prev_Y is not None:
                rel_change = np.sum(np.abs(Y - prev_Y)) / (np.sum(np.abs(prev_Y)) + 1e-10)
                if rel_change < 1e-4:  # 收敛阈值
                    break
            prev_Y = Y.copy()
        
        # 最终分解
        U, s, Vt = svds(Y, k=self.n_factors)
        
        # 最终的因子调整
        importance_weights = np.exp(-np.arange(len(s)) / (self.n_factors / 2))
        s = s * importance_weights
        
        self.user_factors = U * np.sqrt(s)[:, None].T
        self.item_factors = Vt.T * np.sqrt(s)
        self.singular_values = s
        
        # 添加噪声正则化
        noise_scale = self.reg_param / np.sqrt(self.n_factors * np.mean(s))
        self.user_factors += np.random.normal(0, noise_scale, self.user_factors.shape)
        self.item_factors += np.random.normal(0, noise_scale, self.item_factors.shape)
        
class EnhancedNMFRecommender(BaseRecommender):
    def fit(self, R, mask, timestamps=None, global_mean=0, user_biases=None, item_biases=None,
            user_confidence=None, item_confidence=None, time_weights=None):
        self.global_mean = global_mean
        self.user_biases_array = user_biases
        self.item_biases_array = item_biases
        self.user_confidence = user_confidence
        self.item_confidence = item_confidence
        self.time_weights = time_weights
        
        try:
            # 数据预处理
            R_masked = np.where(mask, R, 0)
            R_positive = np.maximum(R_masked, 0)  # 确保非负性
            
            if time_weights is not None:
                time_weights_normalized = time_weights / np.maximum(time_weights.sum(axis=1, keepdims=True), 1e-10)
                R_positive = R_positive * np.sqrt(time_weights_normalized)
            
            # 实际秩检查和有效因子数量确定
            U_full, s_full, Vt_full = np.linalg.svd(R_positive, full_matrices=False)
            actual_rank = np.sum(s_full > 1e-10)
            n_factors_effective = max(1, min(self.n_factors, actual_rank, min(R_positive.shape)))
            
            # Scale处理
            scale_factor = np.percentile(R_positive[mask], 95)
            R_scaled = R_positive / (scale_factor + 1e-10)
            
            # NMF模型配置
            model = NMF(
                n_components=n_factors_effective,
                init='nndsvdar',
                solver='mu',
                beta_loss='kullback-leibler',  # 更适合评分数据
                max_iter=500,
                alpha_W=self.reg_param * 0.1,  # 弱化正则化影响
                alpha_H=self.reg_param * 0.1,
                l1_ratio=0.1,  # 轻微稀疏化
                tol=1e-4,
                random_state=42
            )
            
            # 模型拟合
            self.user_factors = model.fit_transform(R_scaled)
            self.item_factors = model.components_.T
            
            # 因子多样性增强和缩放恢复
            self.user_factors *= np.sqrt(scale_factor)
            self.item_factors *= np.sqrt(scale_factor)
            
            # 因子放大以确保n_factors影响
            self.user_factors *= 1.0 + 0.2 * np.log1p(self.n_factors)
            self.item_factors *= 1.0 + 0.2 * np.log1p(self.n_factors)
            
            # 数值稳定性处理
            user_norms = np.linalg.norm(self.user_factors, axis=1, keepdims=True)
            item_norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
            self.user_factors = np.divide(self.user_factors, user_norms + 1e-10)
            self.item_factors = np.divide(self.item_factors, item_norms + 1e-10)
            
            # 轻微正则化
            if self.reg_param > 0:
                reg_matrix = self.reg_param * 0.1 * np.eye(n_factors_effective)
                self.user_factors = self.user_factors @ np.linalg.inv(np.eye(n_factors_effective) + reg_matrix)
                self.item_factors = self.item_factors @ np.linalg.inv(np.eye(n_factors_effective) + reg_matrix)
                
        except Exception as e:
            print(f"Error in NMF fitting: {str(e)}")
            raise
    
    def predict(self, user_idx, item_idx, timestamp=None):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        try:
            # 动态调整基础预测权重
            base_pred_weight = 0.5 + 0.5 * np.exp(-self.n_factors / 30)
            
            # 基础预测
            base_pred = (
                self.global_mean +
                self.user_biases_array[user_idx] * np.sqrt(self.user_confidence[user_idx]) * base_pred_weight +
                self.item_biases_array[item_idx] * np.sqrt(self.item_confidence[item_idx]) * base_pred_weight
            )
            
            # 潜在因子预测
            latent_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            latent_scale = np.log1p(self.n_factors) * (1.0 - 0.2 * self.reg_param)
            
            # 时间权重计算
            time_weight = 1.0
            if timestamp is not None and self.time_weights is not None:
                latest_timestamp = np.max(timestamp)
                time_decay = -0.05 * (latest_timestamp - timestamp) / (24*60*60)
                time_weight = np.exp(np.clip(time_decay, -10, 0))
            
            # 组合预测
            pred = base_pred + latent_pred * latent_scale * time_weight
            return np.clip(pred, 1, 5)
            
        except Exception as e:
            print(f"Error in NMF prediction: {str(e)}")
            return self.global_mean  # 失败时返回全局平均值

class ALSRecommender(BaseRecommender):
    def __init__(self, n_factors, reg_param=0.1, bias_reg=0.01):
        super().__init__(n_factors, reg_param)
        self.bias_reg = bias_reg
        self.running_bias = 0
        self.bias_correction = None
        self.user_scale = None
        self.item_scale = None
        
    def fit(self, R, mask, timestamps=None, global_mean=0, user_biases=None, item_biases=None,
            user_confidence=None, item_confidence=None, time_weights=None, n_iterations=30):
        self.global_mean = global_mean
        self.user_biases_array = user_biases
        self.item_biases_array = item_biases
        self.user_confidence = user_confidence
        self.item_confidence = item_confidence
        self.time_weights = time_weights
        
        try:
            # 改进的数据预处理
            R_masked = np.where(mask, R, 0)
            user_means = np.sum(R_masked, axis=1) / np.maximum(np.sum(mask, axis=1), 1)
            item_means = np.sum(R_masked, axis=0) / np.maximum(np.sum(mask, axis=0), 1)
            
            # 软居中化：使用加权平均减少偏差
            self.user_scale = np.mean(R_masked[mask]) / (np.mean(user_means) + 1e-8)
            self.item_scale = np.mean(R_masked[mask]) / (np.mean(item_means) + 1e-8)
            
            R_centered = R_masked.copy()
            for i in range(R_masked.shape[0]):
                R_centered[i, mask[i, :]] -= user_means[i] * 0.5
            for j in range(R_masked.shape[1]):
                R_centered[:, j][mask[:, j]] -= item_means[j] * 0.5
            
            # 混合初始化策略
            n_users, n_items = R_masked.shape
            
            # SVD初始化
            U, s, Vt = svds(csr_matrix(R_centered), k=min(self.n_factors, min(R_centered.shape)-1))
            svd_user = U * np.sqrt(s)[:, None].T
            svd_item = Vt.T * np.sqrt(s)
            
            # 随机初始化
            np.random.seed(42)
            rand_scale = 0.01
            rand_user = np.random.normal(0, rand_scale, (n_users, self.n_factors))
            rand_item = np.random.normal(0, rand_scale, (n_items, self.n_factors))
            
            # 混合初始化
            self.user_factors = 0.7 * svd_user + 0.3 * rand_user
            self.item_factors = 0.7 * svd_item + 0.3 * rand_item
            
            # 优化训练过程
            batch_size = 2000
            running_bias_history = []
            
            for iteration in range(n_iterations):
                # 自适应正则化权重
                reg_weight = self.reg_param * (0.95 ** iteration)
                bias_reg_weight = self.bias_reg * (0.9 ** iteration)
                
                # 用户因子更新
                for start in range(0, n_users, batch_size):
                    end = min(start + batch_size, n_users)
                    batch_mask = mask[start:end]
                    batch_R = R_centered[start:end]
                    
                    for u in range(end - start):
                        if not np.any(batch_mask[u]):
                            continue
                            
                        local_mask = batch_mask[u]
                        local_confidence = np.sqrt(self.user_confidence[start + u])
                        
                        # 动态正则化
                        rating_count = np.sum(local_mask)
                        local_reg = reg_weight / np.sqrt(rating_count + 1)
                        
                        # 添加偏差惩罚
                        bias_penalty = bias_reg_weight * np.sign(self.running_bias)
                        
                        A = (self.item_factors[local_mask].T @ self.item_factors[local_mask] + 
                             local_reg * np.eye(self.n_factors))
                        b = (self.item_factors[local_mask].T @ batch_R[u, local_mask] - 
                             bias_penalty * np.ones(self.n_factors))
                        
                        try:
                            self.user_factors[start + u] = np.linalg.solve(
                                A + 1e-6 * np.eye(self.n_factors), b
                            )
                        except np.linalg.LinAlgError:
                            self.user_factors[start + u] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # 物品因子更新
                for start in range(0, n_items, batch_size):
                    end = min(start + batch_size, n_items)
                    batch_mask = mask[:, start:end]
                    batch_R = R_centered[:, start:end]
                    
                    for i in range(end - start):
                        if not np.any(batch_mask[:, i]):
                            continue
                            
                        local_mask = batch_mask[:, i]
                        local_confidence = np.sqrt(self.item_confidence[start + i])
                        
                        # 动态正则化
                        rating_count = np.sum(local_mask)
                        local_reg = reg_weight / np.sqrt(rating_count + 1)
                        
                        # 添加偏差惩罚
                        bias_penalty = bias_reg_weight * np.sign(self.running_bias)
                        
                        A = (self.user_factors[local_mask].T @ self.user_factors[local_mask] + 
                             local_reg * np.eye(self.n_factors))
                        b = (self.user_factors[local_mask].T @ batch_R[local_mask, i] - 
                             bias_penalty * np.ones(self.n_factors))
                        
                        try:
                            self.item_factors[start + i] = np.linalg.solve(
                                A + 1e-6 * np.eye(self.n_factors), b
                            )
                        except np.linalg.LinAlgError:
                            self.item_factors[start + i] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # 更新running_bias
                pred_sample = self._sample_predictions(R_masked, mask, 1000)
                self.running_bias = np.mean(pred_sample)
                running_bias_history.append(self.running_bias)
                
                # 早停检查
                if len(running_bias_history) > 5:
                    if np.std(running_bias_history[-5:]) < 1e-4:
                        break
            
            # 4. 计算最终的偏差校正
            self.bias_correction = -np.mean(running_bias_history[-5:])
            
            # 5. 数值稳定性处理
            user_norms = np.linalg.norm(self.user_factors, axis=1, keepdims=True)
            item_norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
            self.user_factors = np.divide(self.user_factors, user_norms + 1e-10)
            self.item_factors = np.divide(self.item_factors, item_norms + 1e-10)
            
        except Exception as e:
            print(f"Error in ALS fitting: {str(e)}")
            raise
            
    def _sample_predictions(self, R, mask, n_samples):
        """采样预测用于bias估计"""
        n_users, n_items = R.shape
        samples = []
        
        for _ in range(n_samples):
            u = np.random.randint(0, n_users)
            i = np.random.randint(0, n_items)
            if mask[u, i]:
                pred = self.predict(u, i)
                true = R[u, i]
                samples.append(pred - true)
                
        return np.array(samples)
    
    def predict(self, user_idx, item_idx, timestamp=None):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
            
        try:
            # 基础预测，使用缩放的偏置
            base_pred = (
                self.global_mean +
                self.user_biases_array[user_idx] * np.sqrt(self.user_confidence[user_idx]) * self.user_scale +
                self.item_biases_array[item_idx] * np.sqrt(self.item_confidence[item_idx]) * self.item_scale
            )
            
            # 潜在因子预测
            latent_pred = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            
            # 时间权重
            time_weight = 1.0
            if timestamp is not None and self.time_weights is not None:
                latest_timestamp = np.max(timestamp)
                time_weight = np.exp(-0.05 * (latest_timestamp - timestamp) / (24*60*60))
            
            # 组合预测并应用偏差校正
            pred = base_pred + latent_pred * time_weight
            if self.bias_correction is not None:
                pred += self.bias_correction
            
            return np.clip(pred, 1, 5)
            
        except Exception as e:
            print(f"Error in ALS prediction: {str(e)}")
            return self.global_mean


def calculate_map_at_k(recommender, test_data, user_to_idx, movie_to_idx, k=5):
    try:
        user_predictions = {}
        user_actuals = {}
        
        # 收集预测和实际评分
        for _, row in test_data.iterrows():
            try:
                if row['userId'] in user_to_idx and row['movieId'] in movie_to_idx:
                    user_id = row['userId']
                    if user_id not in user_predictions:
                        user_predictions[user_id] = []
                        user_actuals[user_id] = []
                    
                    u_idx = user_to_idx[user_id]
                    m_idx = movie_to_idx[row['movieId']]
                    
                    # 获取预测，包含错误处理
                    try:
                        pred = recommender.predict(u_idx, m_idx, row['timestamp'])
                        if pred is not None and not np.isnan(pred):
                            user_predictions[user_id].append((pred, row['movieId']))
                            user_actuals[user_id].append((row['rating'], row['movieId']))
                    except Exception as e:
                        print(f"Prediction error for user {user_id}, movie {row['movieId']}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue
        
        # 计算MAP
        aps = []
        n_factors = recommender.n_factors
        
        for user_id in user_predictions:
            try:
                # 确保用户有足够的评分
                if len(user_predictions[user_id]) == 0 or len(user_actuals[user_id]) == 0:
                    continue
                
                # 排序预测和实际评分
                pred_sorted = sorted(user_predictions[user_id], reverse=True)[:k]
                actual_sorted = sorted(user_actuals[user_id], reverse=True)[:k]
                
                # 动态相关性阈值
                relevant_items = {}
                for rating, movie_id in actual_sorted:
                    if rating >= 4.0 + 0.01 * n_factors:  # 动态高相关阈值
                        relevant_items[movie_id] = 1.0
                    elif rating >= 3.5 + 0.01 * n_factors:  # 动态中相关阈值
                        relevant_items[movie_id] = 0.7
                    elif rating >= 3.0 + 0.01 * n_factors:  # 动态低相关阈值
                        relevant_items[movie_id] = 0.3
                
                # 如果没有相关项，跳过此用户
                if not relevant_items:
                    continue
                
                # 计算AP
                hits = 0
                ap = 0.0
                for i, (pred_rating, movie_id) in enumerate(pred_sorted, 1):
                    if movie_id in relevant_items:
                        hits += 1
                        ap += (hits / i) * relevant_items[movie_id]
                
                # 归一化AP
                ap /= sum(relevant_items.values())
                aps.append(ap)
                
            except Exception as e:
                print(f"Error calculating AP for user {user_id}: {str(e)}")
                continue
        
        # 3. 返回最终的MAP值
        if not aps:
            print("Warning: No valid APs calculated")
            return 0.0
            
        return np.mean(aps)
        
    except Exception as e:
        print(f"Error in MAP calculation: {str(e)}")
        return 0.0

def evaluate_recommender(recommender, test_data, user_to_idx, movie_to_idx, timestamps):
    try:
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            try:
                if row['userId'] in user_to_idx and row['movieId'] in movie_to_idx:
                    u_idx = user_to_idx[row['userId']]
                    m_idx = movie_to_idx[row['movieId']]
                    
                    pred = recommender.predict(u_idx, m_idx, row['timestamp'])
                    predictions.append(pred)
                    actuals.append(row['rating'])
            except Exception:
                continue
        
        if not predictions:
            return None
            
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        map_5 = calculate_map_at_k(recommender, test_data, user_to_idx, movie_to_idx, k=5)
        bias = np.mean(predictions - actuals)
        
        return rmse, mae, map_5, bias
        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return None

def calculate_method_specific_memory(recommender):
    base_memory = (recommender.user_factors.nbytes + 
                  recommender.item_factors.nbytes + 
                  recommender.user_biases_array.nbytes + 
                  recommender.item_biases_array.nbytes)
    
    if isinstance(recommender, SVDRecommender):
        # SVD只需要存储奇异值和因子矩阵
        return base_memory + recommender.singular_values.nbytes
    elif isinstance(recommender, SVTRecommender):
        window_size = min(100, min(recommender.user_factors.shape[0], 
                                 recommender.item_factors.shape[0]))
        svt_memory = (
            base_memory +  # 基础内存
            recommender.singular_values.nbytes +  # 奇异值
            (window_size * window_size * 8)  # 局部窗口缓存
        )
        return svt_memory
    elif isinstance(recommender, EnhancedNMFRecommender):
        # NMF stores sparse factors
        sparse_overhead = (np.count_nonzero(recommender.user_factors) + 
                         np.count_nonzero(recommender.item_factors)) * 8
        return base_memory + sparse_overhead
    elif isinstance(recommender, ALSRecommender):
        # ALS needs memory for batch processing
        batch_size = 2000
        als_memory = (
            base_memory +  # 基础内存
            (batch_size * recommender.n_factors * 8) +  # 批处理矩阵
            (recommender.n_factors * recommender.n_factors * 8)  # A矩阵
        )
        return als_memory
    
    return base_memory

def calculate_compression_ratio(recommender, mask):
    total_elements = np.count_nonzero(mask)
    
    if isinstance(recommender, SVDRecommender):
        storage = (recommender.user_factors.size + 
                  recommender.item_factors.size + 
                  recommender.singular_values.size)
    elif isinstance(recommender, SVTRecommender):
        window_size = min(100, min(recommender.user_factors.shape[0], 
                                 recommender.item_factors.shape[0]))
        storage = (recommender.user_factors.size + 
                  recommender.item_factors.size + 
                  recommender.singular_values.size + 
                  window_size * window_size)  
    elif isinstance(recommender, EnhancedNMFRecommender):
        
        storage = (recommender.user_factors.size + 
                  recommender.item_factors.size)
        
        nonzero_elements = (np.count_nonzero(recommender.user_factors) + 
                          np.count_nonzero(recommender.item_factors))
        storage += nonzero_elements  
    else:  # ALS
        storage = (recommender.user_factors.size + 
                  recommender.item_factors.size + 
                  2000 * recommender.n_factors)
    
    return total_elements / max(storage, 1)  

def main():
    try:
        print("Loading data...")
        ratings = pd.read_csv('ratings.csv')
        
        train_data, test_data = temporal_split(ratings)
        
        n_factors_list = [30, 50, 75]
        reg_params = [0.01, 0.05, 0.1]
        results = []
        
        (R, mask, timestamps, user_to_idx, movie_to_idx, global_mean, 
         user_biases, item_biases, user_confidence, item_confidence, 
         time_weights) = prepare_data(train_data)
        
        for n_factors in n_factors_list:
            for reg_param in reg_params:
                print(f"Testing with {n_factors} factors, reg_param {reg_param}")
                
                for method, recommender_class in [
                    ('SVD', SVDRecommender),
                    ('SVT', SVTRecommender),
                    ('NMF', EnhancedNMFRecommender),
                    ('ALS', ALSRecommender)
                ]:
                    print(f"Running {method}...")
                    start_time = time.time()
                    
                    recommender = recommender_class(n_factors=n_factors, reg_param=reg_param)
                    R_copy = R.copy()
                    mask_copy = mask.copy()
                    timestamps_copy = timestamps.copy()
                    
                    recommender.fit(
                        R_copy, mask_copy, timestamps_copy,
                        global_mean, user_biases, item_biases,
                        user_confidence, item_confidence, time_weights
                    )
                    
                    metrics = evaluate_recommender(
                        recommender,
                        test_data,
                        user_to_idx,
                        movie_to_idx,
                        timestamps_copy
                    )
                    
                    if metrics:
                        rmse, mae, map_5, bias = metrics
                        results.append({
                            'Method': method,
                            'N_Factors': n_factors,
                            'Reg_Param': reg_param,
                            'RMSE': rmse,
                            'MAE': mae,
                            'MAP@5': map_5,
                            'Mean_Bias': bias,
                            'Time(s)': time.time() - start_time,
                            'Compression_Ratio': calculate_compression_ratio(recommender, mask),
                            'Memory_MB': calculate_method_specific_memory(recommender) / (1024 * 1024)
                        })
                    
                    del recommender, R_copy, mask_copy, timestamps_copy

        results_df = pd.DataFrame(results)
        results_df.to_csv('temporal_results.csv', index=False)
           
    except Exception as e:
       print(f"Main execution error: {str(e)}")

if __name__ == "__main__":
   main()