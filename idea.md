# 单源多目标域自适应（SSMTDA）基于SAUE风格迁移与COCOOP的实现方案

## 📌 项目概述
本项目基于SAUE论文中的风格迁移思想，针对**单源多目标域（Single-Source Multi-Target Domain Adaptation, SSMTDA）**场景构建领域自适应模型。核心创新点在于将SAUE的风格适应模块与COCOOP的提示微调技术结合，解决多目标域分离（非混合）场景下的域适应问题。

> ✅ 与SAUE原始设定的区别：  
> - SAUE处理**混合目标域（BTDA）**（无域标签、目标域混合）  
> - 本项目处理**分离目标域（MTDA）**（有明确域标签、各目标域独立）

---

## 🧠 模型结构设计

### 核心流程
```python
# 1. 预计算目标域统计量（离线完成）
domain_stats = {
    "目标域1": (μ1, σ1),  # 域内所有样本的浅层特征均值/方差平均
    "目标域2": (μ2, σ2),
    ...
}

# 2. 训练时对每个batch：
source_features = shallow_vit(source_images)  # ViT第4层特征
μ_source = source_features.mean(dim=1)       # 样本级均值
σ_source = source_features.std(dim=1)        # 样本级标准差

# 归一化后用目标域统计量反归一化
z_norm = (source_features - μ_source.unsqueeze(1)) / (σ_source.unsqueeze(1) + 1e-8)
z_style = z_norm * μ_target.unsqueeze(1) + σ_target.unsqueeze(1)  # 风格迁移

# 两条路径处理
normal_cls = deep_vit(source_features)[:, 0]    # 原始路径CLS
style_cls = deep_vit(z_style)[:, 0]             # 风格迁移路径CLS

# 可学习权重融合（关键！）
w_style = nn.Parameter(torch.ones(1))
final_cls = w_style * MLP_style(style_cls) + (1 - w_style) * MLP_normal(normal_cls)

# 3. COCOOP集成
prompt_tokens = meta_net(final_cls)  # 用融合后的CLS生成prompt
```

### COCOOP集成要点
- **不直接使用COCOOP原始代码**，需重写prompt生成逻辑
- 将传统COCOOP的输入（原始图像特征）替换为**融合后的CLS特征**
- 修改CLIP的forward过程：
  ```python
  def forward(self, x):
      patches = self.visual(x)  # ViT提取patch特征
      # 插入prompt_tokens到patch序列前
      tokens = torch.cat([prompt_tokens, patches], dim=1)
      return self.transformer(tokens)
  ```

---

## 📂 数据集处理规范

### 📊 数据集选择
| 数据集 | 域数量 | 类别数 | 推荐任务 |
|--------|--------|--------|----------|
| **Office-Home** | 4 (Art, Clipart, Products, RealWorld) | 65 | 源域=Art，目标域=Clipart/Products/RealWorld（独立测试） |
| **Office-31** | 3 (Amazon, Webcam, DSLR) | 31 | 源域=Amazon，目标域=Webcam+DSLR（独立测试） |

### 🔧 数据预处理步骤
1. **离线计算目标域统计量**（对每个目标域单独处理）：
   ```python
   # 以Office-Home的Clipart域为例
   clipart_stats = []
   for img in clipart_dataset:
       shallow_feat = viT_layer4(img)  # ViT第4层特征 [B, seq_len, dim]
       μ = shallow_feat.mean(dim=1)    # [B, dim]
       σ = shallow_feat.std(dim=1)     # [B, dim]
       clipart_stats.append((μ, σ))
   
   # 域级统计量 = 所有样本的统计量平均
   μ_clipart = torch.stack([x[0] for x in clipart_stats]).mean(dim=0)
   σ_clipart = torch.stack([x[1] for x in clipart_stats]).mean(dim=0)
   ```

2. **训练时按目标域标签动态获取统计量**：
   ```python
   # 假设当前batch来自"Products"域
   μ_target, σ_target = domain_stats["Products"]
   ```

---

## 📈 实验设计规范

### ✅ 基线模型选择
| 模型 | 论文 | 发表会议 | 说明 |
|------|------|----------|------|
| **HGAN** | *Heterogeneous Graph Attention Network for Unsupervised Multiple-Target Domain Adaptation* | NeurIPS 2022 | 最新且效果突出，支持分离目标域 |
| **CGCT** | *Curriculum Graph Co-Teaching for Multi-Target Domain Adaptation* | CVPR 2021 | 代码成熟，复现简单 |
| **MTDA-ITA** | *Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach* | IEEE TIP 2020 | 信息论驱动的MTDA方法 |

> ❌ **避免使用**：  
> - SAUE论文中的MCDA（AAAI 2023）——这是**混合目标域（BTDA）**方法  
> - DomainNet（默认用于多源混合目标域）  

### 📊 评估指标
- **Office-Home任务示例**：  
  - 源域 = Art  
  - 目标域 = Clipart, Products, RealWorld（**独立测试，不混合**）  
  - 最终指标 = 三个目标域的平均准确率  
- **正确做法**：  
  ```python
  # 测试时分别计算每个目标域的准确率
  clipart_acc = evaluate(model, clipart_testset)
  products_acc = evaluate(model, products_testset)
  realworld_acc = evaluate(model, realworld_testset)
  final_score = (clipart_acc + products_acc + realworld_acc) / 3
  ```

---

## 🚫 常见错误规避
1. **目标域混合错误**：  
   - ❌ 将Clipart+Products+RealWorld视为一个"大目标域"（这是BTDA设定）  
   - ✅ 必须**单独测试每个目标域**，最后计算平均值  

2. **统计量计算错误**：  
   - ❌ 直接对原始特征计算域级均值/方差  
   - ✅ 先对每个样本计算**样本级**统计量，再平均得到域级统计量  

3. **融合方式错误**：  
   - ❌ 使用1D卷积融合CLS特征（CLS是单向量，无序列维度）  
   - ✅ 采用**可学习权重**：`w_style * MLP_style(style_cls) + (1-w_style) * MLP_normal(normal_cls)`  

---

## 🛠️ 实验步骤速查
1. **预处理阶段**：  
   - 对每个目标域计算域级统计量并保存  
   - 存储格式：`domain_stats = {"domain1": (μ, σ), "domain2": (μ, σ), ...}`

2. **训练阶段**：  
   - 每个batch按目标域标签获取对应统计量  
   - 生成风格迁移特征 + 原始路径特征  
   - 用融合后的CLS生成prompt  
   - 交叉熵损失训练

3. **测试阶段**：  
   - 对每个目标域单独测试  
   - 报告平均准确率

> 💡 **验证建议**：  
> 先单独验证风格迁移效果（可视化t-SNE），再逐步集成COCOOP。SAUE的ablation study显示风格迁移模块单独贡献+2.8%性能提升！