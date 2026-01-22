# VSIG项目架构说明

## 项目逻辑流程

```
1. 初始化模型
   ↓
2. 扫描数据根目录，找到所有指令文件夹
   ↓
3. 对每个指令文件夹：
   a. 读取视频、annotation、description（DataLoader）
   b. 格式化GT数据（GTFormatter）
   c. 对每个样本进行推理
   d. 评估结果
```

## 核心模块说明

### 1. `src/data_loader.py` - 数据加载模块

**职责**：统一处理视频、annotation、description的读取

**主要方法**：
- `load_annotations(annotation_path)`: 加载annotation文件
- `load_description(description_path)`: 加载description文件，返回选项定义和答案映射
- `list_videos(video_dir)`: 列出视频目录中的所有视频文件
- `filter_annotations_by_videos(annotations, video_files)`: 根据视频文件列表过滤annotation数据
- `prepare_dataset(video_dir, annotation_path, description_path)`: 准备数据集，返回(数据集列表, 选项定义文本, 视频答案映射)
- `scan_data_root(data_root)`: 扫描数据根目录，找到所有指令文件夹

**使用示例**：
```python
from src.data_loader import DataLoader

# 准备数据集
dataset, options_text, answers_map = DataLoader.prepare_dataset(
    video_dir="path/to/videos",
    annotation_path="path/to/annotations.json",
    description_path="path/to/description.txt"
)

# 扫描数据根目录
instruction_dirs = DataLoader.scan_data_root("data_new")
```

### 2. `src/gt_formatter.py` - GT格式化模块

**职责**：从annotation和description中提取评估所需的GT格式

**主要方法**：
- `format_gt_for_evaluation(annotation_item, video_dir, answers_map)`: 格式化单个GT数据用于评估
- `format_batch_gt_for_evaluation(annotations, video_dir, answers_map)`: 批量格式化GT数据
- `extract_evaluation_gt(formatted_gt)`: 从格式化后的GT中提取评估所需的字段

**格式化后的GT包含**：
- 原始annotation数据
- `_video_dir`: 视频目录路径（用于评估时定位视频）
- `_processed_gt`: 处理后的GT items（用于空间/时间评估）
- `_correct_options`: 正确答案选项列表（用于intent评估）

**使用示例**：
```python
from src.gt_formatter import GTFormatter

# 格式化GT数据
formatted_gt = GTFormatter.format_gt_for_evaluation(
    annotation_item, video_dir, answers_map
)

# 批量格式化
formatted_list = GTFormatter.format_batch_gt_for_evaluation(
    annotations, video_dir, answers_map
)
```

### 3. `main.py` - 主程序

**主要流程**：

1. **初始化模型**
   ```python
   model = initialize_model(logger)
   ```

2. **扫描数据根目录**
   ```python
   instruction_dirs = DataLoader.scan_data_root(data_root_dir)
   ```

3. **处理每个指令文件夹**
   ```python
   # 读取数据
   dataset, options_text, answers_map = DataLoader.prepare_dataset(
       video_dir, annotation_path, description_path
   )
   
   # 格式化GT数据
   formatted_gt_list = GTFormatter.format_batch_gt_for_evaluation(
       dataset, video_dir, answers_map
   )
   
   # 推理
   predictions, ground_truths = process_instruction_directory(...)
   
   # 评估
   metrics = Evaluator.evaluate_batch(predictions, ground_truths)
   ```

## 数据流

```
数据根目录 (data_new/)
├── 数据集1/
│   ├── 指令1/
│   │   ├── annotations.json      # 标注数据
│   │   ├── description.txt       # 选项定义和答案
│   │   └── *.mp4                 # 视频文件
│   ├── 指令2/
│   └── ...
└── 数据集2/
    └── ...
```

## 评估流程

1. **Intent Grounding评估**：
   - 从`description.txt`中读取正确答案选项
   - 比较模型输出的`selected_options`与正确答案
   - 计算准确率

2. **Spatial Grounding评估**：
   - 从`annotations.json`中提取GT点坐标
   - 比较模型输出的`point_list`与GT点
   - 使用距离阈值或mask检查

3. **Temporal Grounding评估**：
   - 从`annotations.json`中提取ASR时间戳
   - 比较模型输出的`timestamp`与ASR时间戳
   - 计算IoU

## 代码改进点

1. **模块化**：将数据加载和GT格式化逻辑分离到独立模块
2. **清晰性**：每个模块职责单一，逻辑清晰
3. **可维护性**：易于扩展和修改
4. **可测试性**：模块化设计便于单元测试

## 使用建议

- 如果需要修改数据加载逻辑，只需修改`src/data_loader.py`
- 如果需要修改GT格式化逻辑，只需修改`src/gt_formatter.py`
- 如果需要修改评估逻辑，只需修改`src/eval/metrics.py`
- 主程序`main.py`主要负责流程控制和协调各模块
