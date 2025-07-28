# 数据插入脚本使用说明

本目录包含用于将mock数据插入到MongoDB推荐系统数据库的脚本。

## 脚本说明

### 1. insert_likes_data.py - 简化版likes数据插入

**功能：** 专门用于将CSV数据快速插入到MongoDB的likes collection中

**特点：**
- 只插入likes数据（评分>=3的记录）
- 简单快速，适合快速测试
- 包含数据验证和统计功能

**使用方法：**
```bash
cd /path/to/recommendation
python scripts/insert_likes_data.py
```

### 2. insert_mock_data.py - 完整数据插入

**功能：** 全面的数据插入脚本，根据评分生成多种用户行为记录

**特点：**
- 根据评分智能生成read、likes、collection记录
- 生成内容标签数据（category、tags等）
- 模拟真实的用户行为模式
- 包含时间戳和完整的数据验证

**数据生成规则：**
- 评分 >= 2：生成阅读记录
- 评分 >= 3：70%概率生成点赞记录
- 评分 >= 4：50%概率生成收藏记录
- 评分 = 5：90%概率点赞，70%概率收藏

**使用方法：**
```bash
cd /path/to/recommendation
python scripts/insert_mock_data.py
```

## 数据库结构

### MongoDB Collections

1. **likes** - 点赞记录
   ```json
   {
     "user_id": 1001,
     "content_id": "news_001",
     "score": 5,
     "timestamp": "2024-01-01T10:00:00Z",
     "create_time": "2024-01-01T10:00:00Z"
   }
   ```

2. **read** - 阅读记录
   ```json
   {
     "user_id": 1001,
     "content_id": "news_001",
     "timestamp": "2024-01-01T10:00:00Z",
     "create_time": "2024-01-01T10:00:00Z"
   }
   ```

3. **collection** - 收藏记录
   ```json
   {
     "user_id": 1001,
     "content_id": "news_001",
     "timestamp": "2024-01-01T10:00:00Z",
     "create_time": "2024-01-01T10:00:00Z"
   }
   ```

4. **content_labels** - 内容标签
   ```json
   {
     "content_id": "news_001",
     "title": "新闻标题_news_001",
     "category": "科技",
     "tags": ["人工智能", "科技创新"],
     "create_time": "2024-01-01T10:00:00Z",
     "status": "published"
   }
   ```

## 前置条件

1. **MongoDB服务运行**
   ```bash
   # 确保MongoDB服务正在运行
   mongod --dbpath /path/to/your/db
   ```

2. **Python依赖**
   ```bash
   pip install pymongo
   ```

3. **数据文件**
   - 确保 `data/news_score/news_log.csv` 文件存在
   - 文件格式：user_id,score,content_id

## 使用流程

1. **准备环境**
   ```bash
   # 确保MongoDB服务运行
   # 确保CSV数据文件存在
   ls data/news_score/news_log.csv
   ```

2. **快速测试（推荐）**
   ```bash
   python scripts/insert_likes_data.py
   ```

3. **完整数据插入**
   ```bash
   python scripts/insert_mock_data.py
   ```

4. **验证数据**
   - 脚本会自动显示插入的数据统计
   - 可以通过MongoDB客户端查看数据

## 数据验证

脚本执行完成后会显示：
- 各collection的记录数量
- 用户行为统计
- 内容受欢迎度统计
- 示例数据记录

## 注意事项

1. **数据清理**：脚本会询问是否清空现有数据，建议在测试环境中使用
2. **数据量**：当前CSV包含99条记录，完整插入会生成数百条各类记录
3. **时间戳**：脚本生成的时间戳为最近30天内的随机时间
4. **评分阈值**：只有评分>=3的记录才会被认为是"喜欢"

## 故障排除

1. **MongoDB连接失败**
   - 检查MongoDB服务是否运行
   - 检查连接配置（host:127.0.0.1, port:27017）

2. **文件路径错误**
   - 确保在项目根目录执行脚本
   - 检查CSV文件是否存在

3. **权限问题**
   - 确保对MongoDB数据库有写入权限
   - 检查文件系统权限 