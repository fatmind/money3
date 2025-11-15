# money3

基于 Black-Litterman 模型的智能投资组合优化系统，结合 LLM 生成的市场观点进行资产配置。

## 功能特性

- **数据获取**: 从 Tiingo、Finnhub、FRED 获取历史价格、新闻和宏观数据
- **AI 观点生成**: 使用 OpenAI GPT-4 分析市场数据并生成投资观点
- **Black-Litterman 优化**: 结合市场均衡和主观观点进行组合优化
- **回测验证**: 评估投资组合表现，包括收益曲线、回撤、夏普比率等
- **Web 界面**: 前后端分离架构，Flask API + 纯前端页面

## 环境初始化

### 1. 检查 Python 版本
```bash
python3 --version  # 需要 >= 3.9
```

### 2. 创建并激活虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
# 使用国内镜像（推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用默认源
pip install -r requirements.txt
```

### 4. 安装项目为可编辑包（重要！）
```bash
pip install -e .
```

这一步会将 `money3` 包安装为可编辑模式，之后就可以在任何地方直接 `import money3`，**不需要手动设置 PYTHONPATH**。

### 5. 配置 API Keys

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_openai_api_key
TIINGO_API_KEY=your_tiingo_api_key
FINNHUB_API_KEY=your_finnhub_api_key
FRED_API_KEY=your_fred_api_key
```

## 运行方式

### 方式 1: Web 界面（推荐）

```bash
# 使用启动脚本
./run_webapp.sh

# 或直接运行
python3 api_server.py
```

然后在浏览器访问 `http://localhost:5000`

### 方式 2: 命令行脚本

```bash
# 运行主程序（完整流程）
python3 main.py

# 运行测试脚本
python3 bl_test_main.py              # 测试 Black-Litterman 优化
python3 money3/backtest/bt_test_main.py  # 测试回测功能
```

## 项目结构

```
money3/
├── money3/                # 核心功能包（原 src/）
│   ├── __init__.py
│   ├── data/             # 数据获取模块
│   ├── llm/              # LLM 集成模块
│   ├── opt/              # 优化器（Black-Litterman）
│   └── backtest/         # 回测引擎
├── static/               # 前端静态文件
│   └── index.html        # Web 前端页面
├── api_server.py         # Flask API 服务器
├── main.py               # 命令行主程序
├── bl_test_main.py       # Black-Litterman 测试
├── pyproject.toml        # 项目配置（标准）
├── requirements.txt      # 依赖列表
├── run_webapp.sh         # Web 应用启动脚本
└── .env                  # API Keys 配置（需自行创建）
```

## 技术架构

- **包管理**: 标准 Python 包结构 + `pyproject.toml`
- **后端**: Flask + Python
- **前端**: 纯 HTML + JavaScript + ECharts
- **数据**: Tiingo (价格), Finnhub (新闻), FRED (宏观)
- **AI**: OpenAI GPT-4
- **优化**: Black-Litterman 模型
- **回测**: 自研回测引擎

## 为什么不需要设置 PYTHONPATH？

使用 `pip install -e .` 将项目安装为**可编辑模式**后：

1. ✅ Python 能自动找到 `money3` 包
2. ✅ 所有 `from money3.xxx import ...` 都能正常工作
3. ✅ 不需要手动修改 `sys.path`
4. ✅ 符合 Python 项目的标准实践

这是现代 Python 项目的**推荐做法**。

## API 接口

### POST /api/optimize

投资组合优化接口

**请求:**
```json
{
  "tickers": ["SPY", "GLD", "TLT"],
  "start_date": "2025-09-01",
  "end_date": "2025-10-30",
  "backtest_days": 30
}
```

**返回:**
```json
{
  "status": "success",
  "data": {
    "views": {...},
    "optimization": {
      "weights": {...},
      "posterior_returns": {...}
    },
    "backtest": {
      "metrics": {...},
      "equity_curve": [...],
      "drawdown": [...]
    }
  }
}
```

### GET /api/health

健康检查接口

### GET /api/test

使用示例数据测试（不调用外部 API）

## 开发说明

### 添加新的依赖

```bash
# 修改 pyproject.toml 中的 dependencies
# 然后重新安装
pip install -e .
```

### 运行测试

```bash
python3 bl_test_main.py
python3 money3/backtest/bt_test_main.py
```

### 代码结构说明

- **入口文件**: `main.py`, `api_server.py`, `bl_test_main.py` 等位于项目根目录
- **核心包**: `money3/` 包含所有可复用的功能模块
- **导入方式**: 统一使用 `from money3.xxx import ...` 绝对导入
- **无需路径hack**: 不需要在代码中添加 `sys.path.insert()` 等路径设置
