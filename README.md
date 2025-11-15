# money3

基于 Black-Litterman 模型的智能投资组合优化系统，结合 LLM 生成的市场观点进行资产配置。

## 功能特性

- **数据获取**: 从 Tiingo、Finnhub、FRED 获取历史价格、新闻和宏观数据
- **AI 观点生成**: 使用 DeepSeek LLM（通过 OpenAI 兼容 API）分析市场数据并生成投资观点
- **Black-Litterman 优化**: 结合市场均衡和主观观点进行组合优化
- **回测验证**: 使用简化的回测引擎（基于 Pandas/Numpy）评估投资组合表现
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
python3 api_server.py
```

然后在浏览器访问 `http://localhost:5000`

### 方式 2: 命令行脚本

```bash
# 运行主程序（完整流程）
python3 main.py
```

## 项目结构

```
money3/
├── money3/                # 核心功能包
│   ├── __init__.py
│   ├── data/             # 数据获取模块
│   ├── llm/              # LLM 集成模块
│   ├── opt/              # 优化器（Black-Litterman）
│   └── backtest/         # 回测引擎
├── static/               # 前端静态文件
│   └── index.html        # Web 前端页面
├── api_server.py         # Flask API 服务器
├── main.py               # 命令行主程序
├── pyproject.toml        # 项目配置（标准）
├── requirements.txt      # 依赖列表
└── .env                  # API Keys 配置
```

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


### 代码结构说明

- **入口文件**: `main.py`, `api_server.py` 等位于项目根目录
- **核心包**: `money3/` 包含所有可复用的功能模块
- **导入方式**: 统一使用 `from money3.xxx import ...` 绝对导入
