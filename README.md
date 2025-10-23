# money3

最小可运行骨架：LLM + Black-Litterman + vectorbt 回测原型。

## 本地环境初始化（macOS + Homebrew）

- 检查 Python3：
```bash
python3 --version
```
- 创建并激活虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
```
- 使用国内镜像安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- VSCode 将自动使用 `.vscode/settings.json` 中的解释器。

## 运行

```bash
python main.py
```