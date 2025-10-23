from src import __version__
from src.data import DataClient
from src.llm import LLMClient
from src.opt import Optimizer


def main() -> None:
    # 链式依赖：data -> llm -> opt
    data = DataClient(["SPY", "GLD", "TLT"])
    llm = LLMClient(data)
    opt = Optimizer(llm)

    result = opt.optimize()

    print(f"money3 {__version__}")
    print(result)


if __name__ == "__main__":
    main()
