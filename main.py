"""入口转发：`python main.py` 等价于 `python -m dogfight`。"""
from dogfight.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
