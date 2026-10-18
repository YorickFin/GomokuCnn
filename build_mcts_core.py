"""编译 src/mcts_core.cpp → src/mcts_core.pyd（MSYS2 UCRT64 g++）。

用法（在项目 venv 的 Python 下运行）：
    python build_mcts_core.py
    python build_mcts_core.py --clean   # 先删除旧 pyd 再编译

产物 src/mcts_core.pyd 输出到 src/ 下，可 `from src import mcts_core` 导入。
链接的是运行当前解释器的 python3xx.dll（venv 与基础安装共享同一 ABI）。
"""

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "mcts_core.cpp"
OUT = ROOT / "src" / "mcts_core.pyd"

# 常见 MSYS2 g++ 位置；也可用环境变量 GXX 覆盖
GXX_CANDIDATES = [
    r"D:\Programs\msys64\ucrt64\bin\g++.exe",
    r"C:\msys64\ucrt64\bin\g++.exe",
    r"D:\msys64\ucrt64\bin\g++.exe",
    r"C:\Programs\msys64\ucrt64\bin\g++.exe",
]


def find_gxx():
    import os

    if os.environ.get("GXX"):
        p = Path(os.environ["GXX"])
        if p.is_file():
            return p
    for cand in GXX_CANDIDATES:
        if Path(cand).is_file():
            return Path(cand)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="先删除旧 pyd")
    args = parser.parse_args()

    if args.clean and OUT.exists():
        OUT.unlink()

    gxx = find_gxx()
    if gxx is None:
        sys.exit("未找到 g++，请安装 MSYS2 UCRT64 工具链或设置环境变量 GXX")

    import pybind11

    include_py = sysconfig.get_paths()["include"]
    # venv 的 sysconfig 没有 libs 条目，python3xx.lib 位于基础安装的 libs/ 下
    lib_dir = Path(sys.base_prefix) / "libs"
    version = sysconfig.get_config_var("VERSION") or "%d%d" % sys.version_info[:2]
    lib_name = f"python{version.replace('.', '')}"
    lib_file = lib_dir / f"{lib_name}.lib"
    if not lib_file.is_file():
        sys.exit(f"未找到导入库 {lib_file}")

    cmd = [
        str(gxx),
        "-O3",
        "-std=c++17",
        "-shared",
        "-static-libgcc",
        "-static-libstdc++",
        f"-I{include_py}",
        f"-I{pybind11.get_include()}",
        "mcts_core.cpp",
        f"-L{lib_dir}",
        f"-l{lib_name}",
        "-o",
        "mcts_core.pyd",
        # UCRT64 gcc 为 posix 线程模型，libstdc++ 引用 pthread 符号；
        # 将 libwinpthread 整体静态链入，避免运行期依赖该 DLL
        "-Wl,-Bstatic,--whole-archive,-lwinpthread,--no-whole-archive,-Bdynamic",
    ]
    # mingw 的 g++/ld 按 ANSI 代码页解析参数，项目路径含中文时无法
    # 读写文件，因此复制源码到 ASCII 临时目录中编译，产物再拷回。
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="mcts_core_build_"))
    try:
        shutil.copy2(SRC, tmp / "mcts_core.cpp")
        print("[build]", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=tmp)
        if OUT.exists():
            OUT.unlink()
        shutil.copy2(tmp / "mcts_core.pyd", OUT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
