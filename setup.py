import os
import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install


def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

class CustomInstall(install):
    def run(self):
        # 1. Cài package chính trước
        install.run(self)

        # 2. Chuẩn bị thư mục diffvg
        base_dir = os.path.dirname(__file__)
        diffvg_dir = os.path.join(base_dir, "pysvgenius", "diffvg")

        # Clone repo nếu chưa tồn tại
        if not os.path.exists(diffvg_dir):
            print("=== Cloning diffvg repository ===")
            subprocess.check_call([
                "git", "clone", "https://github.com/BachiLi/diffvg.git", diffvg_dir
            ])
        else:
            print(f"=== Found existing diffvg at {diffvg_dir} ===")

        # Update submodules
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=diffvg_dir
        )

        # Xóa build cũ để tránh lỗi cmake cache
        build_dir = os.path.join(diffvg_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # 3. Build và cài diffvg vào env hiện tại
        print("=== Installing diffvg into current environment ===")
        subprocess.check_call(
            [sys.executable, "setup.py", "install"],
            cwd=diffvg_dir
        )

        # 4. Kiểm tra import pydiffvg
        try:
            import importlib
            pydiffvg = importlib.import_module("pydiffvg")
            print(f"✅ Installed pydiffvg at: {pydiffvg.__file__}")
        except ImportError:
            print("⚠ Warning: pydiffvg không import được. Kiểm tra log build.")
setup(
    name="pysvgenius",
    version="0.1.5",
    author="Anh Nguyen",
    author_email="anhndt.work@gmail.com",
    description="A library for text_to_svg, image_to_svg, and SVG resizing and optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    url="https://github.com/tamchamchi/pysvgenius",
    project_urls={
        "Bug Tracker": "https://github.com/tamchamchi/pysvgenius/issues",
        "Documentation": "https://github.com/tamchamchi/pysvgenius#readme"
    },

    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_requires={
        "diffvg": [
            "git+https://github.com/BachiLi/diffvg.git"
        ],
        "clip": [
           "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
        ],
        "diff_jpeg": [
            "git+https://github.com/necla-ml/Diff-JPEG@e81f082896ba145e35cc129bc7743244e10881e5"
        ]
    },
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        "pysvgenius": ["diffvg/*", "diffvg/**/*"]
    },
    cmdclass={
        'install': CustomInstall,
    },
)
