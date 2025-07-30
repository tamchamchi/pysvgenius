from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pysvgenius",
    version="0.1.5.3",
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
    extras_require={
        "diffvg": [
            # "pydiffvg @ git+https://github.com/BachiLi/diffvg.git"
        ],
        "clip": [
            # "clip@git+https://github.com/openai/CLIP.git"
        ],
        "diff_jpeg": [
            # "diff_jpeg@git+https://github.com/necla-ml/Diff-JPEG"
        ]
    },
    python_requires=">=3.10",
    include_package_data=True,
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

