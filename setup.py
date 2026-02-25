from setuptools import find_packages
from distutils.core import setup

setup(
    name="g1_deploy",
    version="1.0.0",
    author="Unitree Robotics",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="support@unitree.com",
    description="Template RL environments for Unitree Robots",
    install_requires=[
        "matplotlib",
        "pyyaml",
        "onnx==1.20.0",
        "onnxruntime==1.23.2",
        "mujoco==3.2.7",
        "opencv-python-headless==4.11.0.86"
    ],
)
