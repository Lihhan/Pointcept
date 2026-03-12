import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

# Add Conda CUDA nvvm/bin to PATH if it exists (needed for cicc)
# Also set CUDA_HOME to help PyTorch find the CUDA toolchain
if "CONDA_PREFIX" in os.environ:
    conda_prefix = os.environ["CONDA_PREFIX"]
    nvvm_bin = os.path.join(conda_prefix, "nvvm", "bin")
    if os.path.exists(nvvm_bin):
        current_path = os.environ.get("PATH", "")
        if nvvm_bin not in current_path:
            os.environ["PATH"] = f"{nvvm_bin}:{current_path}"
            print(f"Added {nvvm_bin} to PATH for CUDA toolchain")
    
    # Set CUDA_HOME to Conda CUDA if not already set
    if "CUDA_HOME" not in os.environ:
        # Check if targets/x86_64-linux exists (Conda CUDA structure)
        conda_cuda = os.path.join(conda_prefix, "targets", "x86_64-linux")
        if os.path.exists(conda_cuda):
            os.environ["CUDA_HOME"] = conda_cuda
            print(f"Set CUDA_HOME to {conda_cuda}")
        elif os.path.exists(os.path.join(conda_prefix, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = conda_prefix
            print(f"Set CUDA_HOME to {conda_prefix}")

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
