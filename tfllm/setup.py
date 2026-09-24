# -*- coding: utf-8 -*-
"""Package the CMake-installed TFLLM tree, not the source checkout.

Run `python -m pip install .` in SDK/tfllm after `cmake --install`.
Native binaries remain in tfllm_runtime/sdk; public tfllm.* Python packages
are also installed at the normal site-packages level. No native compilation.
"""
import os
from pathlib import Path
import re
import shutil

from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

HERE = Path(__file__).resolve().parent
SDK = HERE.parent
SDK_SUBDIRS = ("bin", "lib", "include", "schema", "llama", "share")
SDK_FILES = ("README.md", "TORCH.md", "PIP.md")
DRIVERS = ("libNPU40T.so", "libNPU40T.dylib")
PYTHON_TREE = HERE / "share" / "tfllm" / "python"


def installed_packages():
    for name in ("bin", "lib", "include", "schema", "python/tfllm_runtime"):
        if not (HERE / name).is_dir():
            raise SystemExit(f"[setup] Missing {HERE / name}; run the parent's CMake install first (WITH_TFLLM=ON)")
    packages = find_packages(str(PYTHON_TREE), include=("tfllm", "tfllm.*"))
    has_torch = "tfllm.torch" in packages
    has_library = any((HERE / "lib" / name).is_file() for name in ("libtfllm-torch.so", "libtfllm-torch.dylib"))
    if has_torch != has_library:
        raise SystemExit("[setup] Incomplete Torch installation: reinstall with TFLLM_WITH_TORCH=ON into a clean prefix")
    return ["tfllm_runtime"] + packages


def copy_sdk_tree(destination):
    """Include the installed SDK, but never test/, build/ or Python bytecode."""
    for name in SDK_SUBDIRS:
        src = HERE / name
        if src.is_dir():  # llama/share are optional in core-only installations.
            shutil.copytree(src, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    for name in SDK_FILES:
        if (HERE / name).is_file():
            shutil.copy2(HERE / name, destination / name)
    for name in DRIVERS:
        driver = SDK / "lib" / name
        if driver.is_file():
            shutil.copy2(driver, destination / "lib" / name)
    if not any((destination / "lib" / name).is_file() for name in DRIVERS):
        print("[setup] No SDK NPU40T driver found; an NPU-enabled build will require its driver")


class build_py(_build_py):
    def run(self):
        if self.editable_mode:
            raise SystemExit("[setup] Editable install is unsupported for a bundled native SDK; use pip install .")
        output = Path(self.build_lib).resolve()
        # Only clean our generated packages, never an input installation tree.
        for source in [HERE / "python"] + [HERE / name for name in SDK_SUBDIRS]:
            source = source.resolve()
            if output == source or output in source.parents or source in output.parents:
                raise SystemExit("[setup] build_lib must not overlap the installed SDK input directories")
        for name in ("tfllm", "tfllm_runtime"):
            generated = output / name
            if generated.is_symlink():
                raise SystemExit(f"[setup] Refusing symlink build output: {generated}")
            if generated.exists():
                shutil.rmtree(generated)  # Prevent stale libraries/modules in repeat builds.
        super().run()
        copy_sdk_tree(output / "tfllm_runtime" / "sdk")


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False  # Contains native executables/shared libraries.
        if os.environ.get("TFLLM_WHEEL_PLATFORM"):
            self.plat_name = os.environ["TFLLM_WHEEL_PLATFORM"]
            self.plat_name_supplied = True

    def get_tag(self):
        _, _, platform = super().get_tag()
        # ctypes and exec do not depend on a particular CPython extension ABI.
        return "py3", "none", platform


class sdist(_sdist):
    def run(self):
        raise SystemExit("[setup] This is a prebuilt SDK, not a source distribution; use pip wheel . --no-deps")


def console_scripts():
    names = sorted(p.name for p in (HERE / "bin").iterdir()
                   if p.is_file() and os.access(p, os.X_OK) and not p.name.startswith("."))
    if any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) for name in names):
        raise SystemExit("[setup] Unsupported executable name in SDK bin/")
    return [f"{name} = tfllm_runtime:launch" for name in names]


setup(
    name="tfllm",
    version=os.environ.get("TFLLM_VERSION", "0.1.0"),
    description="ThinkForce native inference runtime, CLI and optional PyTorch adapters",
    long_description="Prebuilt TFLLM SDK with CLI launchers and optional tfllm.torch modules.",
    long_description_content_type="text/markdown",
    package_dir={"": "python", "tfllm": "share/tfllm/python/tfllm"},
    packages=installed_packages(),
    cmdclass={"build_py": build_py, "bdist_wheel": bdist_wheel, "sdist": sdist},
    entry_points={"console_scripts": console_scripts()},
    install_requires=["jinja2>=3.1,<4"],
    extras_require={"torch": ["torch>=2.4", "safetensors>=0.4"]},
    python_requires=">=3.9",
    include_package_data=False,
    zip_safe=False,
)
