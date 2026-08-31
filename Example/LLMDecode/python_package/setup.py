"""Setuptools entry point kept compatible with the target's offline venv."""

from setuptools import setup
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - pip wheel provides this on targets
    bdist_wheel = None


class BinaryDistribution(Distribution):
    """The wheel embeds an ELF binary and must not be tagged pure-Python."""

    def has_ext_modules(self) -> bool:
        return True


class BinaryWheel(bdist_wheel):  # type: ignore[misc,valid-type]
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False


setup(
    name="tfdl-llmdecode",
    version="0.1.0",
    description="Persistent llama.cpp decode worker for external GELab KV caches",
    python_requires=">=3.10",
    install_requires=["numpy>=1.23"],
    packages=["tfdl_llmdecode"],
    package_data={"tfdl_llmdecode": ["bin/*"]},
    include_package_data=True,
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": BinaryWheel} if bdist_wheel is not None else {},
)
