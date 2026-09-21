from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    """Mark the wheel as CPython/platform specific because it contains .so files."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)

