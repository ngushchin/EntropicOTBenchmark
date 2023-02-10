from setuptools import setup

packages = {
    'eot_benchmark': 'benchmark/'
}

setup(
    name="eot_benchmark",
    version='0.1',
    author="Gushchin Nikita (thesinsay)",
    license="MIT",
    description="""""",
    packages=packages,
    package_dir=packages,
    include_package_data=False,
)
