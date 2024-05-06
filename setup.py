from setuptools import setup

classifiers = [
    "Development Status :: 1 - Planning",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
]

with open("README.md") as f:
    long_description = f.read()

with open("requirements.docs.txt") as f:
    requirements = f.readlines()
install_requires = [r.strip() for r in requirements]

setup(
    name="cube_wrangler",
    version="0.0.1",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i-am-sijia/cube_wrangler",
    license="Apache 2",
    platforms="any",
    packages=["cube_wrangler"],
    include_package_data=True,
    install_requires=install_requires,
)
