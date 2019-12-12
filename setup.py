from setuptools import setup, find_packages

setup(
    name='yake',
    packages=find_packages(),
    url='https://github.com/LIAAD/yake',
    description='yake package installation',
    long_description=open('README.md').read(),
    install_requires=[
        "yake",
        ],
    dependency_links=['https://github.com/LIAAD/yake/tarball/master#egg=package-1.0'],
    include_package_data=True,
)
