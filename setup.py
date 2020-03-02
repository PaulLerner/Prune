from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(

    name='Prune',
    packages=find_packages(),
    scripts=[
        "prune/stats.py",
        "prune/visualize.py",
        "prune/convert.py"
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Paul Lerner',
    author_email='lerner@limsi.fr',
    url='https://github.com/PaulLerner/Prune',

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
)
