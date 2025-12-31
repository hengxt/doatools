from setuptools import setup, find_packages
import os
import re

def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "一个现代化的波达方向(DoA)估计工具库。基于原项目morriswmz/doatools.py的重构与增强版本。"

def get_version():
    """从_version.py读取版本号"""
    version_file = os.path.join('doatools', '_version.py')
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            version_match = re.search(
                r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", 
                f.read(), 
                re.M
            )
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
    

    return "0.3.0.dev1"

setup(
    name='doatools',
    version=get_version(),
    description='一个现代化的波达方向(DoA)估计工具库 - 重构与增强版本',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    url='https://github.com/hengxt/doatools',
    author='Xiantao Heng (Based on work by Mianzhi Wang)',
    author_email='hengxt@163.com', 
    
    # 包发现
    packages=find_packages(exclude=('doatools.tests', 'docs', 'examples', 'benchmarks')),
    
    # Python要求
    python_requires='>=3.8',
    
    # 依赖
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'cvxpy>=1.1.0',
        'tqdm>=4.65.0',
        'h5py>=3.9.0'
    ],
    
    # 可选依赖
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'sphinx>=7.0.0'
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0'
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta', 
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    

    zip_safe=False,
    include_package_data=True,
    
    keywords='doa direction-of-arrival estimation array-signal-processing music esprit',
    
    project_urls={
        'Documentation': 'https://github.com/hengxt/doatools#readme',
        'Source': 'https://github.com/hengxt/doatools',
        'Tracker': 'https://github.com/hengxt/doatools/issues',
    },
)