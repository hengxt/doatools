from setuptools import setup, find_packages
import string

# Load description
def find_long_description():
    with open('README.md', 'r') as f:
        return f.read()

# Read the version string
def find_version():
    version_var_name = '__version__'
    with open('doatools/_version.py', 'r') as f:
        for l in f:
            if not l.startswith(version_var_name):
                continue
            return l[len(version_var_name):].strip(string.whitespace + '\'"=')
        raise RuntimeError('Unable to read version string.')

setup(
    name='doatools',
    version=find_version(),
    description='A collection of tools for DOA estimation related research.',
    long_description=find_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/morriswmz/doatools.py',
    author='Mianzhi Wang',
    # author_email='',
    packages=find_packages(exclude=('docs',)),
    python_requires='>=3.11',
    install_requires=[
        'numpy>=2.0.0',
        'scipy>=1.15.2',
        'matplotlib>=3.10.0',
        'cvxpy>=1.6.4'
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ]
)
