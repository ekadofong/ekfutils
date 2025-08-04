import setuptools

setuptools.setup(
    name='ekfplot',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Scientific plotting and visualization utilities for astronomy',
    long_description='Comprehensive plotting toolkit for astronomical research featuring color schemes, legend utilities, scientific plot formatting, axis transformations, and specialized visualization functions for astrophysical data presentation.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
