import setuptools

setuptools.setup(
    name='ekfstats',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Statistical analysis and model fitting toolkit for astronomy',
    long_description='Comprehensive statistical toolkit featuring curve fitting (including Sersic profiles), MCMC sampling with emcee, image statistics, Bayesian inference utilities, mathematical functions, and specialized statistical tools for astronomical data analysis.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'emcee',
        'numba',
        'statsmodels',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
