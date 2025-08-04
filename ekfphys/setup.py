import setuptools

setuptools.setup(
    name='ekfphys',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Astrophysical calculations and modeling utilities',
    long_description='Collection of tools for astrophysical calculations including initial mass functions (Kroupa IMF), stellar lifetime estimates, K-corrections, observer coordinate transformations, calibration routines, and physical modeling utilities for astronomical research.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'pandas',
        'extinction',
        'healpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
