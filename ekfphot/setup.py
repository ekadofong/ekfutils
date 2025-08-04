import setuptools

setuptools.setup(
    name='ekfphot',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Astronomical photometry and image analysis utilities',
    long_description='Tools for astronomical photometry including filter curve handling, image processing, extinction corrections, and photometric measurements. Features support for GALEX filters and custom filter definitions.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
        'astropy',
        'extinction',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
