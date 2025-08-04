import setuptools

setuptools.setup(
    name='ekfobs',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Astronomical observation planning and site utilities for telescopes',
    long_description='Tools for astronomical observation planning including airmass calculations, moon tracking, sun rise/set times, observing site definitions (CTIO, Palomar), and target list generation for observing runs.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'astropy',
        'pandas',
        'matplotlib',
        'pytz',
        'astroquery',
        'progressbar',
        'observing-suite',
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
