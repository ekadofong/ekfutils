import setuptools

setuptools.setup(
    name='ekfparse',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Text parsing and web scraping utilities for academic research',
    long_description='Comprehensive toolkit for parsing academic content including LaTeX documents, journal articles, email processing, ADS queries, and web scraping. Features custom container classes and string manipulation utilities for research workflows.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'requests',
        'beautifulsoup4',
        'numpy',
        'pandas',
        'astropy',
        'astroquery',
        'bibtexparser',
        'pylatexenc',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Markup :: LaTeX',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
