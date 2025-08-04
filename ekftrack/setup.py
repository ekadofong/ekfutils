import setuptools

setuptools.setup(
    name='ekftrack',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Personal productivity and task tracking utilities',
    long_description='Simple productivity tracking system for managing daily tasks with points-based scoring. Features task categorization (checklist, procedural, exploratory) and persistent logging for productivity analysis.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Office/Business :: Scheduling',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
