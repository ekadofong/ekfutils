import setuptools

setuptools.setup(
    name='ekfjobs',
    version='0.0.1',
    author='Erin Kado-Fong',
    description='Email template generator for academic colloquium announcements and meeting requests',
    long_description='Utilities for generating formatted emails for academic colloquium speakers, including announcement templates and meeting request templates with proper pronoun handling.',
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        # No external dependencies required
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
