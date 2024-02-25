from setuptools import setup, find_packages
import os

# Long description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README_pypi.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    name='multiviewstacking',
    version='0.0.2',
    description = 'Python implementation of the Multi-View Stacking algorithm.',
    long_description = long_description,
    long_description_context_type = 'text/markdown',
    author = 'Enrique Garcia-Ceja',
	author_email = 'e.g.mx@ieee.org',
    license='MIT',
	keywords = ['multi-view stacking',
	             'machine learning',
	            'classification',
	            'sensor fusion'],
    package_data={'multiviewstacking': ['data/*.*']},
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'pandas >= 2.0.3',
        'numpy >= 1.24.3',
        'scikit-learn >= 1.3.0',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)