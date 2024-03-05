from setuptools import setup, find_packages
import os

# Long description from README_pypi.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README_pypi.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
    name='multiviewstacking',
    version='0.0.3',
    description = 'Python implementation of the Multi-View Stacking algorithm.',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url='https://github.com/enriquegit/multiviewstacking',
    author = 'Enrique Garcia-Ceja',
	author_email = 'e.g.mx@ieee.org',
    license='MIT',
	keywords = ['multi-view stacking',
	             'machine learning',
	            'classification',
	            'sensor fusion'],
    package_data={'multiviewstacking': ['data/*.*']},
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn >= 1.2.2',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)