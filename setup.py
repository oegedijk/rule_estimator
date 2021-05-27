from setuptools import setup, find_packages

setup(
    name='rule_estimator',
    version='0.2.0',
    packages=find_packages(),
    description='Scikit-learn compatible business rule estimator',
    long_description='Scikit-learn compatible business rule estimator',
    license='Apache-2',
    classifiers=["Development Status :: 3 - Alpha"],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'oyaml', 'python-igraph'],
    python_requires='>=3.6',
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning'],
    url='https://github.com/oegedijk/rule_estimator',
)
