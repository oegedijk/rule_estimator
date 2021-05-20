from setuptools import setup, find_packages

setup(
    name='rule_estimator',
    version='0.0.1',
    description='Scikit-learn compatible business rule estimator',
    long_description='Scikit-learn compatible business rule estimator',
    license='MIT',
    classifiers=["Development Status :: 3 - Alpha"],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'oyaml'],
    python_requires='>=3.6',
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning'],
    url='https://github.com/oegedijk/rule_estimator',
)
