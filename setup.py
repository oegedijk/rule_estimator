from setuptools import setup, find_packages

setup(
    name='rule_estimator',
    version='0.3.0',
    package_dir={'rule_estimator': 'rule_estimator'},  # the one line where all the magic happens
    package_data={
        'rule_estimator': ['data/*'],
    },
    packages=find_packages(),
    description='Scikit-learn compatible business rule estimator',
    long_description='Scikit-learn compatible business rule estimator, with dashboard included',
    license='Apache-2',
    classifiers=["Development Status :: 3 - Alpha"],
    install_requires=['pandas', 'numpy', 'scikit-learn', 'oyaml', 'python-igraph', 'dash', 'dash-bootstrap-components'],
    python_requires='>=3.6',
    author='Oege Dijk',
    author_email='oegedijk@gmail.com',
    keywords=['machine learning'],
    url='https://github.com/oegedijk/rule_estimator',
)
