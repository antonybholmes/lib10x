import setuptools

setuptools.setup(
    name='lib10x',
    version='0.6.0',
    author='Antony B Holmes',
    author_email='antony.b.holmes@gmail.com',
    description='A library for analyzing 10X Genomics data.',
    url='https://github.com/antonybholmes/lib10x',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
