# Always prefer setuptools over distutils
import setuptools
from setuptools import setup
setup(
    name='classifier_nsfw',
    version='0.1',
    author='Shabaz Patel',
    author_email='shabaz@acusense.ai',
    license='See LICENSE.txt',
    keywords='ai api ml development',
    packages=setuptools.find_packages(),
    setup_requires=['pbr>=1.8', 'setuptools>=17.1',
                    'scipy','numpy'],
    extras_require={
        'test': ['coverage'],
    },
    package_data={
        'indexer': ['config.yaml'],
        'indexer.nsfw_model': ['*.prototxt', '*.caffemodel'],
    },
)

