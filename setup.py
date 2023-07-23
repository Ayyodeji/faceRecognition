from setuptools import find_packages, setup

setup(
    name='facial_recognition',
    packages=find_packages(include=['facial_recognition']),
    version='0.1.0',
    description='Facial Recognition Library',
    author='Your Name',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'tensorflow',
        'keras',
        'scikit-learn',
        'mtcnn',
        'matplotlib',
    ],
)
