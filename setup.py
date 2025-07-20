from setuptools import setup, find_packages
import os

setup(
    name='echovit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    'torch',
    'torchvision',
    'scikit-learn',
    'tqdm',
    'numpy',
    'pandas',
    'opencv-python',
    'vidaug',
    'scikit-image',
    'click',
    'matplotlib'
    ],
 
    author='eozkaynar',
    description='EchoVit: EF Estimation from EchoNet-Dynamic using ViViT',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/eozkaynar/EchoViT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)