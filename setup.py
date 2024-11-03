from setuptools import find_packages
from setuptools import setup


setup(
    name = 'twor',
    version = '0.0.1',
    description = 'Investigating rotations in 2D and 3D.',
    author = 'Paul Aljabar',
    url = 'git@github.com:evariste/two_rotations.git',
    packages = find_packages(exclude=('test*',)),
)

