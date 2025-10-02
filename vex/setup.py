import sys
import setuptools

if sys.version_info.major == 2:
  setuptools.setup(name='vex', version='2.0', description='VEX Parser',
        author='Mark Kettenis', author_email='kettenis@jive.nl',
        install_requires=['ply'],
        py_modules=['vex', 'MultiDict'])
else:
  # In python 3 we use the MultiDict from pypi
  setuptools.setup(name='vex', version='2.0', description='VEX Parser',
        author='Mark Kettenis', author_email='kettenis@jive.nl',
        install_requires=['ply', 'multidict'],
        py_modules=['vex'])

