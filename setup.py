from setuptools import setup

setup(name='fours',
      version='0.1',
      description='Data post-processing for high-contrast imaging with the 4S '
                  'Algorithm.',
      url='https://github.com/markusbonse/fours.git',
      author='Markus Bonse',
      author_email='m.bonse@gmx.de',
      license='GPL3',
      install_requires=[
            'seaborn',
            'numpy',
            'tqdm',
            'photutils',
            'matplotlib',
            'scikit-learn',
            'scikit-image',
            'torch',
            'h5py',
            'astropy',
            'tensorboard',
            'pandas'],
      packages=['fours'],
      zip_safe=False)
