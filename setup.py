from setuptools import setup

setup(name='s4hci',
      version='0.1',
      description='Data post-processing for high-contrast imaging with the S4 '
                  'Algorithm.',
      url='https://github.com/markusbonse/s4hci.git',
      author='Markus Bonse',
      author_email='m.bonse@gmx.de',
      license='GPL3',
      install_requires=[
            'seaborn',
            'numpy',
            'tqdm',
            'matplotlib',
            'scikit-learn',
            'scikit-image',
            'torch',
            'astropy',
            'tensorboard',
            'pandas',
            'pynpoint'],
      dependency_links=['git@github.com:markusbonse/GPUTaskManager.git'],
      packages=['s4hci'],
      zip_safe=False)
