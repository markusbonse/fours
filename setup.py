from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fours',
      version='0.1',
      description='Data post-processing for high-contrast imaging with the 4S '
                  'Algorithm.',
      url='https://github.com/markusbonse/fours.git',
      author='Markus Bonse',
      author_email="mbonse@phys.ethz.ch",
      license="MIT License",
      long_description=long_description,
      long_description_content_type="text/markdown",
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
      extras_require={
            "dev": ["furo>=2022.12.7",
                    "sphinx_rtd_theme==1.1.1",
                    "sphinx>=2.1,<6",
                    "myst-parser~=0.18.1",
                    "nbsphinx>=0.8.9",
                    "sphinx-copybutton~=0.5.1",
                    "sphinx-gallery<=0.10",
                    "twine~=4.0.2",
                    # needed for syntax highlighting in jupyter notebooks
                    "IPython~=8.8.0",
                    "ipywidgets~=8.0.4",
                    # spell checking in jupyter notebooks
                    "jupyter_contrib_nbextensions~=0.7.0",
                    "sphinx-autodoc-typehints>1.6"],
            "plotting": ["seaborn~=0.12.1",
                         "matplotlib>=3.4.3",
                         "bokeh>=3.0.3"],
      },
      packages=find_packages(include=['fours', 'fours.*']),
      zip_safe=False
)
