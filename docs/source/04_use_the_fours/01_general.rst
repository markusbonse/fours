Use the 4S
==========
This is the code used to create the plots of the
**4S (Signal Safe Speckle Subtraction)** paper
`(Bonse et al. 2024) <../05_citation.rst>`_. In order to reproduce our results
please `download <02_paper_data.rst>`_ the required data from Zenodo.


.. toctree::
   :hidden:

   02_paper_data.md

.. nbgallery::

   paper_experiments/02_plot_pca_residuals.ipynb
   paper_experiments/03_pca_saliency_map.ipynb
   paper_experiments/05_throughput.ipynb
   paper_experiments/07_4s_saliency_maps.ipynb
   paper_experiments/08_plot_contrast_curves_and_residuals.ipynb
   paper_experiments/09_plot_relative_gain.ipynb
   paper_experiments/10_AF_Lep_S4_PCA.ipynb
   paper_experiments/10_AF_Lep_MCMC.ipynb
   paper_experiments/11_AF_parameters.ipynb
   paper_experiments/0a_residual_noise_distributions.ipynb
   paper_experiments/1a_residual_gallery_af_lep.ipynb
   paper_experiments/2a_residual_gallery_fake_planets.ipynb
   paper_experiments/3a_masked_saliency_maps.ipynb
   paper_experiments/4a_AF_Lep_4S_fake_planets.ipynb

In addition to the code used for the plots we used some ``scripts`` to compute
the fake planet residuals. You can find the code in our github repository.

.. nbgallery::

   paper_experiments/fake_planet_experiments/01_create_dataset_config_files.ipynb
   paper_experiments/fake_planet_experiments/02_compute_contrast_grids.ipynb
   paper_experiments/x_methods_explained_material.ipynb
