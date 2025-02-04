PSF Subtraction
===============
Overview about all models used in 4S. The classical way to use 4S is based on
the Class explained in **FourS**. All other
models are used internally and should usually not be used directly.

FourS
-----
.. automodule:: fours.models.psf_subtraction
   :members:
   :special-members: __init__, _setup_work_dir, _logg_loss_values, _logg_residuals, _create_tensorboard_logger

Linear Noise Model
------------------
.. automodule:: fours.models.noise
   :members:
   :special-members: __init__, _apply

Image Rotation
--------------
This class makes it possible to back-propagate through the ADI rotation
using Spatial Transformer Networks. It might be useful for some applications.

.. automodule:: fours.models.rotation
   :members:
   :special-members: __init__, _build_grid_from_angles

