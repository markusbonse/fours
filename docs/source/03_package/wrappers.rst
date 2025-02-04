Contrast Curves
===============
**4S** is fully compatible with `applefy <https://applefy.readthedocs.io/en/latest/>`_.
The following wrapper class can be used to compute detection limits using the applefy package.
Since the calculation of the residuals with 4S is quite time-consuming, it is recommended to use
a computer cluster for this task. Once all residuals are computed, the detection limits can be
calculated in a few minutes (restoring residuals, more details in the applefy documentation).

.. automodule:: fours.detection_limits.applefy_wrapper
   :members:
   :special-members: __init__, __call__, _build_4s_noise_model, _create_4s_residuals