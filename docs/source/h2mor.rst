==================
H2 Model Reduction
==================

Given model :math:`H\in \mathcal{H}_2`,
the goal of model reduction is to find a simplified model
:math:`H_r\in \mathcal{H}_2` that minimizes the mismatch in the
:math:`\mathcal{H}_2` norm: 

.. math::

   \min_{H_r} \| H - H_r\|_{\mathcal{H}_2} 
   \quad
   \langle F, G \rangle_{\mathcal{H}_2} := 
   \frac{1}{2\pi} \int_{-\infty}^\infty \overline{F(i\omega)} G(i\omega) \mathrm{d} \omega.	




.. autoclass:: mor.H2MOR
   :members:



Projected H2
============

.. autoclass:: mor.ProjectedH2MOR
   :members:


Utility Functions
-----------------
.. autofunction:: mor.ph2.cholesky_inv_norm

.. autofunction:: mor.ph2.subspace_angle_V_M

