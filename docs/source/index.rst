.. H2 Model Reduction documentation master file, created by
   sphinx-quickstart on Thu Oct 25 10:48:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to System-theoretic Model Order Reduction's documentation!
=================================================


System-theoretic model reduction seeks to build reduced order models based on metrics associated with the input-output map.
The simplest case appears in *linear, time-invariant* model reduction. 
One common example is a state-space system relating input :math:`\mathbf{u}(t)` to output :math:`\mathbf{y}(t)` via the dynamical system 

..math::

  \mathbf{x}'(t) &= \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) \\
  \mathbf{y}(t) &= \mathbf{C} \mathbf{x}(t).
  
The goal of the goal of *model reduction* is to construct a reduced order model mimicing
this input-output map.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ratfit
   system
   h2mor
   utils 
   demos



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
