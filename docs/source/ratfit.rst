Rational Fitting
================

Since the transfer function of a state-space system
is a rational function, a key ingredient in our methods is
finding a *rational approximation*.
All methods for rational approximation implement the same interface
described below.

.. autoclass:: mor.RationalFit 
   :members:
   
   .. automethod:: __call__


Optimization-based Approaches
-----------------------------

There are a variety of parameterizations of rational approximation
to use with an optimization approach.
Here we describe two: a pole-residue approach and a polynomial based approach. 

.. autoclass:: mor.PartialFractionRationalFit

.. autoclass:: mor.PolynomialBasisRationalFit


Fixed Point Iterations
----------------------

.. autoclass:: mor.SKRationalFit

.. autoclass:: mor.VFRationalFit


Loewner Framework
-----------------

.. autoclass:: mor.AAARationalFit	
