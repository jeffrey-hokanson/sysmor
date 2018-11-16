System Classes
==============

We define a variety of representations of linear time invariant systems.
All share a similar API inherited from the :code:`LTISystem` class.
Note that we can think of these LTI systems as forming a vector space
and so we imbue the class with the ability to add and subtract systems
as well as perform scalar multiplication.
 

.. autoclass:: mor.LTISystem
   :members:
   
   .. automethod:: __add__
   
   .. automethod:: __sub__
   
   .. automethod:: __mul__


Transfer Function System
------------------------

.. autoclass:: mor.TransferSystem


State-Space System
------------------

.. autoclass:: mor.StateSpaceSystem

.. autoclass:: mor.SparseStateSpaceSystem
  

 
