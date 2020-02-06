brevitas.core package
=====================

Submodules
----------

brevitas.core.bit\_width module
-------------------------------

.. automodule:: brevitas.core.bit_width
   :members:
   :undoc-members:
   :show-inheritance:

brevitas.core.function\_wrapper module
--------------------------------------

.. automodule:: brevitas.core.function_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

brevitas.core.quant module
--------------------------

.. automodule:: brevitas.core.quant
   :members:
   :undoc-members:
   :show-inheritance:


   .. autoclass:: IdentityQuant()
      :show-inheritance:

      .. method:: forward(x, zero_hw_sentinel)

   .. autoclass:: ClampedBinaryQuant(scaling_impl)
      :show-inheritance:

   .. autoclass:: IntQuant(narrow_range, signed, float_to_int_impl, tensor_clamp_impl)
      :show-inheritance:

   .. autoclass:: PrescaledRestrictIntQuantWithInputBitWidth(narrow_range, signed, tensor_clamp_impl, msb_clamp_bit_width_impl, float_to_int_impl)
      :show-inheritance:

   .. autoclass:: IdentityPrescaledIntQuant()
      :show-inheritance:

brevitas.core.restrict\_val module
----------------------------------

.. automodule:: brevitas.core.restrict_val
   :members:
   :undoc-members:
   :show-inheritance:

brevitas.core.scaling module
----------------------------

.. automodule:: brevitas.core.scaling
   :members:
   :undoc-members:
   :show-inheritance:

brevitas.core.stats module
--------------------------

.. automodule:: brevitas.core.stats
   :members:
   :undoc-members:
   :show-inheritance:


Module contents
---------------

.. automodule:: brevitas.core
   :members:
   :undoc-members:
   :show-inheritance:
