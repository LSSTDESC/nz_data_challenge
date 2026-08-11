Validation figures for taskset 2
================================

The data preparation for taskset 2 including the following steps:

1. Starting with either the Cardinal or Flagship simulation truth information.
2. Rotating the field into an area covered by the LSST survey.
3. Selecting objects with a true :math:`i < 25.5`.
4. Applying photometric smearing. In the Rubin bands this used expected observing
   conditions and depth maps for 1 and 10 years of observing.  In the Roman bands
   this used the expected depths for the medium tier of the High-Latitude wide area survey.
5. Emulating spectroscopic selections for the reference redshift creation.  For this taskset
   we simply applied the spectroscopic selection functions to the entire field.  This will
   overestimate the number of objects in some of the selections corresponding to surveys that
   were only performed in small fields.
6. Drawing a training (100k objects) data set from the objects passing the spectroscopic selections
   and a test data sets (20k objects) from all the objects with and observed :math:`i < 25.5`.

*Note specifically that the training dataset is not representative of the test data set.*
   

.. container:: image-gallery

   .. image:: figures/nz_challenge_taskset_2_cardinal_1yr_footprint.png
      :alt: image
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_2_flagship_1yr_footprint.png
      :alt: image
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_2_cardinal_4yr_footprint.png
      :alt: image
      :width: 45.0%
   
   .. image:: figures nz_challenge_taskset_2_cardinal_4yr_footprint.png
      :alt: image
      :width: 45.0%

   Survey footprints for training (left) and test (right) data.  Within each side
   both 1 cardinal (left) and flagship (right) simulations are shown for
   both 1 year (top) and 10 year (bottom) data sets.


.. container:: image-gallery

   .. image:: figures/nz_challenge_taskset_2_cardinal_1yr_rubin_mags.png
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_2_flagship_1yr_rubin_mags.png
      :width: 45.0%
	      
   .. image:: figures/nz_challenge_taskset_2_flagship_1yr_roman_mags.png
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_2_flagship_1yr_roman_mags.png
      :width: 45.0%
	       
   Number counts as a function of magnitude for Rubin (top) and Roman (bottom) bands
   for 1 year training (left) and test (right) data sets.



..  LocalWords:  taskset
