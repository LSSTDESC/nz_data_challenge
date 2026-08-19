================================
Validation figures for taskset 1
================================

The data preparation for taskset 1 including the following steps:

1. Starting with either the Cardinal or Flagship simulation truth information.
2. Rotating the field into an area covered by the LSST survey.
3. Selecting objects with a true :math:`i < 25.5`.
4. Applying photometric smearing. In the Rubin bands this used expected observing
   conditions and depth maps for 1 and 10 years of observing.  In the Roman bands
   this used the expected depths for the medium tier of the High-Latitude wide area survey.
5. Drawing training (100k objects) and test (20k objets) data sets from the catalogs, requiring
   :math:`i < 23.5` for both data sets.   

.. container:: image-gallery

   .. image:: figures/nz_challenge_taskset_1_cardinal_1yr_footprint.png
      :alt: image
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_1_flagship_1yr_footprint.png
      :alt: image
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_1_cardinal_4yr_footprint.png
      :alt: image
      :width: 45.0%
   
   .. image:: figures/nz_challenge_taskset_1_cardinal_4yr_footprint.png
      :alt: image
      :width: 45.0%

   Survey footprints for training (left) and test (right) data.  Within each side
   both 1 cardinal (left) and flagship (right) simulations are shown for
   both 1 year (top) and 10 year (bottom) data sets.


.. container:: image-gallery

   .. image:: figures/nz_challenge_taskset_1_cardinal_1yr_rubin_mags.png
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_1_flagship_1yr_rubin_mags.png
      :width: 45.0%
	      
   .. image:: figures/nz_challenge_taskset_1_flagship_1yr_roman_mags.png
      :width: 45.0%

   .. image:: figures/nz_challenge_taskset_1_flagship_1yr_roman_mags.png
      :width: 45.0%
	       
   Number counts as a function of magnitude for Rubin (top) and Roman (bottom) bands
   for 1 year training (left) and test (right) data sets.



..  LocalWords:  taskset
