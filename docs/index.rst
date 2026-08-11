****************************************************************
Welcome to the Photometric Redshift Ensemble (NZ) Data Challenge
****************************************************************

The Dark Energy Science Collaboration (DESC) invites researchers, data
scientists, and astronomers to participate in the Photometric Redshift
Ensemble (NZ) Data Challenge, a collaborative effort to advance methods for
estimating the distribution of redshifts for ensembles of distant galaxies.
Photometric redshifts, derived from multi-band brightness measurements, are essential for
cosmological surveys like the Legacy Survey of Space and
Time (LSST), enabling us to map the universe's structure and probe the
nature of dark energy. This challenge provides a unique opportunity to
test and benchmark algorithms on realistic simulated data, compare
approaches across diverse methodologies—from template fitting to
machine learning—and help shape the tools that will unlock discoveries
from next-generation sky surveys.

The challenge is framed as a series of sets of PZ estimations tasks
using increasingly realistic data.

`Set up the pz_data_challenge package and download challenge data <installation_and_setup_>`_

`Information about the input data <challenge_input_data_>`_

`How to submit an entry to the challenge <challenge_submissions_>`_

`Description of challenge tasks <tasks_>`_

`Assessment Metrics <metrics_>`_

`Details about challenge data preparation <challenge_data_prep_>`_

`Write of up NZ Data Challenge documentation <https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/nz_challenge.pdf>`_


======================
Background Information
======================

.. _intro:

Introduction
============

Redshift inference is a key element of many DESC science goals, and
redshift uncertainty is one of the leading contributors to overall
uncertainty on cosmological models from imaging survey data. Precursor
surveys took a variety of approaches to this problem, accounting for
differences in underlying data as well as modeling approaches. In all
cases, redshift uncertainty was significantly larger than the DESC
Science Requirements listed in the LSST DESC Science Requirements
Document.

This state of the art motivates a data challenge to characterize and
improve existing methods, as well as to provide infrastructure for the
development of improved methods. Overall, this requires generating
uniform input catalogs to use and infrastructure for comparing output
redshift posteriors for ensembles to each other and to simulated truth
catalogs.

.. _redshift_basics:

Photometric redshift basics
===========================

Photometric redshift estimation involves taking a catalog of galaxies
for which we have observations in several different filters and have
measured the brightness of the galaxies in those bands, and using that
information to estimate the redshift of the galaxies. For LSST we expect
to have measurements in 6 bands: ’u’, ’g’, ’r’, ’i’, ’z’, and ’y’,
covering a wavelength range from approximately 320 to 1600 nanometers.
For the Roman space telescope, this will extend from about 500 to 2300
nanometers.

Much of the information used to estimate photometric redshifts derives
from the ’Balmer break’ present in the rest frame of many spectra at 400
nm. As the break crosses into different optical filters with increasing
redshift, the differences in magnitudes between filters carry
information about the redshift;

.. container:: figure*

   .. image:: figures/static_balmer.png
      :alt: image
      :width: 80.0%

.. container:: figure*

   |image| |image1|

This overly simple picture is complicated somewhat by the fact that
different galaxies have different intrinsic spectra and colors:

.. container:: figure*

   .. image:: figures/gr_vs_sz_sidebyside.jpg
      :alt: image
      :width: 80.0%

This is further complicated by the fact that reference redshifts,
typically obtained by spectroscopy, slitless spectroscopy (i.e., GRISM
measurements), or narrowband photometric measurements, are not a
representative sample, as they are much easier to obtain for brighter
objects. Depending on the method used to obtain the reference redshifts,
they are also susceptible to errors such as confusing different spectral
lines or confusion of blended objects. Some of the tasks in this data
challenge encourage participants to try to address these complications.

.. _challenge_format:

Challenge Format
================

The NZ data challenge comprises a series of sets of tasks for
participants. Submissions will be evaluated to determine how ready
various algorithms are to be used for cutting-edge analysis based on how
well they perform on the various tasks. Readiness will be evaluated on a
few different fronts: 1) Does the algorithm meet performance
requirements? 2) Is it robust, flexible, and relatively easy to use on
different datasets? 3) Is it scalable up to the scales we will need to
use it at?

This document and the associated web pages describe the data being
provided to participants, the tasks they will be asked to perform, the
expected format for submission and the metrics by which the algorithm
readiness will be evaluated.

Scope and Timeline
------------------

The data challenge includes two major parts, with a set of tasks
emulating increasingly realistic scenarios in each part. The first part,
the PZ data challenge, assesssing per-object :math:`p(z)` estimation,
was launched in April, 2026, and will concluded in September, 2026. The
second part, tomography and :math:`n(z)` estimation, will focus on
assigning objects to tomographic bins and estimating the distribution of
redshifts in each bin, and will launch in September 2026.

Preliminary results will be released in January, 2027, with a technical
note summarizing those results to follow shortly thereafter and a
comprehensive journal publication to follow later.

Installing and setting up the ``nz_data_challenge`` package
-----------------------------------------------------------

.. _installation_and_setup:


The ``nz_data_challenge`` package will provide participants with tools
to access data, set up submissions, estimate performance metrics and
format submissions. This can be set up with a few small variants on the
standard ``GitHub`` package setup procedure. Before starting you should
pick a name for your submission, e.g., “example”.

::

   # Create a conda environment
   conda create --name nzdc python=3.13

   # Clone the nz_data_challenge repository (or your fork of the repository)
   git clone git@github.com:LSSTDESC/nz_data_challenge.git
   # or git clone https://github.com/LSSTDESC/nz_data_challenge.git

   # Go into the directory
   cd nz_data_challenge

   # Install the code in "editable" mode
   pip install -e ".[dev]"

   # Use the provided script to set up your submission.
   # Here you should provide the name of your submission
   python scripts/prepare_submission.py <submission_name>

This final step will copy the input data files to
``nz_data_challenge/public``, and set up the three files you will need
to submit your entry.

The notebooks in the ``nz_data_challenge/nb`` area give examples of how
to access the data and create some of the diagnostic plots that were
used to validate the data.

Submission mechanism
--------------------

Submission will take the form of pull request in the
``nz_data_challenge`` repository. Detailed instructions on how to submit
an entry are provided in `challenge_submissions`_ of this
document.

.. _challenge_input_data:

Challenge Input Data
====================

The preparation of the challenge data is described in the appendices.
The data are available as a ``tar`` archive that is downloaded and
unpacked as part of the ``nz_data_challenge`` setup procedure.

Each task set in the data challenge has an associated set of files.
Typically these will be a collection of training files that contain
photometric data and reference redshifts, and a second set of files that
contain photometric data but do not include redshifts. Each task set
will involve estimating something about the redshifts or redshift
distributions in the test files.

Typically there will be several training and test files for a particular
task set, covering different scenarios and using different input
simulations.

Input data format
-----------------

The input data for the challenge are presented in HDF5 files. The naming
convention for the files is
``{challenge}_{taskset}_{simulation}_{label}_{scenario}.hdf5``. The
meanings of the various fields are described in the next table.
The columns in the files are described in table after.
We note that we use ``np.nan`` to in the
magnitude columns to signify non-detections.

.. container::
   :name: tab:file_fields

   .. table:: Fields in the input file names.

      ========== ==========================================================
      Field      Description
      ========== ==========================================================
      challenge  Challenge associated with file (“nz_challenge”)
      taskset    Task set associated with file (e.g., “taskset_1”)
      simulation Simulation used to produce file (“cardinal” or “flagship”)
      label      File label (e.g., “ddf_03”, “wfd)
      scenario   Data scenario (e.g., “1yr”, “4yr”’)
      ========== ==========================================================

.. container::
   :name: tab:columns

   .. table:: Contents of input files.

      ==================== =====================================
      Column               Description
      ==================== =====================================
      redshift             True redshift (training files only)
      ra                   Right ascension (training files only)
      dec                  Declination (training files only)
      object_id            Unique object ID
      mag_{band}_lsst      Magnitude in LSST {band}
      mag_{band}_lsst_err  Magnitude uncertainty in LSST {band}
      mag_{band}_roman     Magnitude in Roman {band}
      mag_{band}_roman_err Magnitude uncertainty in Roman {band}
      ==================== =====================================

We note that the ``table-io`` package  :raw-latex:`\cite{tables_io}`
installed with ``nz_data_challenge`` provided a command line interface
to convert files from ``hdf5`` format to other formats such as
``parquet`` tables or ``pandas`` data frames.

::

   # convert a hdf5 file to pandas dataframe in a parquet file
   tables-io convert
     --input public/nz_challenge_taskset_1_flagship_4yr_ddf_03.hdf5
     --output public/nz_challenge_taskset_1_flagship_4yr_ddf_03.pq

.. _challenge_submissions:

Challenge Submissions
=====================

Challenge subtask types
-----------------------

The challenge is organized as a series of sets of tasks using
increasingly realistic representations of the data. In general, each set
of tasks includes 3 subtasks.

#. Providing tomographic bin assignments for all objects, and estimate
   ensemble :math:`n(z)` distributions for a set of different scenarios
   and provide the estimates in a specified format.

#. Provide trained models for the different scenarios and a Python
   function that can be used to generate the estimates from subtask 1 on
   an arbitrary dataset. We will not actually run these ourselves as
   part of the challenge, but we do ask that you providing timing
   estimates for how long it took you to run this.

#. Provide a Python function that can be used to generate the models and
   estimates from subtasks 1 and 2 on arbitrary datasets. We will not
   actually run these ourselves as part of the challenge, but we do ask
   that you providing timing estimates for how long it took you to run
   this.

The :math:`n(z)` estimates in subtask 1, the trained models in subtask 2
and and the timing estiamtes should be provided in a compressed ``tar``
file. Templates and instructions for the Python functions needed for
subtasks 2 and 3 will be provided and are described below.

Data format for per-object :math:`p(z)` estimates
-------------------------------------------------

The :math:`p(z)` estimates should be submitted in ``qp`` format, which
allows users to specify a complete :math:`p(z)` distribution for each
object, as well as summary statistics for each object.

The ``qp`` package :raw-latex:`\cite{QP}` supports several different
representations of :math:`p(z)`, such as different functional forms as
well as interpolated grids, histograms, and others.

For users unfamiliar with ``qp``, we highly recommend representing the
:math:`n(z)` as either an histogram grid or a Gaussian mixture model.

::

   # Interpolated grid
   import qp
   import numpy as np
   # Define the histogram bins. Note that we put all the
   # n(z) on the same binning
   bins = np.array([0,0.5,1,1.5,2])
   # Define the y-values. Note we provide n_grid_points-1 x n_objects 
   # values, as we need to provide a y-value at histogram bin 
   # for each object.
   yvals = np.array(
    [
      [0.01,0.2,0.3,0.49],
      [0.1,0.3,0.5,0.1]
    ]
   )
   ensemble = qp.hist.create_ensemble(bins,yvals)
   ensemble.write_to(<output_filename.hdf5>)

::

   # Mixture model
   import qp
   import numpy as np
   # Define the means, standard deviations, and weights.
   # These should each have shape n_objects, n_components.
   # In this case we are defining 3 objects with 2-Gaussian 
   # representations.
   # For each object the weights should sum to 1, or they
   # will be normalized.
   means = [[0.3, 0.4], [0.5, 0.5], [0.6, 0.8]]
   stds = [[0.2, 0.4], [0.1, 0.3], [0.05, 0.3]]
   weights = [[0.8, 0.2], [0.7, 0.3], [0.8, 0.2]]
   ensemble = qp.mixmod.create_ensemble(means=means,stds=stds,weights=weights)
   ensemble.write_to(<output_filename.hdf5>)

The submission files should use the same file name conventions defined
in `<tab:file_fields>`_. The labels will typically be
``nz_estimate`` (for the :math:`n(z)` estimates), ``bhat_estimate`` (for
the bin assignements) or ``timing`` and will be specified in the
descriptions of the various tasks, e.g.,

::

   \texttt{nz\_challenge\_taskset\_1\_cardinal\-\_nz\_estimate\_1yr.hdf5}
     or
   \texttt{nz\_challenge\_taskset\_1\_cardinal\_bhat\_1yr.pkl}.

All of these files should then be joined into a ``tar`` file, which
should then be placed somewhere it can be download. The URL for the
``tar`` should be specified in ``tests/test_{submission}.py``

::

   SUBMISSION_NAME = "example"
   SUBMISSION_URL = "https://your.institution.edu/submit_example.tgz"

Format for estimation-only Python functions and trained models
--------------------------------------------------------------

For the second subtask, submissions should provide trained models and
implement a function to run estimation using those trained models on the
test files provided for each task set. The function will look something
like this:

::

   def run_taskset_1_estimation_only(
       model_file: str | Path,
       test_file: str | Path,
       output_file: str | Path,
   ) -> None:
       # do stuff and write p(z) estimates to "output_file"

or

::

   def run_taskset_2_estimation_only(
       model_file: str | Path,
       test_file: str | Path,
       output_file: str | Path,
   ) -> None:
       # do stuff and write p(z) estimates to "output_file"

Templates for these functions are provided in the file
``tests/test_{submission}.py`` created as part of the setup.

Format for training and estimation Python functions
---------------------------------------------------

For the third subtask, submissions should implement a function to train
models and run estimation using those trained models on the training and
test files provided for each task set. The function will look something
like this:

::

   def run_taskset_1_training_and_estimation(
       train_file: str | Path,
       test_file: str | Path,
       output_file: str | Path,
   ) -> None:
       # train a model using the "train_file" and make p(z) estimates 
       # and write them to "output_file"

or

::

   def run_taskset_2_training_and_estimation(
       train_file: str | Path,
       test_file: str | Path,
       output_file: str | Path,
   ) -> None:
       # train a model using the "train_file" and make p(z) estimates
       # and write them to "output_file"

Templates for these functions are provided in the file
``tests/test_{submission}.py`` created as part of the setup.

.. _submission-mechanism-1:

Submission process
------------------

Submissions will take the form of a pull request on the
``bz_data_challenge`` repository and will include:

#. A file ``tests/test_{submission}.py`` that includes the URL from
   which the compressed ``tar`` file should be downloaded as well as the
   Python functions for subtasks 2 and 3. When created this will contain
   empty placeholder functions that will need to be implemented.

#. A file ``requirements_{submission}.txt`` that should be modified to
   include ``pip`` package names of any packages that need to be
   installed in order to run the functions in subtasks 2 and 3.

#. A file ``.github/workflows/submit_{submission}.yaml`` to run the
   submission validation in a GitHub action. This should not need to be
   modified unless the prerequisites installation requires more than
   just ``pip`` installing packages.

All three of these files are created by the
``scripts/prepare_submission.py`` script.

You will need modify the ``tests/test_{submission}.py`` to give the
location of the ``tar`` file containing the NZ estimates, bin
assignemnts, timing estimates and trained models, and to implement the
required functions.

See ``https://github.com/LSSTDESC/nz_data_challenge/pull/6`` for an
example of a submission.

Submission validation
---------------------

The wrapping functions provided in the ``tests/test_{submission}.py``
file implement a number of checks on the data. Specifically, for each
expected file they check that:

#. the :math:`n(z)` file exists;

#. the :math:`n(z)` file contains a valid ``qp`` ensemble with the
   expected number of :math:`n(z)` pdfs;

#. the ``qp`` ensemble includes ancillary data;

#. the ancillary data includes a “n_objects” column with numbers of
   objects assigned to each bin;

#. the :math:`bhat` file exists;

#. the :math:`bhat` file contains a two-column table with “bhat”
   assignement and “object_id” for each object.

#. the object_ids in the submission file match the associated test file.

If any of these checks fail, the GitHub action triggered by the
submission will fail and report the cause of the failure. **Note that
the github actions occasionally fail to download the data files. If this
happens simply rerunning the action typically succeeds.**

The easiest way to test that you have correctly implemented the required
functions is simply to run these commands.

::

   # Make sure that you have installed any packages you need
   pip install -r requirement_{submission_name}.txt

   # Run the functions you have provided as unit tests
   py.test tests/test_{submission_name}.py

if this succeeds, you can use a provided script to help you open the
pull request for your submission.

::

   # run the submission helper script.
   python scripts/submit.py {submission_name}

Note that the help script only prints the required commands, it does not
run them. In short the command are:

::

   # Check status of your local git clone by running git status, and make
   # sure that you are on the branch submit/{submission_name} and do not
   # have any files added or modified
   git status

   # Add your files to git
   git add .github/workflows/submit_example.yaml
     requirements_example.txt
     tests/test_example.py

   # Commit your files to your branch: 
   git commit -m "Submitting {submission_name}"
     .github/workflows/submit_{submission_name}.yaml
     requirements_{submission_name}.txt
     tests/test_{submission_name}.py

   # Push your commit
   git push --set-upstream origin submit/{submission_name}

   # Pushing to git should give you a URL that you can visit to create a
   # pull request, for example:
   #   https://github.com/LSSTDESC/nz_data_challenge/pull/new/submit/example
   # Visit that URL and create a pull request, then add the 'submission'
   # label to the PR.
   # Finally, make sure that the github action validating your submission
   # succeeds and fix any issues.

Submission aids
---------------

A few scripts are provided to help you.

-  ``scripts/download_public.py``: downloads and unpacks the public
   data.

-  ``scripts/prepare_submission.py``: sets up your area for a
   submission, creates the needed files from templates and downloads the
   public data, and suggests that you create a branch for you
   submission.

-  ``scripts/remove_submission_files.py``: removes the submission files
   if you need to start over.

-  ``scripts/run_metrics.py``: run perfomance metrics on files in a
   submission you have created.

-  ``py.test tests/test_{submission_name}.py``: validates all the parts
   of your submission, checking that you have created all the required
   files and that they are properly formatted.

.. _metrics:

Metrics and Assessment Criteria
===============================

We will use a number of different metrics to assess the performance of
the submitted algorithms. Many of these metrics, as well as the
motivations behind them, are defined and discussed in
Ref. :raw-latex:`\cite{therailteam2025}`.

Metrics for bin assignment
--------------------------

Performance on bin-assignement estimates, i.e., how well the algorithm
assigns objects to the desired tomographic bin are based on the
confusion matrix. I.e., the matrix :math:`N_{\rm true, \rm assigned}`.
In the ideal case this matrix would be diagonal, i.e., all the objects
would be assigned to the true bin.

We then use this to construct the following metrics:

-  ``Accuracy`` is simply the median of :math:`\Delta_i`.

-  ``Balanced Accuracy`` is the fraction of the :math:`\Delta`
   distribution outside of
   :math:`[0, {\rm max}(0.06, 3\sigma_{\rm iqr})]`.

-  ``Cohen’s Kappa`` .

-  ``Mutual Information`` .

-  ``Log Loss From Labels`` .

Metrics for ensemble :math:`n(z)` distributions
-----------------------------------------------

We will also assess the algorithm’s ability to provide a precise and
accurate estimate of the posterior distribution, :math:`n(z)`, for each
ensemble following metrics.

-  ``Total information Loss``.

-  ``Wasserstein Distance``.

-  ``RMS of``\ :math:`\delta_{\rm true} - \delta_{\rm est}`.

-  ``RMS of``\ :math:`\sigma_{\rm true} - \sigma_{\rm est}`.


Metrics for computational usability and performance
---------------------------------------------------

We will assess relevant aspects of the computational performance that
will affect usability and scaling.

-  **Ease of use**: We will assess whether the algorithm is easy to
   install and can be run on the different task sets without needing
   excessively complicated additional configuration files.

-  **Training time**: How quickly the algorithm trains models, and how
   this scales with the training sample size. Here we mainly want to
   ensure that the training time will not dominate the iteration cycle.
   Taking a couple of hours to train on 1M objects is fine; taking days
   to do so would be problematic.

-  **Model size**: How large the trained model files are, and how this
   scales with the training sample size. Again, we mainly want to ensure
   that the model size will not tax our resources. If the model files
   are an order of magnitude larger than the input data files, we might
   worry.

-  **Estimation time**: How quickly the algorithm estimates ensemble
   distributions. This will determine the use cases for which we might
   use the algorithm. We can run an algorithm that takes a few ms per
   object on all of the billions of galaxies we will have in the final
   LSST sample; for an algorithm that takes a few seconds per object, we
   would probably be constrained to only run it on much smaller
   particular datasets for specific science cases, such as samples of
   supernovae or strongly lensed objects.

.. _tasks:

Challenge Tasks related to :math:`n(z)` estimation
==================================================


Task set 1: Assign objects to fixed tomographic bins and estimate ensemble PDFs using mostly representative training samples
----------------------------------------------------------------------------------------------------------------------------

The first, simplest task is to estimate redshifts using representative
training samples. I.e., the training samples are drawn from the same
distributions as the test samples. For this task set we did not use any
of the spectroscopic selection emulation, but simply applied a uniform
magnitude cut of :math:`i < 23` in selecting objects for both the
training and test samples.

The four
``nz_challenge_taskset_1_{simulation}_training_{scenario}.hdf5`` files
are the training sets for the “Flagship” and “Cardinal” simulations,
emulating 1 year and 4 years of LSST data under the expected observing
strategy and conditions. These files have true redshifts to serve as
labels.

The corresponding
``nz_challenge_taskset_1_{simulation}_test_{scenario}.hdf5`` files were
drawn from the same distributions. The true redshifts have been removed
from these files. The task is to assign :math:`p(z)` estimates for all
the objects in these 4 test files.

The subtasks in this task set are:

#. Estimate :math:`p(z)` for each object in each of the test files and
   provide the estimates in a downloadable ``tar`` file.

#. Provide pre-trained models appropriate to each of the training files
   and implement a Python function (``run_taskset_1_estimation_only``)
   to use those pre-trained models to estimate :math:`p(z)` for each
   object in the associated test files.

#. Implement a Python function
   (``run_taskset_1_training_and_estimation``) to train a model for each
   training file and use that model to estimate :math:`p(z)` for each
   object in the associated test files.

Task set 2: Assign objects to fixed tomographic bins and estimate ensemble PDFs using non-representative samples
----------------------------------------------------------------------------------------------------------------

The second, slightly more challenging task is to estimate redshifts
using non-representative training samples. I.e., the training samples
are not drawn from the same distributions as the test samples. For this
task set we applied the spectroscopic selection emulation for the
training set, but retained all the objects down to :math:`i < 25.4` in
the test set. Accordingly, the training set will not be representative
of the fainter objects in the test set. This reflects that spectroscopic
redshifts are typically significantly more difficult to obtain than
photometry.

The four
``nz_challenge_taskset_2_{simulation}_training_{scenario}.hdf5`` files
are the training sets for the “Flagship” and “Cardinal” simulations,
emulating 1 year and 10 years of LSST data under the expected observing
strategy and conditions and with spectroscopic selections emulated.

The corresponding
``nz_challenge_taskset_2_{simulation}_test_{scenario}.hdf5`` files were
drawn from the distributions of all objects down to :math:`i <
25.4`, and the true redshifts have been removed from these files. The
task is to assign :math:`p(z)` estimates for all the objects in these 4
test files.

The subtasks in this task set are:

#. Estimate :math:`p(z)` for each object in each of the test files and
   provide the estimates in a downloadable ``tar`` file.

#. Provide pre-trained models appropriate to each of the training files
   and implement a Python function (``run_taskset_2_estimation_only``)
   to use those pre-trained models to estimate :math:`p(z)` for each
   object in the associated test files.

#. Implement a Python function
   (``run_taskset_2_training_and_estimation``) to train a model for each
   training file and use that model to estimate :math:`p(z)` for each
   object in the associated test files.

Task set 3: Assign objects to arbitrary tomographic bins and estimate ensemble PDFs using non-representative samples
--------------------------------------------------------------------------------------------------------------------

The third task is to estimate redshifts using non-representative
training samples that more accurately emulate real reference redshift
samples. This include some narrowband photometric redshifts from the
COSMOS2020 dataset that go deeper than most spectroscopic redshifts, but
have more scatter and more significant levels of catastrophic outliers.

As before, the four
``nz_challenge_taskset_3_{simulation}_training_{scenario}.hdf5`` files
are the training sets for the “Flagship” and “Cardinal” simulations,
emulating 1 year and 10 years of LSST data under the expected observing
strategy and conditions and with spectroscopic selections emulated.
These files include flags showing which spectroscopic survey particular
objects would be associated with, and for the COSMOS2020 field, also
include a column “redshift_manyband” giving the narrow-band photometric
redshifts in addition to the spectroscopic redshifts. The point of this
taskset is to find a way to optimally use the additional information
from the COSMOS2020 field.

The corresponding
``nz_challenge_taskset_3_{simulation}_test_{scenario}.hdf5`` files were
from the distributions of all objects down to :math:`i< 25.4`, and both
the true redshifts and the narrow-band photometric redshifts have been
removed from these files. The task is to assign :math:`p(z)` estimates
for all the objects in these 4 test files.

The subtasks in this task set are:

#. Estimate :math:`p(z)` for each object in each of the test files and
   provide the estimates in a downloadable ``tar`` file.

#. Provide pre-trained models appropriate to each of the training files
   and implement a Python function (``run_taskset_3_estimation_only``)
   to use those pre-trained models to estimate :math:`p(z)` for each
   object in the associated test files.

#. Implement a Python function
   (``run_taskset_1_training_and_estimation``) to train a model for each
   training file and use that model to estimate :math:`p(z)` for each
   object in the associated test files.


.. _input_sims:

Input simulations
=================

The challenge employs simulated galaxy catalogs derived from two
complementary N-body cosmological simulations: the Cardinal simulations
and the Flagship simulation. These synthetic datasets provide a
controlled environment where the true redshifts are known by
construction, enabling rigorous validation of photometric redshift
algorithms and systematic assessment of their performance
characteristics.

The Cardinal simulations comprise a suite of high-resolution N-body
simulations specifically designed to explore the sensitivity of
cosmological observables to variations in fundamental cosmological
parameters. The simulations employ state-of-the-art semi-analytic models
to populate dark matter halos with galaxies, incorporating realistic
prescriptions for star formation, dust attenuation, and spectral energy
distribution modeling.

The Flagship simulation represents a single, ultra-large cosmological
simulation run with fiducial cosmological parameters consistent with
current observational constraints. With a volume exceeding several cubic
gigaparsecs, the Flagship provides statistical power to probe rare
objects and the high-mass end of the galaxy population. Its primary
purpose in the photometric redshift challenge is to provide a realistic
mock catalog that captures the full complexity of galaxy populations
across cosmic time, including correlations between galaxy properties,
environmental dependencies, and the intricate relationships between
spectral features and redshift.

Together, these complementary simulation suites enable challenge
participants to test both the accuracy and the robustness of their
photometric redshift estimation methods under realistic observational
conditions.

.. _emulating_observations:

Emulating observational effects
===============================

To bridge the gap between the idealized simulation outputs and realistic
survey observations, we employ the RAIL (Redshift Assessment
Infrastructure Layers) software package to emulate observational
effects. RAIL provides a modular framework for injecting realistic
photometric uncertainties, applying survey-specific selection functions,
and simulating the measurement errors characteristic of modern
large-scale imaging surveys. This processing ensures that the simulated
galaxy catalogs reflect the complexities of actual observations,
including magnitude-dependent photometric scatter, incomplete sky
coverage, and the effects of source blending in crowded fields, thereby
providing a more stringent and realistic testbed for photometric
redshift estimation algorithms.

Photometric Smearing
--------------------

Central to our observational emulation is RAIL’s wrapping of the
photometric error module, photErr, which we have extended and wrapped to
account for realistic observing strategies and time-dependent survey
conditions. The standard photErr module provides basic photometric error
modeling based on magnitude-dependent noise characteristics, but our
enhanced version incorporates additional complexity including spatially
varying depth maps. This wrapper accesses detailed operational
simulation outputs that emulate the expected LSST survey strategy.

Our photErr implementation computes photometric uncertainties by
combining the intrinsic Poisson noise from source photons with realistic
models of sky background, readout noise, and other systematic
contributions. For each simulated galaxy, we use the expected coadded
depth to derive final photometric error estimates. This approach
captures the heterogeneous nature of survey depth across the footprint,
where some regions benefit from numerous high-quality exposures while
others may be observed only during poor conditions. The resulting
photometric uncertainties vary realistically with position on the sky,
band-dependent limiting magnitudes, and local observing history,
providing challenge participants with mock catalogs whose noise
properties more closely match those expected from the actual survey.

Spectroscopic and narrowband photometric redshift selection
-----------------------------------------------------------

RAIL can emulate the selection functions of several different
spectroscopic redshift surveys, including
VVDSf02 :raw-latex:`\cite{2008yCat.2286....0M}`,
zCOSMOS :raw-latex:`\cite{2007ApJS..172...70L}`,
DEEP2 :raw-latex:`\cite{2013ApJS..208....5N}`, and the DESI BGS, ELG,
and LRG  :raw-latex:`\cite{desicollaboration2025datarelease1dark}`
samples.

We can also use RAIL to emulate narrow-band photometric surveys and
include small amounts of mislabeled reference redshifts. The performance
of the narrow-band photometric redshifts is shown here.

.. container:: figure*

   |image2| |image3|

Emulating unrecognized blending
-------------------------------

In the files for taskset 4 we emulate the effect of unrecognized
blending, i.e., two or more objects being detected as a single object.
Our blending algorithm is relatively simple: we apply a
“friends-of-friends” matching algorithm with a 1.0 arcsecond linking
length replace all groups with a single object with the summed fluxes in
each of the bands.

.. container:: figure*

   |image4| |image5|

   |image6| |image7|

.. _challenge_data_prep:

Preparing Training, Test, and Reserved Datasets
===============================================

All of the data preparation was performed using the ``rail_projects``
and ``rail_package_config`` packages for bookkeeping and
reproducibility. The scripts listed in `<prep_scripts>`_
comprise the entire production pipeline from the simulation truth
catalogs to the files released with the challenge.

.. container::
   :name: tab:prep_scripts

   .. table:: Scripts used in data preparation.

      +-----------------+------------------------+------------------------+
      | Script          | Command Run            | Purpose                |
      +=================+========================+========================+
      | do_00_reduce    | rail-project reduce    | Reduce input truth     |
      |                 |                        | catalogs               |
      +-----------------+------------------------+------------------------+
      |                 |                        | (mag. cut and drop     |
      |                 |                        | columns)               |
      +-----------------+------------------------+------------------------+
      | do_01_build     | rail-project build     | Build configurations   |
      |                 |                        | to run                 |
      +-----------------+------------------------+------------------------+
      |                 |                        | truth-to-observed      |
      |                 |                        | pipeline               |
      +-----------------+------------------------+------------------------+
      | do_02_t2o       | rail-project run       | Run truth-to-observed  |
      |                 | truth-to-observed      |                        |
      +-----------------+------------------------+------------------------+
      |                 |                        | pipelines to make      |
      |                 |                        | degraded catalogs      |
      +-----------------+------------------------+------------------------+
      | do_03_merge     | rail-project merge     | Combine spectroscopic  |
      |                 |                        | selections             |
      +-----------------+------------------------+------------------------+
      | do_04_subselect | rail-project subsample | Make train/test files  |
      |                 |                        | from catalogs          |
      +-----------------+------------------------+------------------------+

.. |image| image:: figures/color_color_redshift_taskset_1_cardinal_10yr.png
   :width: 45.0%
.. |image1| image:: figures/color_color_redshift_taskset_1_flagship_10yr.png
   :width: 45.0%
.. |image2| image:: figures/foutlier_vs_mag_i.jpg
   :width: 45.0%
.. |image3| image:: figures/sigma_nmad_vs_mag_i.jpg
   :width: 45.0%
.. |image4| image:: figures/n_objects_in_blend.png
   :width: 45.0%
.. |image5| image:: figures/blend_fractions.png
   :width: 45.0%
.. |image6| image:: figures/redshift_ratio.png
   :width: 45.0%
.. |image7| image:: figures/flux_contamination.png
   :width: 45.0%
