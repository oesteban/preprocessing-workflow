#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
TopUp helpers
~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits
)
from fmriprep.interfaces.images import genfname

LOGGER = logging.getLogger('interface')

class ConformTopupInputsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='input files')
    in_mats = InputMultiPath(File(exists=True), desc='input files')
    in_ref = traits.Int(0, usedefault=True, desc='reference volume')


class ConformTopupInputsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='merged image')
    out_movpar = File(exists=True, desc='output movement parameters')

class ConformTopupInputs(BaseInterface):

    """
    This function interprets that we are dealing with a
    multiple PE (phase encoding) input if it finds several
    files in in_files.

    If we have several images with various PE directions,
    it will compute the HMC parameters between them using
    an embedded workflow.

    It just forwards the two inputs otherwise.
    """
    input_spec = ConformTopupInputsInputSpec
    output_spec = ConformTopupInputsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ConformTopupInputs, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _one_flirt_file(self, in_files):
        """
        Input is one FLIRT'ed file (typically: _bold),
        must come along with the corresponding HMC matrices.

        """
        nmats = len(self.inputs.in_mats)
        if len(in_files) > 1:
            raise RuntimeError('Only one file, aligned for head motion '
                               'expected. Got %s' % in_files)
        ntrs = nb.load(in_files[0]).get_data().shape[-1]

        if nmats != ntrs:
            raise RuntimeError('Number of TRs (%s) and input HMC matrices (%s) do not '
                               ' match' % (ntrs, nmats))

        self._results['out_file'] = in_files[0]
        self._results['out_movpar'] = genfname(in_files[0], suffix='_movpar')
        movpar = _generate_topup_movpar(self.inputs.in_mats)
        np.savetxt(self._results['out_movpar'], movpar)


    def _run_interface(self, runtime):
        from builtins import (str, bytes)
        in_files = self.inputs.in_files

        if isinstance(in_files, (str, bytes)):
            in_files = [in_files]

        # If HMC mats are present, we expect only
        # one file, aligned with flirt.
        if isdefined(self.inputs.in_mats):
            self._one_flirt_file(in_files)
            return runtime

        # Head motion correction
        fslmerge = fsl.Merge(dimension='t', in_files=in_files)
        hmc = fsl.MCFLIRT(cost='normcorr', ref_vol=self.inputs.in_ref, save_mats=True)
        hmc.inputs.in_file = fslmerge.run().outputs.merged_file
        hmc_res = hmc.run()
        self._results['out_file'] = hmc_res.outputs.out_file
        self._results['out_movpar'] = genfname(in_files[0], suffix='_movpar')
        movpar = _generate_topup_movpar(hmc_res.outputs.par_file)
        np.savetxt(self._results['out_movpar'], movpar)

        # Read Encoding direction

        return runtime

def _generate_topup_movpar(in_mats):
    # TODO!
    return np.zeros((len(in_mats), 6))
