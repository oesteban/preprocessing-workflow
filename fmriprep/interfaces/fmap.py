#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from shutil import copy
import numpy as np
import nibabel as nb
from nipype.interfaces import fsl

from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
                                    File, isdefined)

class FieldCoefficientsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_ref = File(exists=True, mandatory=True, desc='reference file')
    in_movpar = File(exists=True, desc='input head motion parameters')

class FieldCoefficientsOutputSpec(TraitedSpec):
    out_fieldcoef = File(desc='the calculated BSpline coefficients')
    out_movpar = File(desc='the calculated head motion coefficients')

class FieldCoefficients(BaseInterface):
    """
    The FieldCoefficients interface wraps a workflow to compute the BSpline coefficients
    corresponding to the input fieldmap (in Hz). It also sets the appropriate nifti headers
    to be digested by ApplyTOPUP.
    """
    input_spec = FieldCoefficientsInputSpec
    output_spec = FieldCoefficientsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FieldCoefficients, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        movpar = None
        if isdefined(self.inputs.in_movpar):
            movpar = self.inputs.in_movpar

        self._results['out_fieldcoef'], self._results['out_movpar'] = _gen_coeff(
            self.inputs.in_file, self.inputs.in_ref, movpar)
        return runtime


def _gen_coeff(in_file, in_ref, in_movpar=None):
    """Convert to a valid fieldcoeff"""


    def _get_fname(in_file):
        import os.path as op
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        return op.abspath(fname)

    out_topup = _get_fname(in_file)

    # 1. Add one dimension (4D image) of 3D coordinates
    #    so that this is a 3D deformation field
    im0 = nb.load(in_file)
    data = np.zeros_like(im0.get_data())
    sizes = data.shape[:3]
    spacings = im0.get_header().get_zooms()[:3]
    im1 = nb.Nifti1Image(data, im0.get_affine(), im0.get_header())
    im4d = nb.concat_images([im0, im1, im1])
    im4d_fname = '{}_{}'.format(out_topup, 'field4D.nii.gz')
    im4d.to_filename(im4d_fname)

    # 2. Warputils to compute bspline coefficients
    to_coeff = fsl.WarpUtils(out_format='spline', knot_space=(2, 2, 2))
    to_coeff.inputs.in_file = im4d_fname
    to_coeff.inputs.reference = in_ref

    # 3. Remove unnecessary dims (Y and Z)
    get_first = fsl.ExtractROI(t_min=0, t_size=1)
    get_first.inputs.in_file = to_coeff.run().outputs.out_file

    # 4. Set correct header
    # see https://github.com/poldracklab/preprocessing-workflow/issues/92
    img = nb.load(get_first.run().outputs.roi_file)
    hdr = img.get_header().copy()
    hdr['intent_p1'] = spacings[0]
    hdr['intent_p2'] = spacings[1]
    hdr['intent_p3'] = spacings[2]
    hdr['intent_code'] = 2016

    sform = np.eye(4)
    sform[:3, 3] = sizes
    hdr.set_sform(sform, code='scanner')
    hdr['qform_code'] = 1

    out_movpar = '{}_movpar.txt'.format(out_topup)
    copy(in_movpar, out_movpar)

    out_fieldcoef = '{}_fieldcoef.nii.gz'.format(out_topup)
    nb.Nifti1Image(img.get_data(), None, hdr).to_filename(out_fieldcoef)

    return out_fieldcoef, out_movpar
