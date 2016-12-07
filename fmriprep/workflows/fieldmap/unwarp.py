#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import utility as niu
from niworkflows.interfaces.registration import ANTSRegistrationRPT, ANTSApplyTransformsRPT

from fmriprep.interfaces.bids import ReadSidecarJSON
from fmriprep.interfaces.fmap import FieldCoefficients, GenerateMovParams
from fmriprep.interfaces.topup import ConformTopupInputs
from fmriprep.workflows.fieldmap.utils import create_encoding_file
SDC_UNWARP_NAME = 'SDC_unwarp'


def sdc_unwarp(name=SDC_UNWARP_NAME, ref_vol=None, method='jac', testing=False):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.in_file - the image(s) to which this correction will be applied
      inputnode.in_mask - a brain mask corresponding to the in_file image
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space
      inputnode.fmap - a fieldmap in Hz
      inputnode.hmc_movpar - the head motion parameters (iff inputnode.in_file is only
                             one 4D file)

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.out_file - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fmap_ref', 'fmap_mask', 'fmap',
                'hmc_movpar']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(), iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['unwarp_param', 'warp_param'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    # Compute movpar file iff we have several images with different
    # PE directions.
    conform = pe.Node(ConformTopupInputs(), name='ConformInputs')
    if ref_vol is not None:
        conform.inputs.in_ref = ref_vol

    warpref = pe.Node(niu.Function(
        function=_warp_reference,
        input_names=['in_file', 'in_fieldmap', 'in_params'],
        output_names=['out_corrected']),
        name='TestWarpReference')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if testing:
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')

    fmap2ref = pe.Node(ANTSRegistrationRPT(from_file=ants_settings, output_warped_image=True,
                       generate_report=True), name='FMap2ImageMagnitude')

    applyxfm = pe.Node(ANTSApplyTransformsRPT(
        generate_report=True, dimension=3, interpolation='Linear'), name='FMap2ImageFieldmap')

    gen_movpar = pe.Node(GenerateMovParams(), name='GenerateMovPar')
    topup_adapt = pe.Node(FieldCoefficients(), name='TopUpCoefficients')

    # Use the least-squares method to correct the dropout of the input images
    unwarp = pe.Node(fsl.ApplyTOPUP(method=method, in_index=[1]), name='TopUpApply')

    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, conform, [('in_file', 'in_files'),
                              ('hmc_movpar', 'in_mats')]),
        (inputnode, applyxfm, [('fmap', 'input_image')]),
        (inputnode, encfile, [('in_file', 'input_images')]),
        (inputnode, fmap2ref, [('fmap_ref', 'moving_image'),
                               ('fmap_mask', 'moving_image_mask')]),
        (inputnode, warpref, [('fmap_ref', 'in_file'),
                              ('fmap', 'in_fieldmap')]),
        (encfile, warpref, [('unwarp_param', 'in_params')]),
        (conform, fmap2ref, [('ref_vol', 'fixed_image'),
                           ('ref_mask', 'fixed_image_mask')]),
        (conform, applyxfm, [('ref_vol', 'reference_image')]),
        (conform, topup_adapt, [('ref_vol', 'in_ref')]),
        #                      ('out_movpar', 'in_movpar')]),

        (meta, encfile, [('out_dict', 'in_dict')]),

        (fmap2ref, applyxfm, [('forward_transforms', 'transforms')]),
        (conform, gen_movpar, [('out_file', 'in_file'),
                             ('mat_file', 'in_mats')]),
        (gen_movpar, topup_adapt, [('out_movpar', 'in_movpar')]),
        (applyxfm, topup_adapt, [('output_image', 'in_file')]),
        (conform, unwarp, [('out_file', 'in_files')]),
        (topup_adapt, unwarp, [('out_fieldcoef', 'in_topup_fieldcoef'),
                               ('out_movpar', 'in_topup_movpar')]),
        (encfile, unwarp, [('unwarp_param', 'encoding_file')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])

    return workflow


def _warp_reference(in_file, in_fieldmap, in_params):
    import numpy as np
    import nibabel as nb
    from fmriprep.interfaces.fmap import FieldCoefficients
    from nipype.interfaces.fsl import ApplyTOPUP

    nrows = np.loadtxt(in_params).shape[0]

    if nrows > 1:
        nii = nb.load(in_file)
        merged_inputs = 'inputs.nii.gz'
        nb.concat_images([nii] * nrows).to_filename(merged_inputs)
    else:
        merged_inputs = in_file

    movpar = 'movpar.txt'
    np.savetxt(movpar, np.zeros((nrows, 6)))
    coeffs = FieldCoefficients(
        in_ref=in_file, in_file=in_fieldmap, in_movpar=movpar).run()

    warp = ApplyTOPUP(
        method='jac', in_index=[1],
        in_files=merged_inputs,
        in_topup_fieldcoef=coeffs.outputs.out_fieldcoef,
        in_topup_movpar=coeffs.outputs.out_movpar,
        encoding_file=in_params).run()

    return warp.outputs.out_corrected
