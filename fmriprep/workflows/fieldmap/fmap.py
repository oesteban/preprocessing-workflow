#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap B0 estimation
~~~~~~~~~~~~~~~~~~~~~~

When the fieldmap is directly measured with a prescribed sequence (such as
:abbr:`SE (spiral echo)`), we only need to calculate the corresponding B-Spline
coefficients to adapt the fieldmap to the TOPUP tool.

https://cni.stanford.edu/wiki/GE_Processing#Fieldmaps

This corresponds to the section 8.9.3 --fieldmap image (and one magnitude image)--
of the BIDS specification.

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

WORKFLOW_NAME = 'Fieldmap_direct'
def fmap_workflow(name=WORKFLOW_NAME, settings=None):
    """
    Fieldmap workflow - when we have a sequence that directly measures the fieldmap
    we just need to mask it (using the corresponding magnitude image) to remove the
    noise in the surrounding air region, and ensure that units are Hz.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['fieldmap', 'fmap_mask', 'unwarp_direction',
                                                      'dwell_time']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap_rads', 'fmap_unmasked']),
                         name='outputnode')

    # Convert topup fieldmap to rad/s [ 1 Hz = 6.283 rad/s]
    fmap_scale = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=6.283),
                         name='fmap_scale')

    # Compute a mask from the fieldmap (??)
    fmap_abs = pe.Node(fsl.UnaryMaths(operation='abs', args='-bin'), name='fmap_abs')
    fmap_mul = pe.Node(fsl.BinaryMaths(operation='mul'), name='fmap_mul_mask')

    # Compute an smoothed field without mask
    fugue_unmask = pe.Node(fsl.FUGUE(save_unmasked_fmap=True), name='fmap_unmask')

    workflow.connect([
        (inputnode, fmap_scale, [('fieldmap', 'in_file')]),
        (inputnode, fmap_mul, [('fmap_mask', 'operand_file')]),
        (inputnode, fugue_unmask, [('unwarp_direction', 'unwarp_direction'),
                                   ('dwell_time', 'dwell_time')]),
        (fmap_scale, fmap_abs, [('out_file', 'in_file')]),
        (fmap_abs, fmap_mul, [('out_file', 'in_file')]),
        (fmap_scale, fugue_unmask, [('out_file', 'fmap_in_file')]),
        (fmap_mul, fugue_unmask, [('out_file', 'mask_file')]),
        (fmap_scale, outputnode, [('out_file', 'fmap_rads')]),
        (fugue_unmask, outputnode, [('fmap_out_file', 'fmap_unmasked')])
    ])
    return workflow
