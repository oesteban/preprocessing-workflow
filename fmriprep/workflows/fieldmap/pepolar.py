#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
PEpolar B0 estimation
~~~~~~~~~~~~~~~~~~~~~

PE-polar (Phase-Encoding POLARity) is the name coined by GE to the family of
methods to estimate the inhomogeneity of field B0 inside the scanner by using two
acquisitions with different (generally opposed) phase-encoding (PE) directions.

https://cni.stanford.edu/wiki/Data_Processing#Gradient-reversal_Unwarping_.28.27pepolar.27.29

This corresponds to the section 8.9.4 --multiple phase encoded directions (topup)--
of the BIDS specification.

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe
from niworkflows.interfaces.masks import BETRPT

from fmriprep.utils.misc import _first, gen_list
from fmriprep.interfaces import ImageDataSink, ReadSidecarJSON
from fmriprep.interfaces.topup import ConformTopupInputs
from fmriprep.viz import stripped_brain_overlay
from fmriprep.workflows.fieldmap.utils import create_encoding_file

WORKFLOW_NAME = 'FMAP_pepolar'


# pylint: disable=R0914
def pepolar_workflow(name=WORKFLOW_NAME, settings=None):
    """
    Estimates the fieldmap using TOPUP on series of at least two images
    acquired with varying :abbr:`PE (phase encoding)` direction.
    Generally, the images are :abbr:`SE (Spin-Echo)` and the
    :abbr:`PE (phase encoding)` directions are opposed (they can also
    be orthogonal).

    Outputs::

      outputnode.fmap_ref - The average magnitude image, skull-stripped
      outputnode.fmap_mask - The brain mask applied to the fieldmap
      outputnode.fmap - The estimated fieldmap in Hz

    """

    if settings is None:
        settings = {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['input_images']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_mask', 'fmap_ref']), name='outputnode')

    sortfmaps = pe.Node(niu.Function(function=_sort_fmaps,
                                     input_names=['input_images'],
                                     output_names=['pedir1', 'pedir2', 'pedir3', 'pedir4']),
                        name='SortFmaps')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(), iterfield=['in_file'], name='metadata')
    conform = pe.Node(ConformTopupInputs(), name='ConformInputs')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file', 'discard'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    # Run topup to estimate field distortions, do not estimate movement
    # since it is done in hmc_se
    topup = pe.Node(fsl.TOPUP(estmov=0), name='TopUp')

    # Use the least-squares method to correct the dropout of the SE images
    unwarp_mag = pe.Node(fsl.ApplyTOPUP(method='lsr'), name='TopUpApply')

    # Remove bias
    inu_n4 = pe.Node(N4BiasFieldCorrection(dimension=3), name='SE_bias')

    # Skull strip corrected SE image to get reference brain and mask
    mag_bet = pe.Node(BETRPT(mask=True, robust=True), name='SE_brain')

    workflow.connect([
        (inputnode, sortfmaps, [('input_images', 'input_images')]),
        (sortfmaps, meta, [('pe_variations', 'in_file')]),
        (sortfmaps, conform, [('pe_variations', 'in_files')]),
        (sortfmaps, encfile, [('pe_variations', 'input_images')]),
        (meta, encfile, [('out_dict', 'in_dict')]),
        (encfile, topup, [('parameters_file', 'encoding_file')]),
        (conform, topup, [('out_file', 'in_file')]),
        (topup, unwarp_mag, [('out_fieldcoef', 'in_topup_fieldcoef'),
                             ('out_movpar', 'in_topup_movpar')]),
        (encfile, unwarp_mag, [('parameters_file', 'encoding_file')]),
        (conform, unwarp_mag, [('out_file', 'in_files')]),
        (unwarp_mag, inu_n4, [('out_corrected', 'input_image')]),
        (inu_n4, mag_bet, [('output_image', 'in_file')]),

        (topup, outputnode, [('out_field', 'fmap')]),
        (mag_bet, outputnode, [('out_file', 'fmap_ref'),
                               ('mask_file', 'fmap_mask')])
    ])

    # Reports section
    se_svg = pe.Node(niu.Function(
        input_names=['in_file', 'overlay_file', 'out_file'], output_names=['out_file'],
        function=stripped_brain_overlay), name='SVG_SE_corr')
    se_svg.inputs.out_file = 'corrected_SE_and_mask.svg'

    se_svg_ds = pe.Node(
        ImageDataSink(base_directory=settings['output_dir']),
        name='SESVGDS',
    )

    workflow.connect([
        (unwarp_mag, se_svg, [('out_corrected', 'overlay_file')]),
        (mag_bet, se_svg, [('mask_file', 'in_file')]),
        (unwarp_mag, se_svg_ds, [('out_corrected', 'overlay_file')]),
        (mag_bet, se_svg_ds, [('mask_file', 'base_file')]),
        (se_svg, se_svg_ds, [('out_file', 'in_file')]),
        (inputnode, se_svg_ds, [(('input_images', _first), 'origin_file')])
    ])

    return workflow

def _sort_fmaps(input_images):
    ''' just a little data massaging'''
    return (sorted([fname for fname in input_images if 'epi' in fname] +
                   sorted([fname for fname in input_images if 'sbref' in fname])))
