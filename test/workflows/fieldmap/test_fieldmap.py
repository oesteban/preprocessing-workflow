import json
from fmriprep.workflows.fieldmap.utils import create_encoding_file
from fmriprep.workflows.fieldmap import phdiff
from fmriprep.workflows.fieldmap import pepolar
import re
import mock
from test.workflows.utilities import TestWorkflow

class TestFieldMap(TestWorkflow):

    SOME_INT = 3

    def test_phasediff_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.',
            'output_dir': '.'
        }

        # SET UP EXPECTATIONS
        expected_interfaces = [
            'DataSink', 'MultiImageMaths', 'ApplyMask', 'FUGUE', 'Merge', 'MathsCommand',
            'MultiImageMaths', 'IdentityInterface', 'IdentityInterface', 'Function', 'Function',
            'Function', 'BETRPT', 'N4BiasFieldCorrection', 'IntraModalMerge', 'SpatialFilter',
            'PRELUDE', 'Function', 'Function', 'DataSink', 'Function', 'IdentityInterface',
            'ReadSidecarJSON', 'IdentityInterface'
        ]

        expected_outputs = ['outputnode.fmap', 'outputnode.fmap_mask',
                            'outputnode.fmap_ref']
        expected_inputs = ['inputnode.input_images']

        # RUN
        result = phdiff.phdiff_workflow(mock_settings)

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(phdiff.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

    def test_pepolar_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.',
            'output_dir': '.'
        }

        # SET UP EXPECTATIONS
        expected_interfaces = ['Function', 'N4BiasFieldCorrection', 'BETRPT',
                               'MCFLIRT', 'Merge', 'Split', 'TOPUP',
                               'ApplyTOPUP', 'Function', 'ImageDataSink',
                               'IdentityInterface', 'ReadSidecarJSON',
                               'IdentityInterface']
        expected_outputs = ['outputnode.fmap', 'outputnode.fmap_mask',
                            'outputnode.fmap_ref']
        expected_inputs = ['inputnode.input_images']

        # RUN
        result = pepolar.pepolar_workflow(settings=mock_settings)

        # ASSERT
        self.assertIsAlmostExpectedWorkflow(pepolar.WORKFLOW_NAME,
                                            expected_interfaces,
                                            expected_inputs,
                                            expected_outputs,
                                            result)

    @mock.patch('nibabel.load')
    @mock.patch('numpy.savetxt')
    def test_create_encoding_file(self, mock_savetxt, mock_load):
        # SET UP INPUTS
        fieldmaps = 'some_file.nii.gz'
        in_dict = { 'TotalReadoutTime': 'a_time',
                    'PhaseEncodingDirection': ['i']
        }

        # SET UP EXPECTATIONS
        mock_load(fieldmaps).shape = ['', '', '', self.SOME_INT]
        expected_enc_table = ([[1, 0, 0, in_dict['TotalReadoutTime']]] *
                              self.SOME_INT)

        # RUN
        out_file = create_encoding_file(fieldmaps, in_dict)

        # ASSERT
        # the output file is called parameters.txt
        self.assertRegexpMatches(out_file, '/parameters.txt$')
        # nibabel.load was called with fieldmaps
        mock_load.assert_called_with(fieldmaps)
        # numpy.savetxt was called once. It was called with expected_enc_table
        mock_savetxt.assert_called_once_with(mock.ANY, expected_enc_table,
                                             fmt=mock.ANY)

