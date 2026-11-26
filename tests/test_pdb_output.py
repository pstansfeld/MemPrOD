"""
Tests for MemPrOD's PDB output (Deformations.pdb, the contact and surface files).

Run from the repository root with:  python -m unittest discover tests
"""

import os
import sys
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from memprod.MemPrOD import contact_weight, dual_membrane_forwarded_args, option_was_given, pdb_atom_line  # noqa: E402


def fields(line):
    """The fixed PDB columns of an ATOM record."""
    return (line[:6], int(line[6:11]), line[12:16].strip(), line[17:20], line[21], int(line[22:26]),
            float(line[30:38]), float(line[38:46]), float(line[46:54]), float(line[54:60]), float(line[60:66]))


class TestPdbOutput(unittest.TestCase):

    def test_columns(self):
        # A surface bead as Deformations.pdb writes it: coordinates and B-factor read back from their own columns
        line = pdb_atom_line(12, "BB", "DUM", "Z", 1, 38.641, 61.8, -15.92, 1.0, -1.067)
        self.assertNotIn("\t", line)
        self.assertEqual(len(line.rstrip("\n")), 66)
        self.assertEqual(fields(line), ("ATOM  ", 12, "BB", "DUM", "Z", 1, 38.641, 61.8, -15.92, 1.0, -1.07))

    def test_wide_values_stay_in_their_columns(self):
        # Negative coordinates to -999.999 and B-factors to -99.99 fit; serial numbers wrap as in the PDB format
        line = pdb_atom_line(123456, "SC1", "MET", "A", 12, -999.999, 999.999, -100.0, 1.0, -99.99)
        self.assertEqual(fields(line)[1:], (23456, "SC1", "MET", "A", 12, -999.999, 999.999, -100.0, 1.0, -99.99))


class TestCalculationHelpers(unittest.TestCase):

    def test_dual_membrane_forwarding_skips_equals_form_options(self):
        argv = [
            "MemPrOD",
            "--file_name=original.pdb",
            "--output=original-output",
            "--dual_membrane=50",
            "--membrane_thickness=38",
            "--outer_membrane_thickness=40",
            "--grid_density", "3",
        ]
        self.assertEqual(dual_membrane_forwarded_args(argv), ["--grid_density", "3"])

    def test_explicit_equals_form_option_is_detected(self):
        argv = ["MemPrOD", "--membrane_thickness=42"]
        self.assertTrue(option_was_given(argv, "-mt", "--membrane_thickness"))
        self.assertFalse(option_was_given(argv, "-dm", "--dual_membrane"))

    def test_contact_weight_decreases_across_contact_range(self):
        weights = contact_weight(np.array([0.3, 1.0, 1.7]))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))


if __name__ == "__main__":
    unittest.main()
