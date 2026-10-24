import unittest
from lpr_utils import plate_matches_target

class PlateTargetsTests(unittest.TestCase):
    def test_full_and_partial_targets(self):
        for observed, target in [('กข 1234', '1234'), ('กข1234', '234'),
                                 ('กข ๑๒๓๔', '1234'), ('กข1234', 'กข 1234')]:
            with self.subTest(observed=observed, target=target):
                self.assertTrue(plate_matches_target(observed, target))

    def test_unrelated_and_empty_never_match(self):
        for observed, target in [('กข5678','1234'), ('1กข234','1234'),
                                 ('กข1234',''), ('กข1234',' - '), ('','1234')]:
            with self.subTest(observed=observed, target=target):
                self.assertFalse(plate_matches_target(observed, target))

if __name__ == '__main__':
    unittest.main()
