import os   
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from utils.scheduler import SchedulerEnum, get_scheduler_from_str

class SchedulerManageTest(unittest.TestCase):
    def test_get_scheduler_from_str(self):
        """
        Test the get_scheduler_from_str function to ensure it correctly converts
        a string to the corresponding SchedulerEnum value.
        """
        # Test case: Valid scheduler string
        scheduler_str = "CosineAnnealingLR"
        result = get_scheduler_from_str(scheduler_str)
        self.assertEqual(result, SchedulerEnum.CosineAnnealingLR)

        # Test case: Valid scheduler string with different casing
        scheduler_str_lower = "cosineannealinglr"
        with self.assertRaises(ValueError) as context:
            result_lower = get_scheduler_from_str(scheduler_str_lower)
        self.assertIn("Unknown scheduler name", str(context.exception))

        # Test case: Invalid scheduler string
        invalid_scheduler_str = "InvalidScheduler"
        with self.assertRaises(ValueError) as context:
            get_scheduler_from_str(invalid_scheduler_str)
        self.assertIn("Unknown scheduler name", str(context.exception))

if __name__ == "__main__":
    unittest.main()
