import unittest

from weatherlearn.models.pangu.utils.shift_window_mask import get_shift_window_mask


class TestShiftWindowMask(unittest.TestCase):
    def test_shape(self):
        input_resolution = (8, 186, 360)
        window_size = (2, 6, 12)
        shift_size = (1, 3, 6)
        mask = get_shift_window_mask(input_resolution, window_size, shift_size)
        self.assertEqual(mask.shape, (30, 4 * 31, 2 * 6 * 12, 2 * 6 * 12))
