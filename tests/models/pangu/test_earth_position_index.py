import unittest

from weatherlearn.models.pangu.utils.earth_position_index import get_earth_position_index


class TestEarthPositionIndex(unittest.TestCase):
    def test_shape(self):
        window_size = (2, 6, 12)
        earth_position_index = get_earth_position_index(window_size)
        self.assertEqual(earth_position_index.shape, (2 * 6 * 12, 2 * 6 * 12))
