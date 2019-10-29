import unittest

from aki.utils.versioning import Version


class TestVersion(unittest.TestCase):

    def test_set_get(self):
        # Set the version by using the constructor.
        v = Version("1.0.0")
        self.assertEqual("1.0.0", v.get_as_string())
        self.assertEqual([1, 0, 0], v.get_as_array())

        # Set the version by using the setter.
        v.set("0.1.0")
        self.assertEqual("0.1.0", v.get_as_string())
        self.assertEqual([0, 1, 0], v.get_as_array())

        v.set("0.0.1")
        self.assertEqual("0.0.1", v.get_as_string())
        self.assertEqual([0, 0, 1], v.get_as_array())

    def test_compare(self):
        v1 = Version("1.0.0")
        v2 = Version("0.1.0")
        v3 = Version("0.0.1")
        v4 = Version("1.1.0")
        v5 = Version("1.1.1")
        v6 = Version("2.0.0")

        self.assertEqual(0, v1.compare(v1))
        self.assertEqual(1, v1.compare(v2))
        self.assertEqual(-1, v3.compare(v2))
        self.assertEqual(-1, v1.compare(v6))
        self.assertEqual(-1, v4.compare(v5))
        self.assertEqual(1, v6.compare(v5))


if __name__ == '__main__':
    unittest.main()
