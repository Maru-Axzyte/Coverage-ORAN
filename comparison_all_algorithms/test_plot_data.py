import unittest
from main import fingerprint, matched_service


class DataTests(unittest.TestCase):
    def test_fingerprint_tracks_coordinates_and_demand(self):
        a = [dict(id=1, x_m=3., y_m=5., require_mbps=10.)]
        self.assertEqual(fingerprint(a), fingerprint(list(reversed(a))))
        b = [dict(a[0], require_mbps=15.)]
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_resource_matching_is_pairwise_and_requires_feasibility(self):
        row = dict(num_ues=70, map_index=3, map_seed=5, map_fingerprint='x',
                   feasible=True, served=60, demand_mbps=550.)
        self.assertTrue(matched_service(row, dict(row)))
        for change in (dict(map_seed=6), dict(map_fingerprint='y'), dict(served=61),
                       dict(demand_mbps=555.), dict(feasible=False)):
            self.assertFalse(matched_service(row, dict(row, **change)))


if __name__ == '__main__':
    unittest.main()
