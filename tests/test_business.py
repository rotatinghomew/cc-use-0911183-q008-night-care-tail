import unittest
from nightload.app import Application
from nightload.store import Store
from nightload.events import save_event, interval_overlap, window_events
from nightload.load import descriptive_load
from nightload.roster import skills_on_duty, save_roster


class NightTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        Application(self.store).seed()

    def tearDown(self):
        self.store.close()

    def test_cross_midnight_overlap(self):
        self.assertEqual(interval_overlap('2026-10-01T23:50', '2026-10-02T00:30', '2026-10-02T00:00', '2026-10-02T01:00'), 30)

    def test_open_event_uses_window_not_completion(self):
        rows = window_events(self.store, '2026-10-02T05:00', '2026-10-02T06:00')
        self.assertTrue(rows[0]['open_end'])
        self.assertEqual(rows[0]['observed_minutes'], 10)

    def test_nonoverlap_zero(self):
        self.assertEqual(interval_overlap('2026-10-01T10:00', '2026-10-01T11:00', '2026-10-01T12:00', '2026-10-01T13:00'), 0)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            window_events(self.store, '2026-10-02', '2026-10-01')

    def test_invalid_arrival(self):
        row = dict(self.store.get('events', 'e1'), arrived='2026-10-01T23:00')
        with self.assertRaises(ValueError):
            save_event(self.store, row)

    def test_average_preserves_open_count(self):
        self.assertEqual(descriptive_load(self.store, '2026-10-01T22:00', '2026-10-02T06:00')['open_events'], 1)

    def test_skills_are_overlapping_counts(self):
        result = skills_on_duty(self.store, '2026-10-01')
        self.assertEqual(result['people'], 2)
        self.assertEqual(sum(result['qualified_by_skill'].values()), 3)

    def test_duplicate_person_rejected(self):
        row = dict(self.store.get('rosters', 'r1'), id='new')
        with self.assertRaises(ValueError):
            save_roster(self.store, row)

    def test_new_night_allowed(self):
        row = dict(self.store.get('rosters', 'r1'), id='new', night='2026-10-02')
        self.assertEqual(save_roster(self.store, row), 1)

    def test_no_events_wait_is_unknown(self):
        self.assertIsNone(descriptive_load(self.store, '2026-11-01T00:00', '2026-11-01T01:00')['observed_wait_mean'])
