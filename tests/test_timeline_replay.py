import unittest

from nightload.app import Application
from nightload.replay import replay_feasible, replay_greedy, workers_from_roster
from nightload.store import Store
from nightload.timeline import (WINDOW_MINUTES, build_timeline, collect_nights,
                                compute_duration_quantiles, night_bounds, night_of,
                                peak_composition, total_occupancy)


class TimelineFixtureTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        Application(self.store).seed()
        self.events = self.store.all('events')
        self.calls = self.store.all('calls')
        self.rosters = self.store.all('rosters')
        self.targets = {row['id']: row['response_minutes'] for row in self.store.all('targets')}
        self.quantiles = compute_duration_quantiles(self.events, 0.9)

    def tearDown(self):
        self.store.close()

    def test_night_bounds_and_assignment(self):
        start, end = night_bounds('2026-10-01')
        self.assertEqual(start.isoformat(), '2026-10-01T22:00:00')
        self.assertEqual(end.isoformat(), '2026-10-02T06:00:00')
        self.assertEqual(night_of('2026-10-02T00:30'), '2026-10-01')
        self.assertEqual(night_of('2026-10-01T23:59'), '2026-10-01')

    def test_open_end_not_completed_but_censored_to_window(self):
        timeline = build_timeline(self.events, self.calls, '2026-10-01',
                                  duration_quantiles=self.quantiles)
        self.assertEqual(timeline['open_end_event_ids'], ['e3'])
        e3 = next(item for item in timeline['intervals'] if item['event_id'] == 'e3')
        self.assertIsNone(e3['release_minute'])
        # 开放结束事件在 observed 口径下不产生占用网格，但需求时点仍然保留
        self.assertEqual(len(timeline['epochs']), 3)
        self.assertTrue(all(item['event_id'] != 'e3' or item['never_arrived']
                            for item in timeline['epochs']))

    def test_imputed_open_end_gets_quantile_duration(self):
        timeline = build_timeline(self.events, self.calls, '2026-10-01',
                                  open_end_policy='imputed', duration_quantiles=self.quantiles)
        e3 = next(item for item in timeline['intervals'] if item['event_id'] == 'e3')
        self.assertTrue(e3['imputed'])
        # e3 同时缺失到场与结束：从需求时点（05:50 → 第 470 分钟）起按 basic 分位数补全，
        # 补全后仍越过窗口，按窗口截断并保留 censored 标记
        self.assertEqual(e3['arrived_minute'], 470)
        self.assertEqual(e3['release_minute'], WINDOW_MINUTES)
        self.assertTrue(e3['censored'])
        epoch = next(item for item in timeline['epochs'] if item['event_id'] == 'e3')
        self.assertTrue(epoch['never_arrived'])

    def test_call_merge_uncertainty_brackets_demand_count(self):
        episode = build_timeline(self.events, self.calls, '2026-10-01',
                                 call_policy='episode', duration_quantiles=self.quantiles)
        split = build_timeline(self.events, self.calls, '2026-10-01', call_policy='calls',
                               merge_gap=0, duration_quantiles=self.quantiles)
        merged = build_timeline(self.events, self.calls, '2026-10-01', call_policy='calls',
                                merge_gap=5, duration_quantiles=self.quantiles)
        # fixtures 共 3 次呼叫：x1 的两次呼叫在 merge_gap=0 时拆成两个需求，
        # 因此 calls 口径 4 个需求时点，episode 口径每场事件一个共 3 个；
        # 开放结束事件没有呼叫，其需求仍以 never_arrived 保留
        self.assertEqual(len(episode['epochs']), 3)
        self.assertEqual(len(split['epochs']), 4)
        self.assertEqual(len(merged['epochs']), 3)
        self.assertTrue(any(item['repeat'] for item in split['epochs']))

    def test_peak_is_joint_not_sum_of_category_extremes(self):
        timeline = build_timeline(self.events, self.calls, '2026-10-01',
                                  duration_quantiles=self.quantiles)
        peak = peak_composition(timeline)
        self.assertEqual(peak['peak'], 2)
        self.assertEqual(peak['by_skill'], {'basic': 1, 'transfer': 1})
        self.assertEqual({item['event_id'] for item in peak['contributors']}, {'e1', 'e2'})
        self.assertEqual(max(total_occupancy(timeline)), 2)

    def test_collect_nights_groups_after_midnight_events(self):
        self.assertEqual(collect_nights(self.events), ['2026-10-01'])


class SyntheticWindowTests(unittest.TestCase):
    """用合成数据检验跨窗口片段不丢失、共同占用与技能互斥。"""

    def build(self, events, calls=None, rosters=None):
        calls = calls or []
        rosters = rosters or [{
            'id': 'r0', 'person': 'p0', 'night': '2026-11-01',
            'skills': ['basic', 'transfer'],
        }]
        return (build_timeline(events, calls, '2026-11-01',
                               duration_quantiles=compute_duration_quantiles(events, 0.9)),
                workers_from_roster(rosters, '2026-11-01'))

    def test_carry_in_fragment_is_marked_not_dropped(self):
        events = [{
            'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
            'start': '2026-11-01T21:40', 'arrived': '2026-11-01T21:45',
            'end': '2026-11-01T22:20',
        }]
        timeline, _ = self.build(events)
        self.assertEqual(timeline['carry_in_event_ids'], ['e1'])
        interval = timeline['intervals'][0]
        self.assertEqual(interval['start_minute'], 0)
        self.assertEqual(interval['actual_start_minute'], -20)
        self.assertEqual(interval['release_minute'], 20)

    def test_event_crossing_window_end_is_censored(self):
        events = [{
            'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
            'start': '2026-11-02T05:40', 'arrived': '2026-11-02T05:45',
            'end': '2026-11-02T06:30',
        }]
        timeline, _ = self.build(events)
        self.assertEqual(timeline['censored_event_ids'], ['e1'])
        self.assertEqual(timeline['intervals'][0]['release_minute'], WINDOW_MINUTES)

    def test_one_worker_cannot_serve_two_skills_at_once(self):
        events = [
            {'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:30'},
            {'id': 'e2', 'episode': 'x2', 'resident': 'r2', 'skill': 'transfer',
             'start': '2026-11-01T22:05', 'arrived': '2026-11-01T22:05',
             'end': '2026-11-01T22:25'},
        ]
        timeline, workers = self.build(events)
        result = replay_greedy(timeline, workers, {'basic': 10, 'transfer': 15})
        self.assertEqual(result['metrics']['breach_count'], 1)
        waiting = next(item for item in result['breaches'])
        self.assertEqual(waiting['event_id'], 'e2')
        self.assertEqual(waiting['wait_minutes'], 25)

    def test_feasible_search_distinguishes_allocation_from_capacity(self):
        # 两名双证人员、两条同时需求：可行解存在
        rosters = [
            {'id': 'r1', 'person': 'p1', 'night': '2026-11-01', 'skills': ['basic', 'transfer']},
            {'id': 'r2', 'person': 'p2', 'night': '2026-11-01', 'skills': ['basic', 'transfer']},
        ]
        events = [
            {'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:30'},
            {'id': 'e2', 'episode': 'x2', 'resident': 'r2', 'skill': 'transfer',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:20'},
        ]
        timeline, workers = self.build(events, rosters=rosters)
        feasible = replay_feasible(timeline, workers, {'basic': 10, 'transfer': 15})
        self.assertTrue(feasible['feasible'])
        self.assertEqual(feasible['metrics']['breach_count'], 0)

        # 只有一人时无可行解，回退贪心并明确标记
        rosters_one = rosters[:1]
        timeline, workers = self.build(events, rosters=rosters_one)
        infeasible = replay_feasible(timeline, workers, {'basic': 10, 'transfer': 15})
        self.assertFalse(infeasible['feasible'])
        self.assertFalse(infeasible['search_budget_exhausted'])
        self.assertGreaterEqual(infeasible['metrics']['breach_count'], 1)

    def test_repeated_calls_during_active_service_reuse_worker(self):
        events = [{
            'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
            'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:02',
            'end': '2026-11-01T22:40',
        }]
        calls = [
            {'id': 'c1', 'episode': 'x1', 'time': '2026-11-01T22:00'},
            {'id': 'c2', 'episode': 'x1', 'time': '2026-11-01T22:10'},
        ]
        timeline, workers = self.build(events, calls=calls)
        split = build_timeline(events, calls, '2026-11-01', call_policy='calls', merge_gap=5,
                               duration_quantiles=compute_duration_quantiles(events, 0.9))
        result = replay_greedy(split, workers, {'basic': 10})
        self.assertEqual(len(result['assignments']), 2)
        self.assertEqual({item['worker'] for item in result['assignments']}, {'p0'})
        # 第二次呼叫发生在服务进行中：即时承接，不产生等待与未及时响应
        self.assertEqual(result['metrics']['breach_count'], 0)
        self.assertTrue(all(item['wait_minutes'] == 0 for item in result['assignments']))
        self.assertTrue(any(item.get('repeat_in_service') for item in result['assignments']))

        feasible = replay_feasible(split, workers, {'basic': 10})
        self.assertTrue(feasible['feasible'])
        self.assertEqual(feasible['metrics']['breach_count'], 0)

    def test_repeat_call_after_service_ends_is_not_reassigned_to_others(self):
        events = [{
            'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
            'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
            'end': '2026-11-01T22:20',
        }]
        calls = [
            {'id': 'c1', 'episode': 'x1', 'time': '2026-11-01T22:00'},
            {'id': 'c2', 'episode': 'x1', 'time': '2026-11-01T22:26'},
        ]
        rosters = [
            {'id': 'r1', 'person': 'p1', 'night': '2026-11-01', 'skills': ['basic']},
            {'id': 'r2', 'person': 'p2', 'night': '2026-11-01', 'skills': ['basic']},
        ]
        split = build_timeline(events, calls, '2026-11-01', call_policy='calls', merge_gap=0,
                               duration_quantiles=compute_duration_quantiles(events, 0.9))
        # 可行性搜索中重复呼叫必须归属同一事件人员，服务已结束即无解；
        # 贪心口径允许现场把它作为新需求派给另一名空闲人员
        feasible = replay_feasible(split, workers_from_roster(rosters, '2026-11-01'),
                                   {'basic': 10})
        self.assertFalse(feasible['feasible'])
        greedy = replay_greedy(split, workers_from_roster(rosters, '2026-11-01'), {'basic': 10})
        repeat_assignment = next(item for item in greedy['assignments'] if item['repeat'])
        # 服务结束后的重复呼叫作为新需求重新排队，可被任一人承接
        self.assertIn(repeat_assignment['worker'], {'p1', 'p2'})
        self.assertNotIn('repeat_in_service', repeat_assignment)

    def test_search_budget_exhaustion_is_reported_not_disguised(self):
        rosters = [
            {'id': 'r1', 'person': 'p1', 'night': '2026-11-01', 'skills': ['basic', 'transfer']},
            {'id': 'r2', 'person': 'p2', 'night': '2026-11-01', 'skills': ['basic', 'transfer']},
        ]
        events = [
            {'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:30'},
            {'id': 'e2', 'episode': 'x2', 'resident': 'r2', 'skill': 'transfer',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:20'},
        ]
        timeline, workers = self.build(events, rosters=rosters)
        # 预算极小必然耗尽：结果必须显式标记，而不是把超时当作“不可行”
        capped = replay_feasible(timeline, workers, {'basic': 10, 'transfer': 15},
                                 node_budget=2)
        self.assertTrue(capped['search_budget_exhausted'])
        self.assertFalse(capped['feasible'])
        # 同一数据在充足预算下可找到可行解
        full = replay_feasible(timeline, workers, {'basic': 10, 'transfer': 15})
        self.assertTrue(full['feasible'])
        self.assertFalse(full['search_budget_exhausted'])


if __name__ == '__main__':
    unittest.main()
