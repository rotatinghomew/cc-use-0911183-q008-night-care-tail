import json
import unittest
from pathlib import Path

from nightload.attribution import (CAPABILITY_ALLOCATION, CAPACITY_INSUFFICIENT,
                                   DEMAND_BEYOND_DISTRIBUTION, STAFFING_SHORTFALL,
                                   reconcile_night, reconcile_period)
from nightload.contingency import (CONSERVATIVE_SCENARIO, binomial_bounds,
                                   bootstrap_night_rate, classify_surge,
                                   compare_plans, evaluate_plan, validate_plan)

DATA_PATH = Path(__file__).resolve().parent.parent / 'examples' / 'operations.json'
TARGETS = {'basic': 10, 'transfer': 15}
BASELINE = {
    'id': 'plan-baseline',
    'workers': [{'skills': ['basic'], 'count': 2},
                {'skills': ['basic', 'transfer'], 'count': 1}],
}
FLEX_FOUR = {
    'id': 'plan-flex-four',
    'workers': [{'skills': ['basic'], 'count': 2},
                {'skills': ['basic', 'transfer'], 'count': 2}],
}
SPECIALIST_FOUR = {
    'id': 'plan-specialist-four',
    'workers': [{'skills': ['basic'], 'count': 2},
                {'skills': ['basic', 'transfer'], 'count': 1},
                {'skills': ['transfer'], 'count': 1}],
}
ASSUMPTION = {
    'id': 'assumption-night-v1',
    'scenario': {'id': CONSERVATIVE_SCENARIO, 'open_end_policy': 'imputed', 'call_policy': 'calls'},
    'merge_gap_minutes': 5, 'impute_quantile': 0.9,
}


class HistoricalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
        cls.events = cls.data['events']
        cls.calls = cls.data['calls']
        cls.rosters = cls.data['rosters']

    def test_plan_validation_rejects_unknown_and_duplicate_skills(self):
        with self.assertRaises(ValueError):
            validate_plan({'id': 'x', 'workers': [{'skills': ['ghost'], 'count': 1}]},
                          {'basic', 'transfer'})
        with self.assertRaises(ValueError):
            validate_plan({'id': 'x', 'workers': [
                {'skills': ['basic'], 'count': 1},
                {'skills': ['basic'], 'count': 1},
            ]}, {'basic'})

    def test_evaluation_keeps_joint_night_structure(self):
        result = evaluate_plan(self.events, self.calls, self.rosters, BASELINE, TARGETS,
                               replicates=200)
        self.assertEqual(result['conservative']['breach_nights'],
                         result['scenarios'][CONSERVATIVE_SCENARIO]['breach_nights'])
        # 每夜结果必须完整保留，且普通夜与突发夜样本数之和等于总夜数
        groups = result['conservative']['groups']
        self.assertEqual(groups['surge']['nights'] + groups['ordinary']['nights'],
                         groups['all']['nights'])
        surge = result['surge_classification']
        self.assertEqual(len(surge['surge_nights']), groups['surge']['nights'])
        # 同一解释下呈现：普通夜风险不高于共同突发夜
        self.assertLessEqual(groups['ordinary']['point_rate'],
                             groups['surge']['point_rate'])

    def test_uncertainty_scenarios_are_monotone_bracket(self):
        result = evaluate_plan(self.events, self.calls, self.rosters, FLEX_FOUR, TARGETS,
                               replicates=100)
        counts = {key: value['breach_nights']
                  for key, value in result['scenarios'].items()}
        self.assertLessEqual(counts['observed-episode'], counts['observed-calls'])
        self.assertLessEqual(counts['observed-episode'], counts['imputed-episode'])
        self.assertLessEqual(counts['observed-calls'], counts['imputed-calls'])
        self.assertLessEqual(counts['imputed-episode'], counts['imputed-calls'])

    def test_call_merge_uncertainty_changes_reserve_judgment(self):
        # 服务结束后的重复呼叫：episode 口径不计新需求，calls 口径下成为第二次超时
        events = [
            {'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00',
             'end': '2026-11-01T22:40'},
            {'id': 'e2', 'episode': 'x2', 'resident': 'r2', 'skill': 'basic',
             'start': '2026-11-01T22:05', 'arrived': '2026-11-01T22:05',
             'end': '2026-11-01T22:25'},
            {'id': 'e3', 'episode': 'x3', 'resident': 'r3', 'skill': 'basic',
             'start': '2026-11-02T23:00', 'arrived': '2026-11-02T23:00',
             'end': '2026-11-02T23:20'},
        ]
        calls = [
            {'id': 'c1', 'episode': 'x1', 'time': '2026-11-01T22:00'},
            {'id': 'c2', 'episode': 'x2', 'time': '2026-11-01T22:05'},
            {'id': 'c3', 'episode': 'x2', 'time': '2026-11-01T22:26'},
            {'id': 'c4', 'episode': 'x3', 'time': '2026-11-02T23:00'},
        ]
        rosters = [
            {'id': f'r{night}', 'person': 'p0', 'night': night, 'skills': ['basic']}
            for night in ('2026-11-01', '2026-11-02')
        ]
        plan = {'id': 'single', 'workers': [{'skills': ['basic'], 'count': 1}]}
        result = evaluate_plan(events, calls, rosters, plan, {'basic': 10},
                               replicates=80, merge_gap=0)
        self.assertEqual(result['scenarios']['observed-episode']['total_breach_epochs'], 1)
        self.assertEqual(result['scenarios']['observed-calls']['total_breach_epochs'], 2)
        self.assertEqual(result['conservative_scenario'], 'imputed-calls')

    def test_open_end_imputation_changes_occupancy_judgment(self):
        # 开放结束事件在 observed 下不占后续人力，imputed 下按分位数延续占用并挤掉后续需求
        events = [
            {'id': 'e1', 'episode': 'x1', 'resident': 'r1', 'skill': 'basic',
             'start': '2026-11-01T22:00', 'arrived': '2026-11-01T22:00', 'end': None},
            {'id': 'e2', 'episode': 'x2', 'resident': 'r2', 'skill': 'basic',
             'start': '2026-11-01T22:05', 'arrived': '2026-11-01T22:05',
             'end': '2026-11-01T22:20'},
            {'id': 'e3', 'episode': 'x3', 'resident': 'r3', 'skill': 'basic',
             'start': '2026-11-02T23:00', 'arrived': '2026-11-02T23:00',
             'end': '2026-11-02T23:20'},
        ]
        calls = [
            {'id': 'c1', 'episode': 'x1', 'time': '2026-11-01T22:00'},
            {'id': 'c2', 'episode': 'x2', 'time': '2026-11-01T22:05'},
            {'id': 'c3', 'episode': 'x3', 'time': '2026-11-02T23:00'},
        ]
        rosters = [
            {'id': f'r{night}', 'person': 'p0', 'night': night, 'skills': ['basic']}
            for night in ('2026-11-01', '2026-11-02')
        ]
        plan = {'id': 'single', 'workers': [{'skills': ['basic'], 'count': 1}]}
        result = evaluate_plan(events, calls, rosters, plan, {'basic': 10}, replicates=80)
        observed = result['scenarios']['observed-episode']
        imputed = result['scenarios']['imputed-episode']
        # 观察口径把开放事件当作不占人，后续需求被挂起到窗口末端成为未服务；
        # 补全口径下开放事件持续占人，后续需求在第 15 分钟被派班但已超过 10 分钟目标
        self.assertGreater(observed['total_unserved'], 0)
        self.assertEqual(imputed['total_unserved'], 0)
        self.assertGreater(imputed['max_wait_minutes'], 10)
        # 两种口径结论不同，说明储备判断对开放结束假设确实敏感
        self.assertNotEqual(
            (observed['total_unserved'], observed['max_wait_minutes']),
            (imputed['total_unserved'], imputed['max_wait_minutes']))
        # 保守象限必须同时采用两种较不利假设
        self.assertEqual(result['conservative_scenario'], 'imputed-calls')

    def test_exact_bounds_and_bootstrap_cover_point_estimate(self):
        exact = binomial_bounds(5, 60)
        self.assertAlmostEqual(exact['lower'], 0.0272, places=3)
        self.assertGreater(exact['upper'], 5 / 60)
        self.assertGreater(exact['one_sided_upper'], 5 / 60)
        zero = binomial_bounds(0, 60)
        self.assertEqual(zero['lower'], 0.0)
        self.assertAlmostEqual(zero['one_sided_upper'], 0.0487, places=3)
        bootstrap = bootstrap_night_rate([0] * 47 + [1] * 13, replicates=500)
        self.assertLessEqual(bootstrap['lower'], bootstrap['point'] + 1e-12)
        self.assertGreaterEqual(bootstrap['upper'], bootstrap['point'] - 1e-12)

    def test_compare_plans_ranks_by_risk_then_headcount(self):
        comparison = compare_plans(self.events, self.calls, self.rosters,
                                   [BASELINE, FLEX_FOUR, SPECIALIST_FOUR], TARGETS,
                                   replicates=100)
        ranking = [row['plan_id'] for row in comparison['rankings']]
        self.assertEqual(set(ranking), {'plan-baseline', 'plan-flex-four', 'plan-specialist-four'})
        baseline = next(row for row in comparison['rankings']
                        if row['plan_id'] == 'plan-baseline')
        flex = next(row for row in comparison['rankings']
                    if row['plan_id'] == 'plan-flex-four')
        self.assertGreater(baseline['breach_nights'], flex['breach_nights'])
        # 双证与专项方案同为四人，冗余以逐分钟空闲人力度量而非资质人数相加
        self.assertEqual(baseline['headcount'], 3)
        self.assertEqual(flex['headcount'], 4)

    def test_evidence_does_not_claim_proven_low_risk(self):
        result = evaluate_plan(self.events, self.calls, self.rosters, FLEX_FOUR, TARGETS,
                               replicates=100)
        evidence = result['evidence']
        self.assertIn('worst_observed_nights', evidence)
        self.assertIsInstance(evidence['extrapolation_required'], bool)
        # 零失败的单侧上界约为 3/n，小样本不可能得到零上界
        self.assertGreater(evidence['zero_breach_one_sided_upper'], 0.0)

    def test_surge_threshold_uses_observed_peak_distribution(self):
        classification = classify_surge([
            {'night': f'n{i}', 'peak': i % 5, 'by_skill': {}} for i in range(20)
        ], quantile=0.8)
        self.assertTrue(classification['surge_nights'])
        self.assertTrue(classification['ordinary_nights'])
        self.assertLess(classification['threshold'], 5)


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
        cls.events = cls.data['events']
        cls.calls = cls.data['calls']
        cls.rosters = cls.data['rosters']

    def test_reconcile_labels_staffing_shortfall(self):
        short_roster = [row for row in self.rosters
                        if not (row['night'] == '2026-07-14'
                                and tuple(sorted(row['skills'])) == ('basic', 'transfer'))]
        record = reconcile_night(self.events, self.calls, short_roster, '2026-07-14',
                                 BASELINE, TARGETS, ASSUMPTION)
        self.assertIn(STAFFING_SHORTFALL, record['labels'])
        self.assertEqual(record['roster_check']['missing_headcount'], 1)
        self.assertIn('transfer', record['roster_check']['missing_skills'])

    def test_reconcile_distinguishes_allocation_and_capacity(self):
        # 历史最忙夜之一，基线方案在完整在岗下出现缺口
        record = reconcile_night(self.events, self.calls, self.rosters, '2026-07-14',
                                 BASELINE, TARGETS, ASSUMPTION)
        if record['actual_greedy']['breach_count'] > 0:
            self.assertTrue(
                CAPABILITY_ALLOCATION in record['labels']
                or CAPACITY_INSUFFICIENT in record['labels'])
        self.assertNotIn(STAFFING_SHORTFALL, record['labels'])

    def test_period_reconciliation_counts_multiple_causes(self):
        result = reconcile_period(self.events, self.calls, self.rosters,
                                  BASELINE, TARGETS, ASSUMPTION)
        self.assertEqual(result['nights'], 60)
        self.assertEqual(sum(1 for record in result['records'] if record['labels']),
                         len(result['breach_nights']))
        # 同一夜可带多个标签：标签计数之和允许超过缺口夜数
        self.assertGreaterEqual(sum(result['label_counts'].values()),
                                len(result['breach_nights']))
        # 假设编号随复盘结果保留，支持日后追溯
        self.assertEqual(result['assumptions_id'], ASSUMPTION['id'])
        for record in result['records']:
            self.assertEqual(record['assumptions_id'], ASSUMPTION['id'])

    def test_demand_side_label_uses_joint_envelope(self):
        # 在有完整当班记录的夜晚上叠加合成需求，制造超过历史联合包络（basic 峰值 10）的峰值
        night = '2026-08-29'
        synthetic_count = 25
        surge_events = list(self.events) + [
            {
                'id': f'synthetic-{i}', 'episode': f'synthetic-{i}',
                'resident': f'r{i}', 'skill': 'basic',
                'start': '2026-08-29T22:00', 'arrived': '2026-08-29T22:00',
                'end': '2026-08-29T22:40',
            }
            for i in range(synthetic_count)
        ]
        surge_calls = list(self.calls) + [
            {'id': f'synthetic-call-{i}', 'episode': f'synthetic-{i}',
             'time': '2026-08-29T22:00'}
            for i in range(synthetic_count)
        ]
        record = reconcile_night(surge_events, surge_calls, self.rosters, night,
                                 BASELINE, TARGETS, ASSUMPTION)
        self.assertIn(DEMAND_BEYOND_DISTRIBUTION, record['labels'])
        self.assertGreater(record['peak']['by_skill']['basic'],
                           record['historical_envelope']['basic'])


if __name__ == '__main__':
    unittest.main()
