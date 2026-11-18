"""应急储备候选方案的共同发生评估。

核心立场：每个夜晚是一个不可拆分的联合样本——同夜各类需求的共同发生关系、
人员接连参与不同服务的占用关系都在时间线重放中保留，既不把各类别极端负荷分别相加，
也不假定服务类别彼此独立。跨夜不确定性用整夜重采样表达，而不是对单事件做独立抽样。

四个口径情景构成不确定性栅格，储备结论取保守边界：
* observed/episode：只承认窗口内观察到的占用，一场事件计一次需求；
* observed/calls：重复呼叫各自成为需求尝试（按合并间隔归并）；
* imputed/episode：缺失结束按分位数补全；
* imputed/calls：两种不确定性同时取较不利口径。
"""

import numpy as np
from scipy.stats import beta, binomtest

from .replay import replay_feasible, replay_greedy, workers_for
from .timeline import (build_timeline, collect_nights,
                       compute_duration_quantiles, peak_composition)

SCENARIOS = (
    {'id': 'observed-episode', 'open_end_policy': 'observed', 'call_policy': 'episode'},
    {'id': 'observed-calls', 'open_end_policy': 'observed', 'call_policy': 'calls'},
    {'id': 'imputed-episode', 'open_end_policy': 'imputed', 'call_policy': 'episode'},
    {'id': 'imputed-calls', 'open_end_policy': 'imputed', 'call_policy': 'calls'},
)
CONSERVATIVE_SCENARIO = 'imputed-calls'


def validate_plan(plan, known_skills):
    if not isinstance(plan, dict) or not plan.get('id'):
        raise ValueError('候选方案必须包含 id')
    workers = plan.get('workers')
    if not isinstance(workers, list) or not workers:
        raise ValueError('候选方案必须包含非空 workers 列表')
    seen = set()
    for group in workers:
        skills = group.get('skills')
        if not isinstance(skills, list) or not skills or len(set(skills)) != len(skills):
            raise ValueError('每组人员的技能列表不能为空且不能重复')
        unknown = set(skills) - set(known_skills)
        if unknown:
            raise ValueError(f"候选方案包含未知资质：{','.join(sorted(unknown))}")
        count = int(group.get('count', 1))
        if count <= 0:
            raise ValueError('每组人数必须为正')
        marker = tuple(sorted(skills))
        if marker in seen:
            raise ValueError('候选方案中技能组合重复，请合并为同一组')
        seen.add(marker)
    return True


def plan_headcount(plan):
    return sum(int(group.get('count', 1)) for group in plan['workers'])


def plan_skill_capacity(plan):
    """返回各资质的持证人数；同一人出现在多个资质中，调用方不得跨资质相加。"""
    capacity: dict[str, int] = {}
    for group in plan['workers']:
        for skill in group['skills']:
            capacity[skill] = capacity.get(skill, 0) + int(group.get('count', 1))
    return dict(sorted(capacity.items()))


def evaluate_night(events, calls, night, plan, targets, scenario, quantiles,
                   rosters=None, mode='greedy', merge_gap=5, impute_quantile=0.9):
    timeline = build_timeline(
        events, calls, night,
        open_end_policy=scenario['open_end_policy'], call_policy=scenario['call_policy'],
        merge_gap=merge_gap, impute_quantile=impute_quantile, duration_quantiles=quantiles,
    )
    workers = workers_for(rosters or [], night, plan)
    if mode == 'feasible':
        result = replay_feasible(timeline, workers, targets)
    else:
        result = replay_greedy(timeline, workers, targets)
    result['scenario'] = scenario['id']
    result['peak'] = peak_composition(timeline)
    return result, timeline


def observed_envelope(events, calls, nights, quantiles, merge_gap=5, impute_quantile=0.9):
    """历史上真正出现过的联合占用边界（按技能同时计数，不做类别相加）。"""
    envelope = {}
    night_peaks = []
    for night in nights:
        timeline = build_timeline(events, calls, night, duration_quantiles=quantiles,
                                  merge_gap=merge_gap, impute_quantile=impute_quantile)
        peak = peak_composition(timeline)
        night_peaks.append({'night': night, 'peak': peak['peak'], 'by_skill': peak['by_skill']})
        for skill, value in peak['by_skill'].items():
            envelope[skill] = max(envelope.get(skill, 0), value)
    return dict(sorted(envelope.items())), night_peaks


def classify_surge(night_peaks, quantile=0.8):
    """以历史峰值分布的分位数为阈，把夜班划分为共同突发夜与普通夜。"""
    values = sorted(item['peak'] for item in night_peaks)
    if not values:
        return {'threshold': 0, 'surge_nights': [], 'ordinary_nights': []}
    threshold = float(np.quantile(values, quantile))
    surge = [item['night'] for item in night_peaks if item['peak'] >= threshold]
    ordinary = [item['night'] for item in night_peaks if item['peak'] < threshold]
    return {'threshold': threshold, 'surge_nights': surge, 'ordinary_nights': ordinary}


def binomial_bounds(failures, trials, confidence=0.95):
    """Clopper–Pearson 双侧精确区间（针对失败比例）；零失败时额外给出单侧 1-α 风险上界。"""
    if trials == 0:
        return {'lower': None, 'upper': None, 'one_sided_upper': None}
    successes = trials - failures
    alpha = 1 - confidence
    lower = float(beta.ppf(alpha / 2, failures, successes + 1)) if failures > 0 else 0.0
    upper = float(beta.ppf(1 - alpha / 2, failures + 1, successes)) if failures < trials else 1.0
    one_sided = float(beta.ppf(1 - alpha, failures + 1, successes)) if failures < trials else 1.0
    return {'lower': lower, 'upper': upper, 'one_sided_upper': one_sided}


def bootstrap_night_rate(breach_flags, confidence=0.95, replicates=1000, seed=20261118):
    """整夜重采样：每个重采样样本抽取完整夜晚，保留夜内全部共同发生结构。"""
    rng = np.random.default_rng(seed)
    flags = np.asarray(breach_flags, dtype=bool)
    n = flags.size
    if n == 0:
        return {'lower': None, 'upper': None, 'point': None}
    draws = rng.integers(0, n, size=(replicates, n))
    rates = flags[draws].mean(axis=1)
    alpha = 1 - confidence
    return {
        'lower': float(np.quantile(rates, alpha / 2)),
        'upper': float(np.quantile(rates, 1 - alpha / 2)),
        'point': float(flags.mean()),
    }


def evaluate_plan(events, calls, rosters, plan, targets, *, nights=None,
                  scenarios=SCENARIOS, mode='greedy', confidence=0.95,
                  replicates=1000, merge_gap=5, impute_quantile=0.9,
                  surge_quantile=0.8, seed=20261118):
    known_skills = sorted({row['skill'] for row in events} | set(targets))
    validate_plan(plan, known_skills)
    nights = nights or collect_nights(events)
    quantiles = compute_duration_quantiles(events, impute_quantile)
    envelope, night_peaks = observed_envelope(events, calls, nights, quantiles,
                                              merge_gap=merge_gap, impute_quantile=impute_quantile)
    surge = classify_surge(night_peaks, surge_quantile)
    capacity = plan_skill_capacity(plan)

    per_scenario = {}
    for scenario in scenarios:
        night_results = []
        for night in nights:
            result, _ = evaluate_night(events, calls, night, plan, targets, scenario, quantiles,
                                       rosters=rosters, mode=mode, merge_gap=merge_gap,
                                       impute_quantile=impute_quantile)
            night_results.append(result)
        flags = [item['metrics']['breach_count'] > 0 for item in night_results]
        trials = len(flags)
        failures = sum(flags)
        bootstrap = bootstrap_night_rate(flags, confidence, replicates, seed)
        exact = binomial_bounds(failures, trials, confidence)
        groups = {}
        for label, members in (('all', nights), ('surge', surge['surge_nights']),
                               ('ordinary', surge['ordinary_nights'])):
            member_set = set(members)
            sub = [item for item in night_results if item['night'] in member_set]
            sub_flags = [item['metrics']['breach_count'] > 0 for item in sub]
            sub_trials = len(sub_flags)
            sub_failures = sum(sub_flags)
            groups[label] = {
                'nights': sub_trials,
                'breach_nights': sub_failures,
                'point_rate': (sub_failures / sub_trials) if sub_trials else None,
                'exact': binomial_bounds(sub_failures, sub_trials, confidence),
                'mean_peak_slack_workers': float(np.mean([
                    item['metrics']['peak_slack_workers'] for item in sub])) if sub else None,
                'mean_idle_person_minutes': float(np.mean([
                    item['metrics']['idle_person_minutes'] for item in sub])) if sub else None,
            }
        per_scenario[scenario['id']] = {
            'scenario': scenario['id'],
            'night_results': night_results,
            'breach_nights': failures,
            'nights': trials,
            'point_rate': failures / trials,
            'bootstrap': bootstrap,
            'exact': exact,
            'groups': groups,
            'total_breach_epochs': sum(item['metrics']['breach_count'] for item in night_results),
            'total_unserved': sum(item['metrics']['unserved_count'] for item in night_results),
            'max_wait_minutes': max((item['metrics']['max_wait_minutes'] for item in night_results), default=0),
        }

    conservative = per_scenario[CONSERVATIVE_SCENARIO]
    # 证据边界：方案只有在历史最忙的共同突发夜（按联合峰值取前三）也能达标，
    # 才说明其结论由观察样本支持；否则属于向未覆盖组合外推。
    worst_nights = [item['night'] for item in
                    sorted(night_peaks, key=lambda item: item['peak'], reverse=True)[:3]]
    worst_results = [item for item in conservative['night_results'] if item['night'] in set(worst_nights)]
    supported = bool(worst_results) and all(item['metrics']['breach_count'] == 0 for item in worst_results)
    sparse_skill_notes = [
        f"技能 {skill} 历史联合峰值仅 {value} 人次同时占用，样本覆盖有限"
        for skill, value in envelope.items() if value <= 1
    ]
    evidence = {
        'nights_observed': len(nights),
        'observed_envelope': envelope,
        'plan_capacity_by_skill': capacity,
        'worst_observed_nights': worst_nights,
        'supported_on_worst_nights': supported,
        'extrapolation_required': not supported,
        'zero_breach_one_sided_upper': conservative['exact']['one_sided_upper'],
        'coverage_notes': sparse_skill_notes,
        'support_note': (
            '方案在历史最忙共同突发夜仍出现未及时响应，风险结论含向未覆盖组合的外推，不能声称已证明极低风险'
            if not supported else
            '方案在历史最忙共同突发夜可达标，样本内边界由观察夜晚支持；更极端组合仍属未覆盖区域'
        ),
    }
    return {
        'plan_id': plan['id'],
        'mode': mode,
        'headcount': plan_headcount(plan),
        'workers': plan['workers'],
        'targets': targets,
        'surge_classification': surge,
        'scenarios': per_scenario,
        'conservative_scenario': CONSERVATIVE_SCENARIO,
        'conservative': {
            'breach_nights': conservative['breach_nights'],
            'point_rate': conservative['point_rate'],
            'bootstrap_ci': conservative['bootstrap'],
            'exact_ci': conservative['exact'],
            'groups': conservative['groups'],
            'max_wait_minutes': conservative['max_wait_minutes'],
        },
        'evidence': evidence,
    }


def compare_plans(events, calls, rosters, plans, targets, **kwargs):
    rankings = []
    details = {}
    for plan in plans:
        evaluation = evaluate_plan(events, calls, rosters, plan, targets, **kwargs)
        details[plan['id']] = evaluation
        conservative = evaluation['conservative']
        rankings.append({
            'plan_id': plan['id'],
            'headcount': evaluation['headcount'],
            'capacity_by_skill': evaluation['evidence']['plan_capacity_by_skill'],
            'breach_nights': conservative['breach_nights'],
            'point_rate': conservative['point_rate'],
            'rate_upper_bootstrap': conservative['bootstrap_ci']['upper'],
            'rate_upper_exact': conservative['exact_ci']['upper'],
            'one_sided_upper_zero_failures': conservative['exact_ci']['one_sided_upper'],
            'surge_breach_rate': conservative['groups']['surge']['point_rate'],
            'ordinary_breach_rate': conservative['groups']['ordinary']['point_rate'],
            'mean_peak_slack_workers': conservative['groups']['all']['mean_peak_slack_workers'],
            'mean_idle_person_minutes': conservative['groups']['all']['mean_idle_person_minutes'],
            'extrapolation_required': evaluation['evidence']['extrapolation_required'],
        })
    rankings.sort(key=lambda item: (item['rate_upper_exact'], item['headcount']))
    return {'rankings': rankings, 'details': details}


def binomial_test_difference(flags_a, flags_b):
    """两个方案在同一组夜晚的配对失败数比较（McNemar 风格，仅作证据强度提示）。"""
    a = np.asarray(flags_a, dtype=bool)
    b = np.asarray(flags_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('两个方案必须在同一组夜晚上比较')
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(b & ~a))
    if a_only + b_only == 0:
        return {'discordant_nights': 0, 'p_value': 1.0}
    result = binomtest(min(a_only, b_only), n=a_only + b_only, p=0.5)
    return {'discordant_nights': a_only + b_only, 'plan_a_only': a_only,
            'plan_b_only': b_only, 'p_value': float(result.pvalue)}
