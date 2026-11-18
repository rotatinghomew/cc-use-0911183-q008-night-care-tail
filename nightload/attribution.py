"""当班复盘与缺口归因。

复盘不把所有未达标都归因于需求增长，而是对每个夜班分别检验三条独立原因：

* demand_beyond_distribution：当夜联合占用峰值超出历史共同占用包络（分布外需求）；
* staffing_shortfall：实际当班人数或技能组合低于已采用方案（未按计划在岗）；
* capability_allocation：人数与资质满足方案，但贪心派班仍出现超时，
  而最优可行性搜索能找到达标派班（能力分配问题），或搜索确认无解（能力总量不足）。

评估所用的负荷假设（窗口、开放结束/呼叫口径、分位数、响应目标、方案版本）
随复盘记录一起保存，保证日后可追溯。
"""

from .contingency import (CONSERVATIVE_SCENARIO, observed_envelope,
                          plan_skill_capacity, validate_plan)
from .replay import replay_feasible, replay_greedy, workers_from_plan, workers_from_roster
from .timeline import build_timeline, collect_nights, compute_duration_quantiles, peak_composition

DEMAND_BEYOND_DISTRIBUTION = 'demand_beyond_distribution'
STAFFING_SHORTFALL = 'staffing_shortfall'
CAPABILITY_ALLOCATION = 'capability_allocation'
CAPACITY_INSUFFICIENT = 'capacity_insufficient'


def actual_vs_plan_roster(rosters, night, plan):
    actual = [row for row in rosters if row['night'] == night]
    actual_workers = workers_from_roster(rosters, night) if actual else []
    planned_workers = workers_from_plan(plan)
    actual_capacity: dict[str, int] = {}
    for worker in actual_workers:
        for skill in worker['skills']:
            actual_capacity[skill] = actual_capacity.get(skill, 0) + 1
    planned_capacity = plan_skill_capacity(plan)
    missing_skills = {
        skill: planned - actual_capacity.get(skill, 0)
        for skill, planned in planned_capacity.items()
        if actual_capacity.get(skill, 0) < planned
    }
    return {
        'planned_headcount': len(planned_workers),
        'actual_headcount': len(actual_workers),
        'planned_capacity_by_skill': planned_capacity,
        'actual_capacity_by_skill': dict(sorted(actual_capacity.items())),
        'missing_headcount': max(0, len(planned_workers) - len(actual_workers)),
        'missing_skills': missing_skills,
        'staffing_shortfall': bool(missing_skills) or len(actual_workers) < len(planned_workers),
    }


def reconcile_night(events, calls, rosters, night, plan, targets, assumptions,
                    *, merge_gap=5, impute_quantile=0.9, quantiles=None):
    """对单个夜班执行实际派班重放、最优可行性对照和归因。"""
    quantiles = quantiles or compute_duration_quantiles(events, impute_quantile)
    scenario = assumptions.get('scenario', {'id': CONSERVATIVE_SCENARIO,
                                            'open_end_policy': 'imputed',
                                            'call_policy': 'calls'})
    timeline = build_timeline(
        events, calls, night,
        open_end_policy=scenario['open_end_policy'], call_policy=scenario['call_policy'],
        merge_gap=merge_gap, impute_quantile=impute_quantile, duration_quantiles=quantiles,
    )
    actual_workers = workers_from_roster(rosters, night)
    greedy = replay_greedy(timeline, actual_workers, targets)
    feasible = replay_feasible(timeline, actual_workers, targets)

    nights = collect_nights(events)
    # 历史包络只由复盘夜之外的观察夜晚构成：当夜需求超出该基线才属于分布外变化，
    # 否则分布外标签会被当夜自身数据掩盖
    baseline_nights = [item for item in nights if item != night]
    envelope, _ = observed_envelope(events, calls, baseline_nights, quantiles,
                                    merge_gap=merge_gap, impute_quantile=impute_quantile)
    peak = peak_composition(timeline)
    beyond = {skill: {'observed_peak': value, 'historical_envelope': envelope.get(skill, 0)}
              for skill, value in peak['by_skill'].items() if value > envelope.get(skill, 0)}
    roster_check = actual_vs_plan_roster(rosters, night, plan)

    labels = []
    if beyond:
        labels.append(DEMAND_BEYOND_DISTRIBUTION)
    if roster_check['staffing_shortfall']:
        labels.append(STAFFING_SHORTFALL)
    if greedy['metrics']['breach_count'] > 0:
        if feasible.get('feasible'):
            labels.append(CAPABILITY_ALLOCATION)
        else:
            labels.append(CAPACITY_INSUFFICIENT)

    breach_breakdown = []
    for breach in greedy['breaches']:
        reasons = list(labels)
        if breach.get('unserved'):
            reasons = reasons + ['unserved_within_window']
        breach_breakdown.append({
            'epoch': breach['epoch'], 'event_id': breach['event_id'], 'skill': breach['skill'],
            'wait_minutes': breach['wait_minutes'], 'target_minutes': breach['target_minutes'],
            'worker': breach.get('worker'), 'reasons': reasons,
        })

    return {
        'night': night,
        'plan_id': plan['id'],
        'assumptions_id': assumptions.get('id'),
        'scenario': scenario['id'],
        'targets': targets,
        'roster_check': roster_check,
        'peak': peak,
        'historical_envelope': envelope,
        'demand_beyond_distribution': beyond,
        'actual_greedy': {
            'breach_count': greedy['metrics']['breach_count'],
            'unserved_count': greedy['metrics']['unserved_count'],
            'max_wait_minutes': greedy['metrics']['max_wait_minutes'],
            'peak_slack_workers': greedy['metrics']['peak_slack_workers'],
        },
        'feasibility': {
            'feasible': feasible.get('feasible', False),
            'search_budget_exhausted': feasible.get('search_budget_exhausted', False),
            'breach_count': feasible['metrics']['breach_count'],
        },
        'labels': labels,
        'breach_breakdown': breach_breakdown,
        'open_end_event_ids': timeline['open_end_event_ids'],
        'censored_event_ids': timeline['censored_event_ids'],
        'imputed_event_ids': timeline['imputed_event_ids'],
    }


def reconcile_period(events, calls, rosters, plan, targets, assumptions, *, nights=None, **kwargs):
    validate_plan(plan, {row['skill'] for row in events})
    nights = nights or collect_nights(events)
    quantiles = compute_duration_quantiles(events, kwargs.get('impute_quantile', 0.9))
    records = [reconcile_night(events, calls, rosters, night, plan, targets, assumptions,
                               quantiles=quantiles, **kwargs)
               for night in nights]
    counts = {label: sum(label in record['labels'] for record in records)
              for label in (DEMAND_BEYOND_DISTRIBUTION, STAFFING_SHORTFALL,
                            CAPABILITY_ALLOCATION, CAPACITY_INSUFFICIENT)}
    breach_nights = [record['night'] for record in records if record['labels']]
    return {
        'plan_id': plan['id'],
        'assumptions_id': assumptions.get('id'),
        'nights': len(records),
        'breach_nights': breach_nights,
        'label_counts': counts,
        'records': records,
        'interpretation': '同一夜可同时带有多个归因标签；标签互斥性不做人为假定',
    }
