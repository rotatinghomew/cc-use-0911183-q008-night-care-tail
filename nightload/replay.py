"""夜班队列重放。

在共同的分钟网格上模拟“需求进入 → 等待合格人员 → 到场占用 → 释放”的完整过程，
而不是分别计算各服务类别的极端负荷后相加。人员以其技能组合参与，一次只能承接一个服务，
跨技能容量不可重复使用。

提供两种派班口径：
* greedy：固定优先级的确定性派班，用于复现现场可执行的规则；
* feasible：在同一需求序列下搜索是否存在满足响应目标的可行派班，用于区分
  “总能力不足”与“能力分配方式造成的缺口”。
"""

from .timeline import WINDOW_MINUTES


def workers_from_roster(rosters, night):
    workers = []
    for row in sorted(rosters, key=lambda item: item['person']):
        if row['night'] != night:
            continue
        workers.append({'person': row['person'], 'skills': frozenset(row['skills'])})
    if not workers:
        raise ValueError(f'{night} 没有当班记录')
    return workers


def workers_from_plan(plan):
    workers = []
    for index, group in enumerate(plan.get('workers', [])):
        count = int(group.get('count', 1))
        if count <= 0:
            raise ValueError('候选人数必须为正')
        if not group.get('skills'):
            raise ValueError('候选人员必须至少具备一项资质')
        for offset in range(count):
            workers.append({
                'person': f"{plan['id']}-skill{index}-{offset}",
                'skills': frozenset(group['skills']),
            })
    if not workers:
        raise ValueError('候选方案没有人员')
    return workers


def workers_for(rosters, night, plan=None):
    return workers_from_plan(plan) if plan is not None else workers_from_roster(rosters, night)


def _skill_map(workers):
    return {item['person']: item['skills'] for item in workers}


def replay_greedy(timeline, workers, targets, window=WINDOW_MINUTES):
    """确定性贪心派班重放。

    targets: {skill: 响应目标分钟数}。等待超过目标即记一次未及时响应。
    人员选择顺序为“技能最少（更专一）者优先”，避免通用人员被普通任务占满。
    """
    skill_sets = _skill_map(workers)
    busy_until = {item['person']: 0 for item in workers}
    event_owner: dict[str, str] = {}
    pending: list[dict] = []
    breaches = []
    assignments = []
    never_arrived = []
    events = []
    epochs = sorted(timeline['epochs'], key=lambda item: (item['demand_minute'], item['key']))

    for minute in range(window):
        for epoch in epochs:
            if epoch['demand_minute'] == minute:
                if epoch['never_arrived']:
                    never_arrived.append(_never_arrived_record(epoch))
                else:
                    pending.append(dict(epoch))
        for person, until in list(busy_until.items()):
            if until <= minute:
                busy_until[person] = 0
        # 服务进行中的重复呼叫由同一事件的在场人员即时承接，不重新占用一个人、不产生等待
        retained = []
        for epoch in pending:
            owner = _active_owner(busy_until, epoch, minute, event_owner)
            if owner is not None:
                record = _assignment_record(epoch, minute, busy_until[owner], owner, 0, targets)
                record['repeat_in_service'] = True
                assignments.append(record)
                events.append({'minute': minute, 'type': 'repeat-in-service', **record})
            else:
                retained.append(epoch)
        pending = retained
        still_pending = []
        for epoch in pending:
            worker = _pick_worker(busy_until, skill_sets, epoch, minute)
            if worker is None:
                still_pending.append(epoch)
                continue
            release = _release_minute(epoch, minute, window)
            busy_until[worker] = release
            event_owner[epoch['event_id']] = worker
            wait = minute - epoch['demand_minute']
            record = _assignment_record(epoch, minute, release, worker, wait, targets)
            assignments.append(record)
            if _is_breach(record):
                breaches.append(record)
            events.append({'minute': minute, 'type': 'assign', **record})
        pending = still_pending

    for epoch in pending:
        record = {
            'epoch': epoch['key'], 'event_id': epoch['event_id'], 'episode': epoch['episode'],
            'resident': epoch['resident'], 'skill': epoch['skill'],
            'demand_minute': epoch['demand_minute'], 'assigned_minute': None,
            'wait_minutes': window - epoch['demand_minute'], 'release_minute': None,
            'worker': None, 'target_minutes': targets.get(epoch['skill']),
            'repeat': epoch['repeat'], 'source': epoch['source'], 'unserved': True,
        }
        breaches.append(record)
        events.append({'minute': window, 'type': 'unserved', **record})

    return _summary(timeline, workers, assignments, breaches, never_arrived, events,
                    mode='greedy', window=window)


def replay_feasible(timeline, workers, targets, window=WINDOW_MINUTES, node_budget=200_000):
    """搜索满足全部响应目标的可行派班；不可行时回退到贪心结果并标记。

    搜索按需求顺序推进，每个需求可在目标时限内派给任一空闲合格人员，或推迟到后续时点。
    一旦某需求等待超过目标即剪枝。搜索有节点预算，超预算时回退并在结果中说明，
    不把超时搜索伪装成“不可行”。
    """
    epochs = [item for item in sorted(timeline['epochs'], key=lambda item: (item['demand_minute'], item['key']))
              if not item['never_arrived']]
    never_arrived = [_never_arrived_record(item) for item in timeline['epochs'] if item['never_arrived']]
    worker_ids = [item['person'] for item in workers]
    skill_sets = _skill_map(workers)
    decision_minutes = sorted({item['demand_minute'] for item in epochs} | {window})

    nodes = 0
    memo: set = set()

    def search(index, busy_until, chosen, event_owner):
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return None
        if index >= len(epochs):
            return list(chosen)
        epoch = epochs[index]
        # 重复呼叫只能由同一事件的在场人员即时承接；若该人员此刻已不在服务，
        # 此路径无解，必须剪枝而不是转去占用其他人
        if epoch['repeat']:
            owner = event_owner.get(epoch['event_id'])
            if owner is None or busy_until[owner] <= epoch['demand_minute']:
                return None
            chosen.append((epoch, epoch['demand_minute'], owner, busy_until[owner], 0))
            answer = search(index + 1, busy_until, chosen, event_owner)
            if answer is not None:
                return answer
            chosen.pop()
            return None
        for candidate in decision_minutes:
            if candidate < epoch['demand_minute'] or candidate >= window:
                continue
            wait = candidate - epoch['demand_minute']
            target = targets.get(epoch['skill'])
            if target is not None and wait > target:
                break
            available = [person for person in worker_ids
                         if busy_until[person] <= candidate and epoch['skill'] in skill_sets[person]]
            marker = (index, candidate, tuple(sorted(busy_until.items())))
            if not available or marker in memo:
                continue
            for person in sorted(available, key=lambda value: (len(skill_sets[value]), value)):
                release = _release_minute(epoch, candidate, window)
                next_busy = dict(busy_until)
                next_busy[person] = release
                next_owner = dict(event_owner)
                next_owner[epoch['event_id']] = person
                chosen.append((epoch, candidate, person, release, wait))
                answer = search(index + 1, next_busy, chosen, next_owner)
                if answer is not None:
                    return answer
                chosen.pop()
            memo.add(marker)
        return None

    found = search(0, {person: 0 for person in worker_ids}, [], {})
    if found is None:
        fallback = replay_greedy(timeline, workers, targets, window)
        fallback['mode'] = 'feasible'
        fallback['feasible'] = False
        fallback['search_nodes'] = nodes
        fallback['search_budget_exhausted'] = nodes > node_budget
        return fallback

    assignments = []
    breaches = []
    events = []
    for epoch, assigned_minute, person, release, wait in found:
        record = _assignment_record(epoch, assigned_minute, release, person, wait, targets)
        if epoch['repeat'] and wait == 0:
            record['repeat_in_service'] = True
        assignments.append(record)
        if _is_breach(record):
            breaches.append(record)
        events.append({'minute': assigned_minute,
                       'type': 'repeat-in-service' if record.get('repeat_in_service') else 'assign',
                       **record})
    summary = _summary(timeline, workers, assignments, breaches, never_arrived, events,
                       mode='feasible', window=window)
    summary['feasible'] = True
    summary['search_nodes'] = nodes
    summary['search_budget_exhausted'] = False
    return summary


def _pick_worker(busy_until, skill_sets, epoch, minute):
    candidates = [person for person, until in busy_until.items()
                  if until <= minute and epoch['skill'] in skill_sets[person]]
    candidates.sort(key=lambda person: (len(skill_sets[person]), person))
    return candidates[0] if candidates else None


def _active_owner(busy_until, epoch, minute, event_owner):
    """若同一事件已有人员正在服务，则该人员即时承接重复呼叫。"""
    if not epoch['repeat']:
        return None
    owner = event_owner.get(epoch['event_id'])
    if owner is not None and busy_until.get(owner, 0) > minute:
        return owner
    return None


def _release_minute(epoch, assigned_minute, window):
    if epoch['observed_release'] is not None:
        return max(assigned_minute, min(epoch['observed_release'], window))
    return min(window, assigned_minute + max(1, epoch['duration_minutes']))


def _assignment_record(epoch, assigned_minute, release, worker, wait, targets):
    return {
        'epoch': epoch['key'], 'event_id': epoch['event_id'], 'episode': epoch['episode'],
        'resident': epoch['resident'], 'skill': epoch['skill'],
        'demand_minute': epoch['demand_minute'], 'assigned_minute': assigned_minute,
        'wait_minutes': wait, 'release_minute': release, 'worker': worker,
        'target_minutes': targets.get(epoch['skill']), 'repeat': epoch['repeat'],
        'source': epoch['source'],
    }


def _is_breach(record):
    return record['target_minutes'] is not None and record['wait_minutes'] > record['target_minutes']


def _never_arrived_record(epoch):
    return {
        'epoch': epoch['key'], 'event_id': epoch['event_id'], 'episode': epoch['episode'],
        'resident': epoch['resident'], 'skill': epoch['skill'],
        'demand_minute': epoch['demand_minute'], 'reason': 'never_arrived',
    }


def _occupied_grid(assignments, window):
    occupied = [0] * window
    for record in assignments:
        end = record['release_minute'] if record['release_minute'] is not None else window
        for minute in range(record['assigned_minute'], min(end, window)):
            occupied[minute] += 1
    return occupied


def _summary(timeline, workers, assignments, breaches, never_arrived, events, *, mode, window):
    occupied = _occupied_grid(assignments, window)
    idle_person_minutes = sum(len(workers) - value for value in occupied)
    peak_slack = min(len(workers) - value for value in occupied) if assignments else len(workers)
    waits = [item['wait_minutes'] for item in assignments]
    return {
        'mode': mode,
        'night': timeline['night'],
        'options': timeline['options'],
        'workers': [{'person': item['person'], 'skills': sorted(item['skills'])} for item in workers],
        'headcount': len(workers),
        'demand_epochs': len(timeline['epochs']),
        'assignments': assignments,
        'breaches': breaches,
        'never_arrived': never_arrived,
        'events': events,
        'metrics': {
            'breach_count': len(breaches),
            'unserved_count': sum(1 for item in breaches if item.get('unserved')),
            'breach_rate_epochs': len(breaches) / len(timeline['epochs']) if timeline['epochs'] else 0.0,
            'max_wait_minutes': max(waits, default=0),
            'mean_wait_minutes': sum(waits) / len(waits) if waits else 0.0,
            'idle_person_minutes': idle_person_minutes,
            'peak_slack_workers': peak_slack,
            'occupancy_person_minutes': sum(occupied),
        },
    }
