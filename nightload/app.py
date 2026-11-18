from pathlib import Path
import json

from .attribution import reconcile_night, reconcile_period
from .contingency import compare_plans, evaluate_plan
from .events import *
from .load import *
from .replay import replay_feasible, replay_greedy, workers_from_roster
from .roster import *
from .timeline import build_timeline, peak_composition


class Application:
    def __init__(self, store):
        self.store = store

    def seed(self):
        directory = Path(__file__).resolve().parent.parent / 'fixtures'
        for path in sorted(directory.glob('*.json')):
            if not self.store.all(path.stem):
                self.store.put_many(path.stem, json.loads(path.read_text(encoding='utf-8')))

    def demo(self):
        return {'load': descriptive_load(self.store, '2026-10-01T22:00', '2026-10-02T06:00'), 'roster': skills_on_duty(self.store, '2026-10-01')}

    def check(self):
        value = self.demo()
        return {'status': 'ok', 'summary': value}

    def _records(self, kind):
        return self.store.all(kind)

    def _targets(self):
        targets = {}
        for row in self._records('targets'):
            targets[row['id']] = row['response_minutes']
        if not targets:
            raise ValueError('缺少服务响应目标记录')
        return targets

    def _assumption(self, assumption_id):
        for row in self._records('assumptions'):
            if row['id'] == assumption_id:
                return row
        raise KeyError(f'没有找到负荷假设 {assumption_id}')

    def _plan(self, plan_id):
        for row in self._records('plans'):
            if row['id'] == plan_id:
                return row
        raise KeyError(f'没有找到候选方案 {plan_id}')

    def night_timeline(self, night, open_end_policy='observed', call_policy='episode', merge_gap=5):
        timeline = build_timeline(
            self._records('events'), self._records('calls'), night,
            open_end_policy=open_end_policy, call_policy=call_policy, merge_gap=int(merge_gap),
        )
        timeline['peak'] = peak_composition(timeline)
        return timeline

    def night_replay(self, night, mode='greedy', open_end_policy='observed',
                     call_policy='episode', merge_gap=5):
        if mode not in ('greedy', 'feasible'):
            raise ValueError('mode 只能是 greedy 或 feasible')
        timeline = build_timeline(
            self._records('events'), self._records('calls'), night,
            open_end_policy=open_end_policy, call_policy=call_policy, merge_gap=int(merge_gap),
        )
        workers = workers_from_roster(self._records('rosters'), night)
        runner = replay_feasible if mode == 'feasible' else replay_greedy
        return runner(timeline, workers, self._targets())

    def plan_evaluation(self, payload):
        plan_ids = payload.get('plan_ids')
        plans = self._records('plans')
        if plan_ids:
            wanted = set(plan_ids)
            plans = [row for row in plans if row['id'] in wanted]
            if len(plans) != len(wanted):
                raise KeyError('部分候选方案不存在')
        if not plans:
            raise ValueError('没有可评估的候选方案')
        kwargs = {
            'nights': payload.get('nights'),
            'mode': payload.get('mode', 'greedy'),
            'replicates': int(payload.get('replicates', 1000)),
            'merge_gap': int(payload.get('merge_gap', 5)),
            'impute_quantile': float(payload.get('impute_quantile', 0.9)),
            'confidence': float(payload.get('confidence', 0.95)),
            'surge_quantile': float(payload.get('surge_quantile', 0.8)),
        }
        if kwargs['mode'] not in ('greedy', 'feasible'):
            raise ValueError('mode 只能是 greedy 或 feasible')
        if not 0 < kwargs['confidence'] < 1:
            raise ValueError('confidence 必须位于 0 与 1 之间')
        if len(plans) == 1:
            return evaluate_plan(self._records('events'), self._records('calls'),
                                 self._records('rosters'), plans[0], self._targets(), **kwargs)
        return compare_plans(self._records('events'), self._records('calls'),
                             self._records('rosters'), plans, self._targets(), **kwargs)

    def night_reconciliation(self, payload):
        night = payload['night']
        plan = self._plan(payload['plan_id'])
        assumption = self._assumption(payload.get('assumption_id', 'assumption-night-v1'))
        result = reconcile_night(
            self._records('events'), self._records('calls'), self._records('rosters'),
            night, plan, self._targets(), assumption,
            merge_gap=int(assumption.get('merge_gap_minutes', 5)),
            impute_quantile=float(assumption.get('impute_quantile', 0.9)),
        )
        saved = dict(result)
        saved['id'] = f"recon-{night}-{plan['id']}"
        self.store.put('reconciliations', saved)
        return {'revision': self.store.revision('reconciliations', saved['id']), 'reconciliation': result}

    def period_reconciliation(self, payload):
        plan = self._plan(payload['plan_id'])
        assumption = self._assumption(payload.get('assumption_id', 'assumption-night-v1'))
        result = reconcile_period(
            self._records('events'), self._records('calls'), self._records('rosters'),
            plan, self._targets(), assumption,
            nights=payload.get('nights'),
            merge_gap=int(assumption.get('merge_gap_minutes', 5)),
            impute_quantile=float(assumption.get('impute_quantile', 0.9)),
        )
        return result

    def call(self, method, route, payload):
        if method == 'GET' and route == 'summary':
            return self.demo()
        if method == 'POST' and route == 'event':
            return {'revision': save_event(self.store, payload)}
        if method == 'POST' and route == 'roster':
            return {'revision': save_roster(self.store, payload)}
        if method == 'GET' and route == 'load':
            return descriptive_load(self.store, payload['start'], payload['end'])
        if method == 'GET' and route == 'timeline':
            return self.night_timeline(
                payload['night'], payload.get('open_end_policy', 'observed'),
                payload.get('call_policy', 'episode'), payload.get('merge_gap', 5))
        if method == 'GET' and route == 'replay':
            return self.night_replay(
                payload['night'], payload.get('mode', 'greedy'),
                payload.get('open_end_policy', 'observed'),
                payload.get('call_policy', 'episode'), payload.get('merge_gap', 5))
        if method == 'POST' and route == 'evaluate':
            return self.plan_evaluation(payload)
        if method == 'POST' and route == 'reconcile-night':
            return self.night_reconciliation(payload)
        if method == 'POST' and route == 'reconcile-period':
            return self.period_reconciliation(payload)
        raise KeyError('接口不存在')
