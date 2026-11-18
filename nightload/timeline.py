"""夜班时间线重建。

把事件、呼叫还原到同一个夜晚窗口（默认 22:00 至次日 06:00），保留以下事实：

* 需求发生（start/呼叫）、人员到场（arrived）、实际占用（arrived→end）是不同时间；
* 一场事件可产生多次呼叫，呼叫是否合并为同一需求存在口径不确定性；
* 缺失结束时间的事件不能被当成短时完成，只能按观察截断或分位数补全两种口径处理；
* 观察窗口之外的前后片段以标记形式保留，不会悄悄丢弃；
* 各服务类别的占用在同一分钟网格上共同计数，不做跨类别相加或独立性假设。
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

WINDOW_START_HOUR = 22
WINDOW_END_HOUR = 6
WINDOW_MINUTES = (24 - WINDOW_START_HOUR + WINDOW_END_HOUR) * 60


def night_bounds(night):
    """返回某个夜班标签对应的窗口 [开始, 结束)。"""
    day = date.fromisoformat(night)
    start = datetime.combine(day, time(WINDOW_START_HOUR, 0))
    end = datetime.combine(day + timedelta(days=1), time(WINDOW_END_HOUR, 0))
    return start, end


def night_of(value):
    """把任意时刻归入其所属夜班标签（0 点至 12 点属于前一天的夜班）。"""
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    assigned = moment.date() if moment.hour >= 12 else moment.date() - timedelta(days=1)
    return assigned.isoformat()


def collect_nights(rows, time_field='start'):
    return sorted({night_of(row[time_field]) for row in rows if row.get(time_field)})


def _minute(value, origin):
    return int((datetime.fromisoformat(value) - origin).total_seconds() // 60)


@dataclass
class Interval:
    event_id: str
    episode: str
    resident: str
    skill: str
    start_minute: int
    arrived_minute: int | None
    release_minute: int | None
    actual_start_minute: int
    actual_arrived_minute: int | None
    actual_release_minute: int | None
    open_end: bool
    censored: bool
    carry_in: bool
    imputed: bool


@dataclass
class Epoch:
    """一次需要派人的需求时点。"""
    key: str
    event_id: str
    episode: str
    resident: str
    skill: str
    demand_minute: int
    actual_demand_minute: int
    duration_minutes: int
    observed_release: int | None
    repeat: bool
    never_arrived: bool
    carry_in: bool
    source: str
    call_ids: tuple


def compute_duration_quantiles(events, quantile):
    """按技能从已结束事件估计到场后服务时长的分位数（分钟）。

    同时返回全样本兜底值；样本不足的技能不会得到虚构的分位数。
    """
    grouped: dict[str, list[int]] = {}
    overall = []
    for row in events:
        if not row.get('end') or not row.get('arrived'):
            continue
        duration = int((datetime.fromisoformat(row['end']) - datetime.fromisoformat(row['arrived'])).total_seconds() // 60)
        grouped.setdefault(row['skill'], []).append(max(0, duration))
        overall.append(max(0, duration))
    import numpy
    result = {skill: float(numpy.quantile(values, quantile)) for skill, values in grouped.items()}
    result['*'] = float(numpy.quantile(overall, quantile)) if overall else None
    return result


def build_timeline(events, calls, night, *, open_end_policy='observed', call_policy='episode',
                   merge_gap=5, impute_quantile=0.9, duration_quantiles=None):
    """构建单个夜班的时间线。

    open_end_policy:
      - observed：缺失结束或越过窗口的占用一律截到窗口末端，保留 open_end/censored 标记；
      - imputed：缺失结束用同技能服务时长分位数补全，仍越过窗口则截断并保留标记。
    call_policy:
      - episode：一场事件只产生一个需求时点（首呼/需求发生时间）；
      - calls：每次呼叫都是需求尝试；同事件距上一呼叫不超过 merge_gap 分钟的呼叫并入同一需求。
    """
    if open_end_policy not in ('observed', 'imputed'):
        raise ValueError('open_end_policy 只能是 observed 或 imputed')
    if call_policy not in ('episode', 'calls'):
        raise ValueError('call_policy 只能是 episode 或 calls')
    origin, window_end = night_bounds(night)
    window = WINDOW_MINUTES
    quantiles = duration_quantiles
    if quantiles is None:
        quantiles = compute_duration_quantiles(events, impute_quantile)

    intervals = []
    night_events = []
    for row in events:
        if night_of(row['start']) != night:
            continue
        night_events.append(row)
        actual_start = _minute(row['start'], origin)
        actual_arrived = _minute(row['arrived'], origin) if row.get('arrived') else None
        open_end = row.get('end') is None
        actual_release = _minute(row['end'], origin) if row.get('end') else None
        imputed = False
        fallback = quantiles.get(row['skill'], quantiles.get('*'))
        if actual_arrived is None and open_end and open_end_policy == 'imputed':
            # 连到场也缺失时，占用从需求时点起按分位数补全，并保留 never_arrived 标记
            if fallback is None:
                raise ValueError(f"技能 {row['skill']} 没有可用于补全的历史样本")
            imputed = True
            arrived_for_grid = max(0, actual_start)
            release = min(window, arrived_for_grid + max(1, int(round(fallback))))
        elif actual_arrived is None:
            arrived_for_grid = None
            release = None
        elif open_end and open_end_policy == 'imputed':
            if fallback is None:
                raise ValueError(f"技能 {row['skill']} 没有可用于补全的历史样本")
            imputed = True
            arrived_for_grid = max(0, actual_arrived)
            release = min(window, arrived_for_grid + max(1, int(round(fallback))))
        elif open_end:
            arrived_for_grid = max(0, actual_arrived)
            release = window
        else:
            arrived_for_grid = max(0, actual_arrived)
            release = min(window, actual_release)
        intervals.append(Interval(
            event_id=row['id'], episode=row['episode'], resident=row['resident'], skill=row['skill'],
            start_minute=max(0, actual_start),
            arrived_minute=arrived_for_grid,
            release_minute=release,
            actual_start_minute=actual_start,
            actual_arrived_minute=actual_arrived,
            actual_release_minute=actual_release,
            open_end=open_end,
            censored=(actual_release is not None and actual_release > window) or (open_end and release == window),
            carry_in=actual_start < 0,
            imputed=imputed,
        ))

    calls_by_episode: dict[str, list[dict]] = {}
    for call in calls:
        if night_of(call['time']) != night:
            continue
        calls_by_episode.setdefault(call['episode'], []).append(call)
    for buckets in calls_by_episode.values():
        buckets.sort(key=lambda item: item['time'])

    epochs = []
    interval_by_episode = {item.episode: item for item in intervals}
    for row in sorted(night_events, key=lambda item: (item['start'], item['id'])):
        interval = interval_by_episode[row['episode']]
        episode_calls = calls_by_episode.get(row['episode'], [])
        never_arrived = row.get('arrived') is None
        if open_end_policy == 'imputed' and interval.imputed:
            duration = max(1, int(round(quantiles.get(row['skill'], quantiles.get('*')))))
        elif row.get('arrived') and row.get('end'):
            duration = max(0, int((datetime.fromisoformat(row['end']) - datetime.fromisoformat(row['arrived'])).total_seconds() // 60))
        else:
            duration = max(1, int(round(quantiles.get(row['skill'], quantiles.get('*')) or 1)))
        if call_policy == 'episode' or not episode_calls:
            demand_actual = interval.actual_start_minute
            epochs.append(Epoch(
                key=f"{row['id']}:0", event_id=row['id'], episode=row['episode'], resident=row['resident'],
                skill=row['skill'], demand_minute=max(0, demand_actual), actual_demand_minute=demand_actual,
                duration_minutes=duration, observed_release=interval.release_minute, repeat=False,
                never_arrived=never_arrived, carry_in=interval.carry_in,
                source='episode' if not episode_calls else 'call',
                call_ids=tuple(call['id'] for call in episode_calls),
            ))
            continue
        demand_actual = interval.actual_start_minute
        groups: list[list[dict]] = []
        for call in episode_calls:
            offset = _minute(call['time'], origin)
            if not groups or offset - _minute(groups[-1][-1]['time'], origin) > merge_gap:
                groups.append([call])
            else:
                groups[-1].append(call)
        for index, group in enumerate(groups):
            offset = _minute(group[0]['time'], origin)
            epochs.append(Epoch(
                key=f"{row['id']}:{index}", event_id=row['id'], episode=row['episode'], resident=row['resident'],
                skill=row['skill'], demand_minute=max(0, offset), actual_demand_minute=offset,
                duration_minutes=duration, observed_release=interval.release_minute, repeat=index > 0,
                never_arrived=never_arrived, carry_in=interval.carry_in, source='call',
                call_ids=tuple(call['id'] for call in group),
            ))

    epochs.sort(key=lambda item: (item.demand_minute, item.episode, item.key))
    occupancy, skills = _occupancy_grid(intervals, window)
    return {
        'night': night,
        'window_minutes': window,
        'options': {
            'open_end_policy': open_end_policy,
            'call_policy': call_policy,
            'merge_gap': merge_gap,
            'impute_quantile': impute_quantile,
        },
        'intervals': [item.__dict__ for item in intervals],
        'epochs': [item.__dict__ for item in epochs],
        'occupancy': occupancy,
        'skills': skills,
        'open_end_event_ids': [item.event_id for item in intervals if item.open_end],
        'censored_event_ids': [item.event_id for item in intervals if item.censored],
        'carry_in_event_ids': [item.event_id for item in intervals if item.carry_in],
        'imputed_event_ids': [item.event_id for item in intervals if item.imputed],
    }


def _occupancy_grid(intervals, window):
    skills = sorted({item.skill for item in intervals})
    grid = {skill: [0] * window for skill in skills}
    for item in intervals:
        if item.arrived_minute is None or item.release_minute is None:
            continue
        for minute in range(item.arrived_minute, min(item.release_minute, window)):
            grid[item.skill][minute] += 1
    return grid, skills


def total_occupancy(timeline):
    if not timeline['skills']:
        return [0] * timeline['window_minutes']
    totals = [0] * timeline['window_minutes']
    for skill in timeline['skills']:
        column = timeline['occupancy'][skill]
        for minute, value in enumerate(column):
            totals[minute] += value
    return totals


def peak_composition(timeline):
    """找出最忙时段及其共同事件构成。

    峰值按同时占用的服务总数定义，返回首个达到峰值的连续时段、各技能人数与事件清单。
    各类别在同一分钟网格上共同计数，因此不会把各类别极端值分别相加。
    """
    totals = total_occupancy(timeline)
    window = timeline['window_minutes']
    peak = max(totals, default=0)
    if peak == 0:
        return {'peak': 0, 'start_minute': None, 'end_minute': None, 'by_skill': {}, 'contributors': []}
    first = totals.index(peak)
    last = first
    while last + 1 < window and totals[last + 1] == peak:
        last += 1
    by_skill = {skill: timeline['occupancy'][skill][first] for skill in timeline['skills']}
    contributors = []
    for item in timeline['intervals']:
        if item['arrived_minute'] is None or item['release_minute'] is None:
            continue
        if item['arrived_minute'] <= first < item['release_minute']:
            contributors.append({
                'event_id': item['event_id'], 'episode': item['episode'], 'resident': item['resident'],
                'skill': item['skill'], 'open_end': item['open_end'], 'imputed': item['imputed'],
                'overlap_minutes': max(0, min(item['release_minute'], last + 1) - max(item['arrived_minute'], first)),
            })
    contributors.sort(key=lambda item: (item['skill'], item['event_id']))
    return {
        'peak': peak,
        'start_minute': first,
        'end_minute': last + 1,
        'by_skill': by_skill,
        'contributors': contributors,
    }
