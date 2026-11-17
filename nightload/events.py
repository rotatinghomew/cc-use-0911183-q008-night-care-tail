from datetime import datetime


def save_event(store, row):
    start = datetime.fromisoformat(row['start'])
    if row['end'] is not None and datetime.fromisoformat(row['end']) < start:
        raise ValueError('事件结束早于开始')
    if row['arrived'] is not None and datetime.fromisoformat(row['arrived']) < start:
        raise ValueError('到场早于需求')
    return store.put('events', row)


def interval_overlap(start, end, window_start, window_end):
    left = max(datetime.fromisoformat(start), datetime.fromisoformat(window_start))
    right = min(datetime.fromisoformat(end or window_end), datetime.fromisoformat(window_end))
    return max(0, int((right - left).total_seconds() // 60))


def window_events(store, start, end):
    if start >= end:
        raise ValueError('观察窗口不合法')
    return [{**row, 'observed_minutes': interval_overlap(row['start'], row['end'], start, end), 'open_end': row['end'] is None} for row in store.all('events') if row['start'] < end and (row['end'] is None or row['end'] > start)]
