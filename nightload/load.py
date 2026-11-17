from datetime import datetime
from .events import window_events


def descriptive_load(store, start, end):
    rows = window_events(store, start, end)
    total = sum(row['observed_minutes'] for row in rows)
    window = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
    waits = [(datetime.fromisoformat(row['arrived']) - datetime.fromisoformat(row['start'])).total_seconds() / 60 for row in rows if row['arrived'] is not None]
    return {'events': len(rows), 'observed_minutes': total, 'mean_occupancy': total / window, 'open_events': sum(row['open_end'] for row in rows), 'observed_wait_mean': sum(waits) / len(waits) if waits else None}
