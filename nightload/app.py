from pathlib import Path
import json

from .events import *
from .load import *
from .roster import *


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

    def call(self, method, route, payload):
        if method == 'GET' and route == 'summary':
            return self.demo()
        if method == 'POST' and route == 'event':
            return {'revision': save_event(self.store, payload)}
        if method == 'POST' and route == 'roster':
            return {'revision': save_roster(self.store, payload)}
        if method == 'GET' and route == 'load':
            return descriptive_load(self.store, payload['start'], payload['end'])
        raise KeyError('接口不存在')
