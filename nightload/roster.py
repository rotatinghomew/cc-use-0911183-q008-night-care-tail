def skills_on_duty(store, day):
    result = {}
    seen = set()
    for row in store.all('rosters'):
        if row['night'] == day:
            if row['person'] in seen:
                raise ValueError('同一夜人员重复登记')
            seen.add(row['person'])
            for skill in row['skills']:
                result[skill] = result.get(skill, 0) + 1
    return {'night': day, 'people': len(seen), 'qualified_by_skill': result, 'interpretation': '资质人数可交叉，不可跨技能直接相加'}


def save_roster(store, row):
    if len(set(row['skills'])) != len(row['skills']):
        raise ValueError('资质重复')
    for old in store.all('rosters'):
        if old['id'] != row['id'] and old['night'] == row['night'] and old['person'] == row['person']:
            raise ValueError('同一夜人员重复登记')
    return store.put('rosters', row)
