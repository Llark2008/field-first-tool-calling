"""Public synthetic state and two independently implemented result calculators."""
import sqlite3
from statistics import mean


def canonical_action(action, bindings=None):
    name, args = action['name'], dict(action['arguments'])
    binding = (bindings or {}).get(name)
    if binding:
        name = binding['name']
        args = {binding['arguments'][key]: value for key, value in args.items()}
    return name, args


def execute_fixture(fixture, action, bindings=None):
    """Runtime API: deliberately accepts no case, target or expected result."""
    name, a = canonical_action(action, bindings)
    rows = [r for r in fixture['runs'] if all(r[k] == a[k] for k in ('project', 'environment', 'algorithm') if k in a)]
    if name == 'list_runs': return sorted(r['run_id'] for r in rows)
    if name == 'count_runs': return len(rows)
    if name == 'summarize_metric': return {'mean': mean, 'min': min, 'max': max}[a['stat']]([r[a['metric']] for r in rows])
    if name == 'rank_runs':
        return [r['run_id'] for r in sorted(rows, key=lambda r: r[a['metric']], reverse=a['order'] == 'desc')[:a['limit']]]
    if name == 'compare_algorithms':
        return {algo: mean(r[a['metric']] for r in rows if r['algorithm'] == algo) for algo in a['algorithms']}
    if name == 'get_run_config': return dict(fixture['configs'][a['run_id']])
    if name == 'get_metric_series': return fixture['series'][a['run_id']][:a.get('max_points', 3)]
    if name == 'describe_columns': return list(fixture['tables'][a['table']][0])
    if name == 'group_metric':
        return {key: mean(r[a['metric']] for r in rows if r[a['group_by']] == key) for key in sorted({r[a['group_by']] for r in rows})}
    if name == 'count_missing': return sum(r[a['column']] is None for r in fixture['tables'][a['table']])
    raise ValueError('Unknown fixture operation: ' + name)


def reference_result(fixture, action, bindings=None):
    """Offline oracle: SQL filtering, ordering and aggregation, not the executor."""
    name, a = canonical_action(action, bindings)
    if name == 'get_run_config':
        return {key: value for key, value in fixture['configs'][a['run_id']].items()}
    if name == 'get_metric_series':
        return [value for i, value in enumerate(fixture['series'][a['run_id']]) if i < a.get('max_points', 3)]
    if name == 'describe_columns': return [key for key in fixture['tables'][a['table']][0].keys()]
    if name == 'count_missing':
        values = [row[a['column']] for row in fixture['tables'][a['table']]]
        return len(values) - len([value for value in values if value is not None])
    with sqlite3.connect(':memory:') as db:
        columns = list(fixture['runs'][0])
        db.execute('CREATE TABLE runs (' + ','.join('"' + col + '"' for col in columns) + ')')
        db.executemany('INSERT INTO runs VALUES (' + ','.join('?' for _ in columns) + ')', [tuple(r[c] for c in columns) for r in fixture['runs']])
        keys = [k for k in ('project', 'environment', 'algorithm') if k in a]
        where = ' WHERE ' + ' AND '.join(k + ' = ?' for k in keys) if keys else ''
        values = [a[k] for k in keys]
        if name == 'list_runs': return [r[0] for r in db.execute('SELECT run_id FROM runs' + where + ' ORDER BY run_id', values)]
        if name == 'count_runs': return db.execute('SELECT count(*) FROM runs' + where, values).fetchone()[0]
        if name == 'rank_runs':
            assert a['metric'] in ('final_return', 'peak_return') and a['order'] in ('asc', 'desc')
            return [r[0] for r in db.execute('SELECT run_id FROM runs' + where + ' ORDER BY ' + a['metric'] + ' ' + a['order'] + ' LIMIT ?', values + [a['limit']])]
        if name == 'summarize_metric':
            assert a['metric'] in ('final_return', 'peak_return')
            fn = {'mean': 'avg', 'min': 'min', 'max': 'max'}[a['stat']]
            return db.execute('SELECT ' + fn + '(' + a['metric'] + ') FROM runs' + where, values).fetchone()[0]
        if name in ('compare_algorithms', 'group_metric'):
            key = 'algorithm' if name == 'compare_algorithms' else a['group_by']
            assert key in ('algorithm', 'environment') and a['metric'] in ('final_return', 'peak_return')
            result = dict(db.execute('SELECT ' + key + ', avg(' + a['metric'] + ') FROM runs' + where + ' GROUP BY ' + key, values))
            return {k: result[k] for k in a['algorithms']} if name == 'compare_algorithms' else result
    raise ValueError('No reference for ' + name)
