"""Bind every resource session to its raw samples and the persistent baseline."""
from toolcall_lab.ablation_data import read_rows
from toolcall_lab.formal_data import sha
from toolcall_lab.resource_monitor import resource_violation
from toolcall_lab.run_clarification_ablation import read_json as read,write_json

def verify_sessions(root,require_complete=False):
    logs=sorted(root.glob('resources-*.jsonl'))
    if not logs:
        if require_complete:raise ValueError('No resource evidence')
        return []
    initial=read(root/'resource-baseline.json');evidence=[];summaries={}
    for p in root.glob('resource-summary-*.json'):
        summary=read(p);name=summary.get('session_log')
        if not name or name in summaries:raise ValueError('Resource summary session missing/duplicate')
        if summary.get('error'):raise ValueError('Earlier resource session failed: '+str(summary['error']))
        summaries[name]=summary
    for p in logs:
        samples=read_rows(p)
        if not samples:raise ValueError('Empty resource session')
        for sample in samples:
            error=resource_violation(sample,initial['swap_used_bytes'])
            if error:raise ValueError('Recorded resource violation in '+p.name+': '+error)
        summary=summaries.get(p.name)
        if summary and summary.get('log_sha256')!=sha(p):raise ValueError('Resource log changed after summary')
        if require_complete and not summary:raise ValueError('Interrupted resource session needs audit before a final verdict: '+p.name)
        evidence.append(dict(session_log=p.name,samples=len(samples),status='complete' if summary else 'interrupted',sha256=sha(p)))
    if set(summaries)-{p.name for p in logs}:raise ValueError('Summary lacks raw log')
    return evidence

def save_summary(root,monitor):
    path=monitor.path
    summary=dict(monitor.summary(),session_log=path.name,log_sha256=sha(path))
    write_json(root/path.name.replace('resources-','resource-summary-').replace('.jsonl','.json'),summary)
