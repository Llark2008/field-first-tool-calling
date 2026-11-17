"""Authored paired training expressions and separately authored pilot expressions.

Render only supplied slots. Absence drops content; it never adds an explicit
missing-information warning. Template identity is kept out of model inputs.
"""
NUMBERS={1:'one',2:'two',3:'three',4:'four',5:'five'}


def render_train(name, a, family):
    f=family%6
    scopes=[
        ('within project {}','on {}','using {}'),
        ('from project {}','in environment {}','for algorithm {}'),
        ('for project {}','on environment {}','restricted to {}'),
        ('in the {} project','for {}','with algorithm {}'),
        ('belonging to project {}','running on {}','using algorithm {}'),
        ('under project {}','in {}','for {}'),
    ][f]
    scope=' '.join(t.format(a[k]) for k,t in zip(['project','environment','algorithm'],scopes) if k in a)
    metric=a.get('metric',['performance','run scores','evaluation results','recorded returns','run-level measurements','recorded scores'][f])
    if name=='rank_runs':
        if 'order' in a and 'limit' in a:
            obj=f"the IDs of the {NUMBERS[a['limit']]} {'run' if a['limit']==1 else 'runs'} with {'lowest' if a['order']=='asc' else 'highest'} {metric}, in that order"
        else:
            obj='a ranking of '+(f"{NUMBERS[a['limit']]} run {'ID' if a['limit']==1 else 'IDs'}" if 'limit' in a else 'run IDs')+' by '+metric
            if 'order' in a: obj+=', '+('smallest to largest' if a['order']=='asc' else 'largest to smallest')
        return ['Please retrieve {obj} {scope}.','I need {obj} {scope}.','For my review, return {obj} {scope}.',
                'Select {obj} {scope}.','Give me {obj} {scope}.','Fetch {obj} {scope}.'][f].format(obj=obj,scope=scope).replace(' .','.')
    if name=='summarize_metric':
        stat=[{'mean':'the average','min':'the smallest value','max':'the largest value'},
              {'mean':'the arithmetic mean','min':'the minimum','max':'the maximum'}][f%2].get(a.get('stat'),['a descriptive statistic','an aggregate','a summary value','a one-number summary','a statistical summary','a scalar summary'][f])
        obj=f'{stat} of {metric} across runs'
        return ['Calculate {obj} {scope}.','Report {obj} {scope}.','I need {obj} {scope}.',
                'Compute {obj} {scope}.','Return {obj} {scope}.','Obtain {obj} {scope}.'][f].format(obj=obj,scope=scope).replace(' .','.')
    if name=='compare_algorithms':
        alg=' versus '.join(a['algorithms'])
        return ['Compare {alg} by average {metric} {scope}.','Give separate mean {metric} values for {alg} {scope}.',
                'I need the mean {metric} comparison: {alg} {scope}.','Contrast {alg} using their mean {metric} {scope}.',
                'For {alg}, report each algorithm’s average {metric} {scope}.','Compare the average {metric} for {alg} {scope}.'][f].format(alg=alg,metric=metric,scope=scope).replace(' .','.')
    if name=='list_runs':
        return ['List run identifiers {scope}.','Fetch the IDs of matching runs {scope}.','Which run IDs are recorded {scope}?',
                'Return matching run identifiers {scope}.','Give me a list of run IDs {scope}.','Enumerate the recorded run IDs {scope}.'][f].format(scope=scope).replace(' .','.').replace(' ?','?')
    if name=='get_run_config':
        subject=' for run '+a['run_id'] if 'run_id' in a else ''
        return ['Read the stored training configuration{subject}.','Retrieve the recorded training settings{subject}.',
                'Show the training configuration{subject}.','Fetch the saved training setup{subject}.',
                'I need the recorded configuration{subject}.','Return the training settings from storage{subject}.'][f].format(subject=subject)
    if name=='get_metric_series':
        subject=' for run '+a['run_id'] if 'run_id' in a else ''
        n=a.get('max_points'); points=f'the first {NUMBERS[n]}' if n else 'the recorded'
        text=['Read {points} {metric} points{subject}.','Retrieve {points} values in the {metric} series{subject}.',
                'Show {points} {metric} observations{subject}.','Fetch {points} entries of {metric}{subject}.',
                'I need {points} {metric} values{subject}.','Return {points} samples in {metric}{subject}.'][f].format(points=points,metric=a['metric'],subject=subject)
        if n==1:
            for plural,singular in [('points','point'),('values','value'),('observations','observation'),('entries','entry'),('samples','sample')]:text=text.replace(plural,singular)
        return text
    raise ValueError(name)


def render_dev(name,a,family,boundary=False):
    """Independent sentence structures, including natural grammatical omission."""
    f=family%4
    parts=[]
    if 'project' in a: parts.append(['project '+a['project'],a['project']+' project','project '+a['project'],'the project named '+a['project']][f])
    if 'environment' in a: parts.append(['environment '+a['environment'],a['environment'],'environment '+a['environment'],a['environment']][f])
    if 'algorithm' in a: parts.append(a['algorithm'])
    scope=[' / '.join(parts),', '.join(parts),' and '.join(parts),', '.join(parts)][f]
    suffix=(', considering '+scope) if scope else ''
    metric=a.get('metric',['recorded measurements','run scores','performance values','evaluation results'][f])
    if name=='rank_runs':
        count=NUMBERS[a['limit']]+' ' if 'limit' in a else ''
        direction=a.get('order')
        if boundary and direction and 'limit' in a:
            ranking=f"{count}IDs with {'greatest' if direction=='desc' else 'least'} {metric}, ordered {'largest first' if direction=='desc' else 'smallest first'}"
        else:
            ranking=f'{count}IDs sorted on {metric}'
            if direction: ranking+=' in '+('descending' if direction=='desc' else 'ascending')+' order'
        ranking=ranking.replace('one IDs','one ID')
        return ['The shortlist should contain {ranking}{suffix}.','For a run ranking, produce {ranking}{suffix}.',
                'My requested output is {ranking}{suffix}.','Build a ranked ID list: {ranking}{suffix}.'][f].format(ranking=ranking,suffix=suffix)
    if name=='summarize_metric':
        stat={'mean':'average','min':'lowest value','max':'highest value'}.get(a.get('stat'),'summary value') if boundary else {'mean':'mean','min':'minimum','max':'maximum'}.get(a.get('stat'),'summary statistic')
        return ['Reduce the run values by computing the {stat} of {metric}{suffix}.','For the run collection, what is the {stat} of {metric}{suffix}?',
                'Use {metric} from matching runs to obtain its {stat}{suffix}.','The result I want is the {stat} of {metric} across recorded runs{suffix}.'][f].format(stat=stat,metric=metric,suffix=suffix)
    if name=='compare_algorithms':
        alg=' and '.join(a['algorithms'])
        return ['Put {alg} side by side, using each one’s average {metric}{suffix}.','For {alg}, calculate a mean {metric} for each{suffix}.',
                'A per-algorithm comparison is needed for {alg}: average {metric}{suffix}.','Report the average {metric} separately for {alg}{suffix}.'][f].format(alg=alg,metric=metric,suffix=suffix)
    if name=='list_runs':
        return ['Which stored runs match {scope}? Return their identifiers.','Give an ID inventory for the runs{suffix}.',
                'I want identifiers for matching runs{suffix}.','What are the recorded run identifiers{suffix}?'][f].format(scope=scope or 'the requested collection',suffix=suffix)
    if name=='get_run_config':
        run='run '+a['run_id'] if 'run_id' in a else 'a recorded run'
        return ['What training settings were saved for {run}?','Inspect {run} and return its stored configuration.',
                'The training setup attached to {run} is what I need.','Look up the persisted training parameters of {run}.'][f].format(run=run)
    if name=='get_metric_series':
        run=' belonging to '+a['run_id'] if 'run_id' in a else ''
        cap=('; keep only the first '+NUMBERS[a['max_points']]+(' point' if a['max_points']==1 else ' points')) if 'max_points' in a else ''
        return ['What observations are stored in the {metric} sequence{run}{cap}?','Open the recorded {metric} sequence{run}{cap}.',
                'The stored {metric} samples{run} are needed{cap}.','Look up the existing values along the {metric} series{run}{cap}.'][f].format(metric=a['metric'],run=run,cap=cap)
    raise ValueError(name)


DEV_UNSUPPORTED=[
    'Duplicate the stored configuration into a new run and launch its training.',
    'Permanently remove the recorded evaluation curves from storage.',
    'Send the experiment comparison to a remote messaging service.',
    'Overwrite the recorded return values with newly calculated values.',
]
