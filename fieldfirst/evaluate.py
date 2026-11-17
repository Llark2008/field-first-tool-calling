"""Fresh checkpoint load, strict free generation and separate teacher-forced loss."""
import json
from fieldfirst.verify import ROOT, read_rows, validate_alignment
from experiments.fields_sft.data import messages, encoded_target, digest
from experiments.fields_sft.train import verify_checkpoint
from experiments.fields_sft.evaluate import validate_teacher, teacher_summary
from experiments.fields_epochs.report import cell_scores
from experiments.field_action_diagnostics.run import immutable, write
from toolcall_lab.train_formal import configure_mlx_memory
from toolcall_lab.formal_data import sha


def evaluate_checkpoint(snapshot, out, config, micro, resources, limit=None):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    checkpoint = out/f'step-{micro:04d}'
    verify_checkpoint(checkpoint, config, micro)
    dest = out/f'eval-{micro:04d}'; dest.mkdir(exist_ok=True)
    immutable(dest/'identity.json', dict(config=config['fingerprint'], adapter_sha256=sha(checkpoint/'adapters.safetensors'), limit=limit))
    configure_mlx_memory(mx); resources.check()
    model, tok = load(snapshot, adapter_path=str(checkpoint)); model.eval(); resources.check()
    result = {}
    try:
        for split in ('seen', 'dev'):
            cases = read_rows(ROOT/f'data/{split}-cases.jsonl')
            if limit: cases = cases[:limit]
            path = dest/f'{split}-free.jsonl'
            raw = read_rows(path) if path.exists() else []
            validate_alignment(cases[:len(raw)], raw)
            for c in cases[len(raw):]:
                resources.check(); msgs = messages({k:c[k] for k in ('request','context','tools')}, 'F')
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                parts=[]; last=None
                for response in stream_generate(model, tok, prompt=prompt, max_tokens=384, sampler=make_sampler(temp=0)):
                    resources.check(); parts.append(response.text); last=response
                if last is None: raise ValueError('Empty generation')
                row=dict(case_id=c['id'], output=''.join(parts), finish_reason=last.finish_reason, prompt_sha256=digest(msgs))
                with path.open('a') as h: h.write(json.dumps(row)+'\n'); h.flush()
                raw.append(row); mx.clear_cache()
            validate_alignment(cases, raw)
            path = dest/f'{split}-teacher.jsonl'
            rows = read_rows(path) if path.exists() else []
            validate_teacher(cases, rows, 'F', tok, 2048, partial=True)
            for c in cases[len(rows):]:
                resources.check(); _, ids, offset, groups = encoded_target(c, 'F', tok, 2048)
                batch=mx.array([ids]); logits=model(batch[:,:-1])
                ce=nn.losses.cross_entropy(logits,batch[:,1:]).astype(mx.float32)
                correct=mx.argmax(logits,axis=-1)==batch[:,1:]; mx.eval(ce,correct)
                row=dict(case_id=c['id'], tokens=ids, offset=offset, groups=groups,
                         losses=ce[0,offset-1:].tolist(), correct=correct[0,offset-1:].tolist())
                with path.open('a') as h: h.write(json.dumps(row)+'\n'); h.flush()
                rows.append(row); del batch,logits,ce,correct; mx.clear_cache()
            validate_teacher(cases, rows, 'F', tok, 2048)
            result[split] = dict(**cell_scores(cases,raw), teacher=teacher_summary(rows))
            print(json.dumps(dict(microstep=micro,split=split,metrics=result[split])),flush=True)
        resources.check(); write(dest/'metrics.json', result)
    finally:
        del model,tok; mx.clear_cache()
