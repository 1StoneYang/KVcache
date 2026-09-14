import json,time
from pathlib import Path
from vlesshallu.config import load_config
from vlesshallu.resources import load_resource_lock
from vlesshallu.transformers_runtime import TransformersRuntime
cfg=load_config('configs/local_qwen25_4090.toml')
runtime=TransformersRuntime(cfg,load_resource_lock(cfg),'baseline')
sample={'sample_id':147,'image_id':147,'image_path':'/root/autodl-tmp/Method/dataset/AMBER/image/AMBER_147.jpg','prompt':cfg['decoding']['prompt']}
results=[]
for variant in ['baseline','mmshift','myopia_score','prunehal','kvsmooth','core4_no_rb']:
    runtime.variant=variant
    print('START',variant,flush=True)
    r=runtime.generate(sample)
    row={'variant':variant,'visual_tokens':r.visual_token_count,'final_visual_tokens':r.method_trace['final_visual_tokens'],'output_tokens':r.output_token_count,'seconds':r.elapsed_seconds,'peak_GiB':r.peak_gpu_memory_bytes/1024**3,'prunes':len(r.pruning_events),'caption':r.caption}
    results.append(row)
    Path('work/memory_fix_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print('PASS',json.dumps({k:v for k,v in row.items() if k!='caption'}),flush=True)
