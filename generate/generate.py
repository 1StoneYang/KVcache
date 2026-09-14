#!/usr/bin/env python3
"""Standalone Qwen2.5-VL FullKV AMBER generation; no method patches."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/root/autodl-tmp/model/Qwen2.5-VL-7B')
    p.add_argument('--amber-root', default='/root/autodl-tmp/Method/dataset/AMBER')
    p.add_argument('--output', type=Path, default=ROOT/'outputs'/'amber_baseline')
    p.add_argument('--ids', help='Comma-separated AMBER sample IDs, e.g. 1,147')
    p.add_argument('--limit', type=int, default=0, help='0 runs all selected samples')
    p.add_argument('--max-pixels', type=int, default=1605632)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--prompt', default=None, help='Default: use each AMBER query unchanged')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.limit < 0 or args.max_pixels < 3136 or args.max_new_tokens < 1:
        p.error('limit >= 0, max-pixels >= 3136 and max-new-tokens >= 1 are required')
    return args

def main():
    args = parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    query_path = Path(args.amber_root)/'data/query/query_generative.json'
    query_bytes = query_path.read_bytes()
    all_samples = json.loads(query_bytes)
    if len({s['id'] for s in all_samples}) != len(all_samples):
        raise ValueError('Duplicate AMBER sample IDs')
    wanted = set(map(int, args.ids.split(','))) if args.ids else None
    if wanted and not wanted.issubset({s['id'] for s in all_samples}):
        raise ValueError('Unknown sample ID in --ids')
    samples = [s for s in all_samples if wanted is None or s['id'] in wanted]
    if args.limit:
        samples = samples[:args.limit]
    for s in samples:
        if not (Path(args.amber_root)/'image'/s['image']).is_file():
            raise FileNotFoundError(s['image'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # OS releases this lock even when the process is interrupted.
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {'model': str(Path(args.model).resolve()), 'dtype': 'bfloat16',
            'attention': 'flash_attention_2', 'kv_method': 'full_kv', 'quantization': None,
            'max_pixels': args.max_pixels, 'max_new_tokens': args.max_new_tokens,
            'prompt_override': args.prompt, 'seed': args.seed,
            'query_sha256': hashlib.sha256(query_bytes).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch': torch.__version__, 'transformers': transformers.__version__}
        mpath = output/'manifest.json'
        if mpath.exists():
            if json.loads(mpath.read_text()) != manifest:
                raise RuntimeError('Settings/code changed: use a different --output directory')
        else:
            if (output/'records.json').exists() or (output/'responses.json').exists():
                raise RuntimeError('Existing results have no manifest; choose a new --output')
            atomic_json(mpath, manifest)
        rpath = output/'records.json'
        records = json.loads(rpath.read_text()) if rpath.exists() else []
        done = {r['id'] for r in records}
        def export():
            atomic_json(output/'responses.json', [{'id': r['id'], 'response': r['response']}
                for r in sorted(records, key=lambda r: r['id'])])
        export()
        pending = [s for s in samples if s['id'] not in done]
        print(f'Selected {len(samples)}; remaining {len(pending)}; output {output}', flush=True)
        if not pending:
            print('Already complete; model not loaded.', flush=True)
            return
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
        processor.image_processor.max_pixels = args.max_pixels
        if isinstance(getattr(processor.image_processor, 'size', None), dict):
            processor.image_processor.size['longest_edge'] = args.max_pixels
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
            device_map='cuda:0').eval()
        generation = copy.deepcopy(model.generation_config)
        generation.do_sample = False
        generation.num_beams = 1
        generation.max_new_tokens = args.max_new_tokens
        generation.temperature = None
        generation.top_p = None
        generation.top_k = None
        generation.use_cache = True
        generation.output_attentions = False
        generation.output_scores = False
        generation.return_dict_in_generate = False
        for s in pending:
            sid = s['id']
            image = (Path(args.amber_root)/'image'/s['image']).resolve()
            prompt = args.prompt if args.prompt is not None else s['query']
            print(f'START id={sid}', flush=True)
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image.as_uri(), 'max_pixels': args.max_pixels},
                    {'type': 'text', 'text': prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                inputs = processor(text=[text], images=images, videos=videos,
                                   padding=False, return_tensors='pt').to('cuda:0')
                prompt_len = inputs.input_ids.shape[1]
                visual_tokens = int(inputs.input_ids.eq(model.config.image_token_id).sum())
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation)
                new_ids = generated[:, prompt_len:]
                response = processor.batch_decode(new_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0].strip()
                torch.cuda.synchronize()
                row = {'id': sid, 'response': response, 'prompt': prompt, 'image': str(image),
                    'input_tokens': prompt_len, 'visual_tokens': visual_tokens,
                    'output_tokens': new_ids.shape[1], 'seconds': time.perf_counter()-started,
                    'peak_gpu_GiB': torch.cuda.max_memory_allocated()/1024**3}
                records.append(row)
                atomic_json(rpath, records)
                export()
                print(f"DONE id={sid} visual={visual_tokens} generated={row['output_tokens']} "
                      f"seconds={row['seconds']:.2f} peak={row['peak_gpu_GiB']:.2f} GiB", flush=True)
                del inputs, generated, new_ids, images, videos
            except Exception as exc:
                atomic_json(output/'last_error.json', {'id': sid, 'image': str(image), 'error': str(exc)})
                raise
        print(f'Completed. AMBER responses: {output / "responses.json"}', flush=True)

if __name__ == '__main__':
    main()
