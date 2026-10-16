"""Strict subject splits, released manifest bindings and frozen VAE cache."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, config, sha, write_json
from .features import StaticFeatures


class Data:
    def __init__(self):
        self.cfg = config()
        self.root = Path(self.cfg['data_root'])
        self.cache = ROOT/self.cfg['run_root']/'cache'
        self.rows = [json.loads(line) for line in (self.root/'manifest.jsonl').read_text().splitlines()]
        self.rows.sort(key=lambda r:r['sample_id'])
        candidates = {r['sample_id']:r for r in map(json.loads,(self.root/'candidate_manifest_v2.jsonl').read_text().splitlines())}
        subjects = {r['identity_group']:r for r in map(json.loads,(self.root/'metadata/subjects.jsonl').read_text().splitlines())}
        splits = json.loads((self.root/'metadata/identity_splits.json').read_text())
        self.identities = sorted(subjects)
        self.identity_index = {s:i for i,s in enumerate(self.identities)}
        self.groups = defaultdict(list)
        self.captions = []
        self.text_index = {}
        self.features = StaticFeatures(self.cfg['body_model'])
        self.core, self.shoulder, self.meta = [], [], []
        for identity in self.identities:
            s = subjects[identity]
            assert s['split'] == splits[identity] and len(s['references']) == 1
            ref = next(iter(s['references'].values()))
            assert sha(self.root/ref['file']) == ref['sha256']
            m = json.loads((self.root/ref['file']).read_text())
            assert m['identity_group'] == identity and not m['reference_target_shared_recording']
            assert m['native_up_axis'] in ['y','z']
            p = m['fitted_parameters']
            assert sha(self.root/p['file']) == p['sha256']
            with np.load(self.root/p['file']) as geometry:
                core, shoulder = self.features(geometry['fullpose'][m['fitted_representative_index']],m['native_up_axis'])
            self.core.append(core)
            self.shoulder.append(shoulder)
            self.meta.append(dict(identity=identity,source=s['source_dataset'],split=s['split'],kind=m['kind']))
        self.core = np.stack(self.core)
        self.shoulder = np.stack(self.shoulder)
        # Features use physical scales, not statistics estimated on held-out subjects.
        for i,r in enumerate(self.rows):
            identity = r['identity_group']
            assert r['split'] == splits[identity]
            c = candidates[r['sample_id']]
            assert c['identity_group'] == identity and c['binding_sha256'] == r['candidate_binding_sha256']
            assert not c['reference_target_shared_recording']
            r['identity_index'] = self.identity_index[identity]
            r['task'] = c['original_task']
            r['length'] = min(c['motion_length']//4*4,196)
            assert r['length'] >= 40
            raw = (self.root/r['text']).read_text()
            assert sha(self.root/r['text']) == r['text_sha256']
            r['captions'] = []
            r['tokens'] = []
            for line in raw.splitlines():
                caption,tokens,start,end = line.split('#')
                assert float(start)==float(end)==0 and caption.strip()
                if caption not in self.text_index:
                    self.text_index[caption] = len(self.captions)
                    self.captions.append(caption)
                r['captions'].append(caption)
                r['tokens'].append(tokens.split())
            self.groups[identity].append(i)
        self.text_index[''] = len(self.captions)
        self.captions.append('')
        sets = {s:{r['identity_group'] for r in self.rows if r['split']==s} for s in ['train','validation','test']}
        assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
        self.split_ids = {s:sorted(v) for s,v in sets.items()}
        self.train_tasks = {}
        for identity in self.split_ids['train']:
            tasks = defaultdict(list)
            for i in self.groups[identity]:
                tasks[self.rows[i]['task']].append(i)
            self.train_tasks[identity] = list(tasks.values())
        self.bindings = dict(manifest=sha(self.root/'manifest.jsonl'),
                             candidates=sha(self.root/'candidate_manifest_v2.jsonl'),
                             identity_splits=sha(self.root/'metadata/identity_splits.json'),
                             subjects=sha(self.root/'metadata/subjects.jsonl'),
                             body_model=sha(self.cfg['body_model']),
                             mean=sha(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy'),
                             std=sha(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy'),
                             counts={s:dict(subjects=len(v),clips=sum(len(self.groups[x]) for x in v)) for s,v in sets.items()},
                             cross_source_identity_audited=False)

    def motion(self,i):
        r = self.rows[i]
        path = self.root/r['motion']
        assert sha(path) == r['motion_sha256']
        x = np.load(path)[:r['length']]
        assert x.shape==(r['length'],263) and np.isfinite(x).all()
        return x

    def open_cache(self):
        binding = json.loads((self.cache/'binding.json').read_text())
        assert binding == self.bindings
        self.mu = np.load(self.cache/'mu.npy',mmap_mode='r')
        self.std = np.load(self.cache/'std.npy',mmap_mode='r')
        self.words = np.load(self.cache/'words.npy',mmap_mode='r')
        self.word_masks = np.load(self.cache/'word_masks.npy',mmap_mode='r')
        self.word_pos = np.load(self.cache/'word_pos.npy',mmap_mode='r')

    def install_text_cache(self,model):
        original_encode=model.backbone.clip_model.encode_text
        def encode(text):
            if any(t not in self.text_index for t in text):
                return original_encode(text)
            indices = [self.text_index[t] for t in text]
            return (torch.from_numpy(np.array(self.words[indices])).cuda(),
                    torch.from_numpy(np.array(self.word_masks[indices])).cuda(),
                    torch.from_numpy(np.array(self.word_pos[indices])).cuda())
        model.backbone.clip_model.encode_text = encode

    def conditions(self,indices,variant='paired',donors=None):
        identities = [self.rows[i]['identity_index'] for i in indices]
        if donors is not None:
            identities = [donors.get(i,i) for i in identities]
        core = torch.tensor(self.core[identities],device='cuda')
        shoulder = torch.tensor(self.shoulder[identities],device='cuda')
        if variant=='fixed':
            core.zero_()
            shoulder.zero_()
        return core,shoulder

    def batch(self,indices,captions=None,sample_posterior=True):
        mu = torch.from_numpy(np.array(self.mu[indices])).cuda()
        std = torch.from_numpy(np.array(self.std[indices])).cuda()
        z = mu + std*torch.randn_like(mu) if sample_posterior else mu
        lengths = torch.tensor([self.rows[i]['length']//4 for i in indices],device='cuda')
        max_len = int(lengths.max())
        mask = torch.arange(max_len,device='cuda')[None]<lengths[:,None]
        z = z[:,:max_len] * mask[...,None,None]
        text = [self.rows[i]['captions'][0] for i in indices] if captions is None else captions
        return z,mask,text

    def sample_indices(self,rng,count):
        indices=[]
        for _ in range(count):
            identity = self.split_ids['train'][rng.randint(len(self.split_ids['train']))]
            tasks = self.train_tasks[identity]
            clips = tasks[rng.randint(len(tasks))]
            indices.append(clips[rng.randint(len(clips))])
        return indices

    def donor_map(self,seed,split='train'):
        rng = np.random.RandomState(seed)
        groups = defaultdict(list)
        for i,m in enumerate(self.meta):
            if m['split']==split:
                groups[(m['source'],m['kind'])].append(i)
        donors={}
        for key,items in groups.items():
            if len(items)<2:
                raise ValueError('No within-source derangement: '+str(key))
            perm = rng.permutation(items).tolist()
            donors.update({a:b for a,b in zip(perm,perm[1:]+perm[:1])})
        assert all(a!=b for a,b in donors.items())
        return donors


@torch.no_grad()
def prepare(data,model,vae):
    from .runtime import posterior
    cache=data.cache
    cache.mkdir(parents=True,exist_ok=True)
    if (cache/'binding.json').exists():
        data.open_cache()
        return
    n=len(data.rows)
    mu=np.lib.format.open_memmap(cache/'mu.npy',mode='w+',dtype='float32',shape=(n,49,7,32))
    std=np.lib.format.open_memmap(cache/'std.npy',mode='w+',dtype='float32',shape=mu.shape)
    mean=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    scale=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
    for start in range(0,n,32):
        indices=list(range(start,min(n,start+32)))
        motion=np.zeros((len(indices),196,263),dtype='float32')
        for j,i in enumerate(indices):
            x=data.motion(i)
            motion[j,:len(x)]=(x-mean)/scale
        m,logvar=posterior(vae,torch.from_numpy(motion).cuda())
        mu[indices]=m.cpu().numpy()
        std[indices]=(logvar*.5).exp().cpu().numpy()
        if start%1024==0:
            print('latent cache',start,n,flush=True)
    mu.flush();std.flush()
    words=np.lib.format.open_memmap(cache/'words.npy',mode='w+',dtype='float32',shape=(len(data.captions),77,512))
    masks=np.lib.format.open_memmap(cache/'word_masks.npy',mode='w+',dtype='bool',shape=(len(data.captions),77))
    positions=np.lib.format.open_memmap(cache/'word_pos.npy',mode='w+',dtype='int64',shape=(len(data.captions),))
    for start in range(0,len(data.captions),64):
        text=data.captions[start:start+64]
        w,m,p=model.backbone.clip_model.encode_text(text)
        words[start:start+len(text)]=w.cpu().numpy()
        masks[start:start+len(text)]=m.cpu().numpy()
        positions[start:start+len(text)]=p.cpu().numpy()
        if start%2048==0:
            print('text cache',start,len(data.captions),flush=True)
    words.flush();masks.flush();positions.flush()
    write_json(cache/'binding.json',data.bindings)
    data.open_cache()
