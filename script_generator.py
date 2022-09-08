from itertools import permutations
from argparse import Namespace
from pathlib import Path
from mdh import GlobalHandler as GH
from numpy.random import default_rng

mother_seed = 2436
# mother_seed = 9453
dir_ = Path('script')
device = [0,1,2,4,5,6]

rng = default_rng(seed=mother_seed)
seed_list = rng.integers(1e4, size=12)

args = Namespace()
args.num_iters = 5000
args.method = 'MME_LCD_PPC'
args.alpha = 0.3
args.T = 0.6
args.lr = 0.01
args.update_interval = 500
args.note = 'based_on_mme'
gh = GH()

l = [[] for _ in range(len(device))]
for i, (s, t) in enumerate(permutations(range(4), 2)):
    idx = i % len(device)
    args.source, args.target, args.seed = s, t, seed_list[i]
    args.init = gh.regSearch(f':MME/.*seed:{args.seed}.*source.{s}.target.{t}')[0]
    cmd = 'python new_main.py ' + ' '.join([f'--{k} {v}' for k, v in args.__dict__.items()]) + f' --device {device[idx]}'
    l[idx].append(cmd)

for i in range(len(device)):
    with (dir_ / f'script{args.method.upper()}{i}.sh').open('w') as f:
        f.write('\n'.join(l[i]))