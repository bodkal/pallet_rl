"""Training: MCTS data generation + parameter update (Sec. V-C, Eq. 2)."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .env import Cfg, combo_dims, compute_masks, features
from .model import Nets, Evaluator
from .mcts import Runner
from .augment import Augmenter
from .replay import PER


def setup_torch():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    except Exception:
        pass


def masked_logp(net, feat, mask_flat, device):
    t = torch.from_numpy(feat).to(device)
    lg = net.policy(t).reshape(t.shape[0], -1)
    m = torch.from_numpy(mask_flat).to(device)
    return t, torch.log_softmax(lg.masked_fill(~m, -1e30), dim=1)


def evaluate(cfg, ev, seqs, lens, n_sims, c_puct, scale, seed, chunk=100):
    """Test-set evaluation: MCTS (temp=0, no Dirichlet) and plain greedy policy."""
    r = Runner(cfg, ev, n_sims=n_sims, c_puct=c_puct, scale=scale, seed=seed)
    it, ut = [], []
    for s in range(0, len(lens), chunk):
        rec = r.play(seqs[s:s + chunk], lens[s:s + chunk], temp=0.0,
                     train=False, collect=False)
        it += [x['items'] for x in rec]
        ut += [x['util'] for x in rec]
    gr = r.greedy_return(seqs, lens)
    return dict(mcts_items=float(np.mean(it)), mcts_util=float(np.mean(ut)) * 100,
                greedy_util=float(np.mean(gr)) / 10 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cut1')
    ap.add_argument('--root', default='.')
    ap.add_argument('--name', default=None)
    ap.add_argument('--b', type=int, default=1)
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--games', type=int, default=256)
    ap.add_argument('--sims', type=int, default=100)
    ap.add_argument('--eval_sims', type=int, default=100)
    ap.add_argument('--c_puct', type=float, default=2.5)
    ap.add_argument('--dir_eps', type=float, default=0.25)
    ap.add_argument('--dir_alpha', type=float, default=0.3)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--temp_moves', type=int, default=0)
    ap.add_argument('--eval_c_puct', type=float, default=None)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_min', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--updates', type=int, default=60)
    ap.add_argument('--cap', type=int, default=200_000)
    ap.add_argument('--per_alpha', type=float, default=0.6)
    ap.add_argument('--augment', type=int, default=1)
    ap.add_argument('--filter', type=int, default=1)
    ap.add_argument('--value', type=int, default=1)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--eval_n', type=int, default=100)
    ap.add_argument('--eval_split', default='test',
                    help="which split the periodic eval reads; use 'val' for "
                         'hyperparameter search so test stays untouched')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--budget_min', type=float, default=1e9)
    args = ap.parse_args()

    setup_torch()
    name = args.name or f'{args.data}_b{args.b}_k{args.k}'
    run = os.path.join(args.root, 'runs', name)
    os.makedirs(run, exist_ok=True)
    with open(f'{run}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    cfg = Cfg(b=args.b, k=args.k)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Nets(cfg, with_value=bool(args.value)).to(dev)
    ev = Evaluator(net, cfg, dev)
    aug = Augmenter(cfg) if args.augment else None
    per = PER(args.cap, cfg, alpha=args.per_alpha, seed=args.seed + 7)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    tr = np.load(os.path.join(args.root, 'data', f'{args.data}_train.npz'))
    te = np.load(os.path.join(args.root, 'data',
                              f'{args.data}_{args.eval_split}.npz'))
    S, Ln = tr['seqs'], tr['lens']
    TS, TL = te['seqs'][:args.eval_n], te['lens'][:args.eval_n]

    start_it, best = 0, -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt'])
        start_it = ck['it'] + 1; best = ck.get('best', -1.0)
        print(f'resumed from {args.resume} at iter {start_it}', flush=True)

    rng = np.random.default_rng(args.seed)
    logf = open(f'{run}/log.jsonl', 'a')
    t_start = time.time()
    ep_total = start_it * args.games

    for it in range(start_it, args.iters):
        frac = it / max(1, args.iters - 1)
        lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * frac))
        for g in opt.param_groups:
            g['lr'] = lr

        # -------------------------------------------------- data generation
        net.eval()
        sel = rng.choice(len(Ln), size=args.games, replace=False)
        temp = args.temp
        runner = Runner(cfg, ev, n_sims=args.sims, c_puct=args.c_puct,
                        dir_alpha=args.dir_alpha, dir_eps=args.dir_eps,
                        scale=args.scale, seed=int(rng.integers(1 << 30)))
        t0 = time.time()
        rec = runner.play(S[sel], Ln[sel], temp=temp, train=True, collect=True,
                          temp_moves=args.temp_moves)
        t_sp = time.time() - t0
        ep_total += args.games

        utils = np.array([r['util'] for r in rec])
        bases = np.array([r['base'] for r in rec]) / 10
        kept = 0
        hs, bs, ps, zs = [], [], [], []
        for r in rec:
            if args.filter and r['G'] < r['base'] - 1e-9:
                continue
            kept += 1
            z = float(np.tanh((r['G'] - r['base']) / args.scale))
            for i in range(len(r['pi'])):
                hs.append(r['hm'][i]); bs.append(r['buf'][i])
                ps.append(r['pi'][i]); zs.append(z)
        if hs:
            per.add_many(np.stack(hs), np.stack(bs), np.stack(ps),
                         np.array(zs, np.float32))

        # -------------------------------------------------- parameter update
        net.train()
        beta = 0.4 + 0.6 * frac
        pl = vl = 0.0
        nupd = 0
        for _ in range(args.updates):
            if per.n < args.batch:
                break
            idx, w = per.sample(args.batch, beta)
            hm = per.hm[idx].copy(); bf = per.buf[idx].copy(); pi = per.pi[idx].copy()
            z = per.z[idx]
            if aug is not None:
                ts = rng.integers(0, 8, size=len(idx))
                for i, t in enumerate(ts):
                    if t:
                        hm[i], bf[i], pi[i] = aug.transform(hm[i], bf[i], pi[i], int(t))
            dims = combo_dims(bf, cfg.k)
            mask, _ = compute_masks(hm, dims, cfg.H)
            mf = mask.reshape(len(idx), -1)
            feat = features(hm, bf, cfg.H)
            x, logp = masked_logp(net, feat, mf, dev)
            pit = torch.from_numpy(pi).to(dev)
            wt = torch.from_numpy(w).to(dev)
            ce = -(pit * logp).sum(1)
            loss = (wt * ce).mean()
            pl += float(ce.mean())
            if net.value is not None:
                v = net.value(x)
                zt = torch.from_numpy(z).to(dev)
                se = (v - zt) ** 2
                loss = loss + (wt * se).mean()
                vl += float(se.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            per.update(idx, ce.detach().abs().cpu().numpy() + 0.01)
            nupd += 1

        row = dict(it=it, ep=ep_total, lr=round(lr, 6), t_sp=round(t_sp, 1),
                   sp_util=round(float(utils.mean()) * 100, 2),
                   base_util=round(float(bases.mean()) * 100, 2),
                   kept=kept, n_replay=per.n,
                   pol_loss=round(pl / max(nupd, 1), 4),
                   val_loss=round(vl / max(nupd, 1), 4),
                   mins=round((time.time() - t_start) / 60, 1))

        # -------------------------------------------------- evaluation
        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            net.eval()
            m = evaluate(cfg, ev, TS, TL, args.eval_sims,
                         args.eval_c_puct or args.c_puct,
                         args.scale, args.seed + 1000 + it)
            row.update({k: round(v, 3) for k, v in m.items()})
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                        'it': it, 'best': best, 'cfg': cfg.as_dict()},
                       f'{run}/last.pt')
            if m['mcts_util'] > best:
                best = m['mcts_util']
                torch.save({'net': net.state_dict(), 'it': it, 'best': best,
                            'cfg': cfg.as_dict(), 'metrics': m}, f'{run}/best.pt')
                with open(f'{args.root}/results/{name}_best.json', 'w') as f:
                    json.dump(dict(name=name, data=args.data, b=args.b, k=args.k,
                                   iter=it, episodes=ep_total, **m), f, indent=2)
            row['best'] = round(best, 3)

        print(json.dumps(row), flush=True)
        logf.write(json.dumps(row) + '\n'); logf.flush()
        if (time.time() - t_start) / 60 > args.budget_min:
            print('budget reached', flush=True)
            break
    logf.close()


if __name__ == '__main__':
    main()
