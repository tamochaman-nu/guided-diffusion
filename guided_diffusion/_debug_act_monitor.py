"""
TEMPORARY DEBUG CODE (ACT-MONITOR): remove after diagnosis.

Records activation statistics of UNet blocks via forward hooks to find which
block grows until it overflows fp16. The model itself is never modified: no
buffers, parameters or attributes are added, and every hook handle is kept so
that detach() removes them all.

Enable with environment variables:
    ACT_MONITOR=1          enable (default: disabled, nothing is attached)
    ACT_MONITOR_EVERY=100  record full statistics every N steps
    ACT_MONITOR_WATCH=1    on other steps, keep a cheap max|x| per module on the
                           GPU (no sync) so that the first non-finite activation
                           can be located; 0 makes hooks return immediately
Output: <logdir>/act_monitor/act.jsonl (+ first_nonfinite.json)
"""

import atexit
import json
import math
import os

import torch as th
import torch.distributed as dist

from . import logger
from .unet import AttentionBlock, ResBlock


def maybe_attach_act_monitor(model):
    """Return an attached ActMonitor if ACT_MONITOR=1 (rank 0 only), else None."""
    if os.environ.get("ACT_MONITOR", "0") != "1":
        return None
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return None
    return ActMonitor(
        model,
        out_dir=os.path.join(logger.get_dir(), "act_monitor"),
        every=int(os.environ.get("ACT_MONITOR_EVERY", "100")),
        watch=os.environ.get("ACT_MONITOR_WATCH", "1") == "1",
    )


def _norm(params):
    sq = sum(th.linalg.vector_norm(p.detach().float()) ** 2 for p in params)
    return float(math.sqrt(sq)) if params else 0.0


class ActMonitor:
    def __init__(self, model, out_dir, every=100, watch=True):
        self.model = model
        self.out_dir = out_dir
        self.every = every
        self.watch = watch
        self.step = None
        self.measuring = False
        self.microbatch = 0
        self.seen = {}  # target index -> microbatch of the last accepted call
        self.cur = []  # calls of the current step: (target index, stats tensors)
        self.watch_buf = []  # (step, [target index], stacked max|x|) kept on GPU
        self.first_nonfinite = None
        self.handles = []
        self.targets = []  # (name, kind, module whose params give w_norm)
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = open(os.path.join(out_dir, "act.jsonl"), "a")
        self._register()
        atexit.register(self.close)
        logger.log(f"ACT-MONITOR enabled: every={every} watch={watch} -> {out_dir}")

    # ---- hook registration -------------------------------------------------

    def _add(self, module, name, kind, w_params, extra=None):
        idx = len(self.targets)
        self.targets.append(
            dict(name=name, kind=kind, w_params=list(w_params), extra=extra or {})
        )
        self.handles.append(module.register_forward_hook(self._make_hook(idx)))

    def _register(self):
        m = self.model
        self.handles.append(m.register_forward_pre_hook(self._on_model_forward))
        if hasattr(m, "enc"):  # MPUNetModel (--arch mp)
            self._register_mp(m)
            return
        groups = [("input_blocks", b) for b in m.input_blocks]
        groups.append(("middle_block", m.middle_block))
        groups += [("output_blocks", b) for b in m.output_blocks]
        n_in = len(m.input_blocks)
        for gi, (group, block) in enumerate(groups):
            if group == "middle_block":
                bname = "middle_block"
            elif group == "input_blocks":
                bname = f"input_blocks.{gi}"
            else:
                bname = f"output_blocks.{gi - n_in - 1}"
            for li, layer in enumerate(block):
                lname = f"{bname}.{li}"
                if isinstance(layer, ResBlock):
                    # With use_scale_shift_norm, ResBlock calls out_layers[1:]
                    # (a new Sequential), so hook the last layer of the branch.
                    branch_params = [
                        p
                        for n, p in layer.named_parameters()
                        if not n.startswith("skip_connection")
                    ]
                    self._add(
                        layer.out_layers[-1],
                        f"{lname}.out_layers",
                        "res_branch",
                        branch_params,
                        extra=dict(emb=list(layer.emb_layers.parameters())),
                    )
                    self._add(
                        layer.skip_connection,
                        f"{lname}.skip_connection",
                        "skip",
                        layer.skip_connection.parameters(),
                    )
                    self._add(layer, lname, "resblock_out", layer.parameters())
                elif isinstance(layer, AttentionBlock):
                    self._add(
                        layer.proj_out,
                        f"{lname}.proj_out",
                        "attn_branch",
                        layer.parameters(),
                    )
                    self._add(layer, lname, "attn_out", layer.parameters())
            self._add(block, bname, "block", block.parameters())

    def _register_mp(self, m):
        from .unet_mp import MPBlock

        for side in ("enc", "dec"):
            for key, blk in getattr(m, side).items():
                bname = f"{side}.{key}"
                if isinstance(blk, MPBlock):
                    # w_norm_emb holds |emb_gain| (a scalar gain) for MP blocks.
                    self._add(blk.conv_res1, f"{bname}.conv_res1", "res_branch",
                              blk.parameters(), extra=dict(emb=[blk.emb_gain]))
                    if blk.attn_proj is not None:
                        self._add(blk.attn_proj, f"{bname}.attn_proj", "attn_branch",
                                  blk.parameters())
                self._add(blk, bname, "block", blk.parameters())
        self._add(m.out_conv, "out_conv", "block", m.out_conv.parameters())

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---- hooks -------------------------------------------------------------

    def _on_model_forward(self, module, args):
        self.microbatch += 1

    def _make_hook(self, idx):
        def hook(module, inputs, output):
            if not (self.measuring or self.watch) or self.step is None:
                return
            # use_checkpoint re-runs the forward during backward; keep only the
            # first call of each module per microbatch (forward precedes backward).
            if self.seen.get(idx) == self.microbatch:
                return
            self.seen[idx] = self.microbatch
            with th.no_grad():
                x = output.detach()
                max_abs = x.abs().amax().float()  # NaN propagates
                if self.measuring:
                    xf = x.float()
                    sq = xf.square().mean()  # fp32: fp16 square overflows
                    nonfinite = ~th.isfinite(xf).all()
                    self.cur.append(
                        (idx, max_abs, sq, nonfinite, str(x.dtype), x.shape[-1])
                    )
                else:
                    self.cur.append((idx, max_abs))

        return hook

    # ---- step boundaries (called from TrainLoop.run_step) ------------------

    def begin_step(self, step):
        self.step = step
        self.measuring = step % self.every == 0
        self.cur = []
        self.seen = {}

    def end_step(self):
        if self.step is None or not self.cur:
            return
        if self.measuring:
            self._flush_watch()  # earlier steps first, to keep "first" correct
            self._write_measurement()
        elif self.watch and self.first_nonfinite is None:
            idxs = [c[0] for c in self.cur]
            self.watch_buf.append((self.step, idxs, th.stack([c[1] for c in self.cur])))
        self.cur = []

    def _level(self, width):
        return int(round(math.log2(self.model.image_size / width)))

    def _write_measurement(self):
        n = len(self.cur)
        stats = th.stack(
            [c[1] for c in self.cur]
            + [c[2] for c in self.cur]
            + [c[3].float() for c in self.cur]
        ).cpu()  # the only GPU->CPU sync of this step
        agg = {}  # aggregate microbatches per target
        for k, (idx, _, _, _, dtype, width) in enumerate(self.cur):
            mx, sq, bad = stats[k].item(), stats[n + k].item(), stats[2 * n + k].item()
            a = agg.setdefault(
                idx, dict(max=mx, sq=0.0, cnt=0, bad=False, dtype=dtype, width=width)
            )
            if math.isnan(mx) or math.isnan(a["max"]):
                a["max"] = float("nan")
            else:
                a["max"] = max(a["max"], mx)
            a["sq"] += sq
            a["cnt"] += 1
            a["bad"] |= bool(bad)
        order = []
        for c in self.cur:
            if c[0] not in order:
                order.append(c[0])
        records = {}
        for idx in order:
            t, a = self.targets[idx], agg[idx]
            rec = dict(
                step=self.step,
                module=t["name"],
                kind=t["kind"],
                level=self._level(a["width"]),
                res=a["width"],
                dtype=a["dtype"],
                max_abs=a["max"],
                rms=math.sqrt(a["sq"] / a["cnt"]) if math.isfinite(a["sq"]) else float("nan"),
                w_norm=_norm(t["w_params"]),
                nonfinite=a["bad"],
            )
            if "emb" in t["extra"]:
                rec["w_norm_emb"] = _norm(t["extra"]["emb"])
            self.jsonl.write(json.dumps(rec) + "\n")
            records[idx] = rec
        self.jsonl.flush()
        if self.first_nonfinite is None:
            for k, idx in enumerate(order):
                if agg[idx]["bad"]:
                    preceding = [
                        {key: records[i][key] for key in ("module", "kind", "max_abs", "rms")}
                        for i in order[max(0, k - 5) : k]
                    ]
                    self._record_first_nonfinite(self.step, idx, preceding)
                    break

    def _flush_watch(self):
        """Locate the first non-finite activation among buffered watch steps."""
        buf, self.watch_buf = self.watch_buf, []
        if self.first_nonfinite is not None:
            return
        for step, idxs, vals in buf:
            vals = vals.cpu().tolist()
            for k, v in enumerate(vals):
                if not math.isfinite(v):
                    preceding = [
                        dict(module=self.targets[i]["name"], kind=self.targets[i]["kind"], max_abs=pv)
                        for i, pv in zip(idxs[max(0, k - 5) : k], vals[max(0, k - 5) : k])
                    ]
                    self._record_first_nonfinite(step, idxs[k], preceding)
                    return

    def _record_first_nonfinite(self, step, idx, preceding):
        rec = dict(
            event="first_nonfinite",
            step=step,
            module=self.targets[idx]["name"],
            kind=self.targets[idx]["kind"],
            preceding=preceding,  # up to 5 modules executed just before it
        )
        self.first_nonfinite = rec
        self.jsonl.write(json.dumps(rec) + "\n")
        self.jsonl.flush()
        with open(os.path.join(self.out_dir, "first_nonfinite.json"), "w") as f:
            json.dump(rec, f, indent=2)
        logger.log(f"ACT-MONITOR: first non-finite activation at step {step}: {rec['module']}")

    def close(self):
        if self.jsonl.closed:
            return
        self._flush_watch()
        self.jsonl.close()
        self.detach()
