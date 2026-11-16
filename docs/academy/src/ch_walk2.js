'use strict';
/* ═════════ Chapter 1b — Learning to Walk ═════════ */

/* ── pure math for the mini-PPO playground (also unit-checked in node) ── */
const MP = {
  reward(a) { return Math.exp(-((a - 1.3) ** 2) / (2 * .3 * .3)) + .5 * Math.exp(-((a + 1) ** 2) / (2 * .35 * .35)); },
  logp(a, mu, ls) { const s = Math.exp(ls); return -.5 * ((a - mu) / s) ** 2 - ls - .9189385; },
  step(st, P, rand) {
    const sig = Math.exp(st.ls), N = P.n, a = [], rew = [];
    for (let i = 0; i < N; i++) { a.push(st.mu + sig * gauss(rand)); rew.push(MP.reward(a[i]) + .03 * gauss(rand)); }
    const m = rew.reduce((x, y) => x + y, 0) / N, sd = Math.sqrt(rew.reduce((x, y) => x + (y - m) ** 2, 0) / N) + 1e-8;
    const adv = rew.map(r => (r - m) / sd), lpOld = a.map(x => MP.logp(x, st.mu, st.ls));
    let clipped = 0, seen = 0;
    for (let e = 0; e < P.epochs; e++) {
      const s = Math.exp(st.ls); let gm = 0, gl = 0; clipped = 0; seen = 0;
      for (let i = 0; i < N; i++) {
        const ratio = Math.exp(MP.logp(a[i], st.mu, st.ls) - lpOld[i]), A = adv[i], z = (a[i] - st.mu) / s;
        const live = (A >= 0 && ratio < 1 + P.eps) || (A < 0 && ratio > 1 - P.eps); seen++;
        if (live) { gm += ratio * A * z / s; gl += ratio * A * (z * z - 1); } else clipped++;
      }
      gm /= N; gl = gl / N + P.ent;
      st.mu += P.lr * s * s * gm; st.ls = clamp(st.ls + P.lr * gl, Math.log(.03), Math.log(1.6));
    }
    return { a, rew, adv, clipFrac: clipped / Math.max(1, seen), meanR: m };
  },
};

/* 6 ─ mini PPO playground */
lesson({
  id: 'w-sandbox', ch: 'walk', mins: 8,
  title: "PPO on a one-joint task",
  lede: "The whole PPO loop on the smallest problem that still shows what it does.",
  build() {
    const wid = Widget('PPO on one joint', body => {
      let startSig = 1.2, startMu = -1, ent = .02, eps = .2, it = 0, st, last = null, hist = [], rand = rng(5);
      const p1 = h('div'), p2 = h('div'), out = h('div.readout');
      const reset = () => { st = { mu: startMu, ls: Math.log(startSig) }; it = 0; last = null; hist = []; rand = rng(5 + Math.floor(Math.random() * 1000)); draw(); };
      const run = k => { for (let i = 0; i < k; i++) { last = MP.step(st, { n: 64, epochs: 4, lr: .03, eps, ent }, rand); hist.push(last.meanR); it++; } draw(); };
      const draw = () => {
        const sig = Math.exp(st.ls), p = Plot(660, 250, { x: [-2.5, 2.5], y: [-.05, 1.25], xt: [-2, -1, 0, 1, 2], yt: [0, .5, 1], xl: 'action a', yl: 'reward', m: { l: 40 } });
        const rc = []; for (let a = -2.5; a <= 2.5; a += .02) rc.push([p.sx(a), p.sy(MP.reward(a))]);
        const pdf = a => Math.exp(-.5 * ((a - st.mu) / sig) ** 2) / (sig * 2.5066), pm = pdf(st.mu), pp = []; for (let a = -2.5; a <= 2.5; a += .02) pp.push([p.sx(a), p.sy(-.05 + pdf(a) / pm * .5)]);
        p.svg.append(S('path', { d: path(pp) + ` L${p.sx(2.5)} ${p.sy(-.05)} L${p.sx(-2.5)} ${p.sy(-.05)}Z`, fill: 'var(--c-soft)', opacity: .9 }), Path(path(pp), 'var(--c)', 2), Path(path(rc), 'var(--ink3)', 2.5));
        if (last) last.a.forEach((a, i) => p.svg.append(Circle(p.sx(clamp(a, -2.5, 2.5)), p.sy(clamp(last.rew[i], -.05, 1.2)), 3 + Math.min(4, Math.abs(last.adv[i])), last.adv[i] > 0 ? 'var(--ok)' : 'var(--bad)', { opacity: .75 })));
        p.svg.append(Line(p.sx(st.mu), p.sy(-.05), p.sx(st.mu), p.sy(1.2), 'var(--c)', 1.5, { 'stroke-dasharray': '4 3' }), T(p.sx(1.3), p.sy(1.13), 'big hill', 'axis', 'middle'), T(p.sx(-1), p.sy(.58), 'small hill', 'axis', 'middle'), T(p.m.l + 6, p.m.t + 12, 'dots: last batch, green above average, red below', 'lbl', 'start', { style: 'fill:var(--ink3)' }));
        p1.innerHTML = ''; p1.append(p.svg);
        const q = Plot(660, 110, { x: [0, Math.max(30, it)], y: [0, 1], xt: [], yt: [0, .5, 1], m: { l: 40, t: 8, b: 22 }, xl: 'iteration' });
        if (hist.length > 1) q.svg.append(Path(path(hist.map((v, i) => [q.sx(i), q.sy(clamp(v, 0, 1))])), 'var(--c)', 2.5));
        p2.innerHTML = ''; p2.append(q.svg);
        out.innerHTML = ''; out.append(Chip('iteration', it), Chip('mean μ', fx(st.mu), 'c'), Chip('std σ', fx(sig), 'c'), Chip('batch reward', last ? fx(last.meanR) : '–'), Chip('clipped samples', last ? Math.round(last.clipFrac * 100) + '%' : '–', 'hl'));
      };
      body.append(p1, p2, out,
        h('div.row', null, Btn('Run 1 iteration', () => run(1)), Btn('Run 10', () => run(10)), Btn('Run 50', () => run(50)), Btn('Reset', reset, 'ghost')),
        h('div.grid2', null, h('div', null, Slider({ label: 'start σ', min: .2, max: 1.4, step: .05, value: startSig, on: v => { startSig = v; reset(); } }), Slider({ label: 'entropy weight', min: 0, max: .6, step: .02, value: ent, fmt: v => fx(v, 2), on: v => { ent = v; } })),
          h('div', null, Slider({ label: 'clip ε', min: .05, max: .4, step: .01, value: eps, on: v => { eps = v; } }), Seg([{ v: -1, label: 'Start by the small hill' }, { v: 0.2, label: 'Start in the middle' }], -1, v => { startMu = v; reset(); }))),
        h('p.cap', null, 'Each iteration samples 64 actions from N(μ, σ), normalizes their rewards into advantages, and runs 4 epochs of clipped-objective updates plus an entropy bonus. It is the real update rule applied to two parameters instead of a network. Step sizes are scaled by σ² as a stand-in for Adam.'));
      reset();
    });

    return [
      P('The grey curve is a reward landscape over a single action. The filled shape is the policy, a Gaussian with mean μ and standard deviation σ, and the dots are the last batch of 64 samples, green if the advantage is positive and red if it’s negative. Each iteration samples actions, turns their rewards into advantages, and runs 4 epochs of clipped updates with an entropy bonus.'),
      wid,
      H3('Things to try'),
      OL('Hit <b>Run 50</b> twice with the defaults (start σ = 1.2). A few wide samples land on the big hill, get large advantages and drag μ over. After that σ shrinks.', 'Drop <b>start σ</b> to 0.35 and run it again. The policy climbs the small hill and stays there. It never samples the big hill, so it never finds out it exists.', 'Raise <b>entropy weight</b> to 0.6 in that case. σ collapses more slowly, but the bonus can’t produce samples from a hill that’s many σ away.', 'Go back to the wide start with a high entropy weight. σ stays big and the policy never settles down.', 'Set <b>clip ε</b> to 0.4 and watch the clipped percentage fall as the updates get bigger.'),
      Note('Advantage only scores the samples the policy actually drew. Where it explores (the starting σ and the entropy bonus) decides what it ever gets to learn about.'),
      Check({ q: 'The policy starts near a small reward hill and its σ shrinks quickly. Where does it end up?', opts: ['On the bigger hill, since that’s better', 'On the small hill, since it never samples the bigger one', 'Bouncing between the two'], ans: 1, why: 'Advantage is only computed from samples the policy actually drew.' }),
      Check({ q: 'What does a high clipped-sample percentage mean?', opts: ['Many samples have already pushed the ratio to its limit, so more epochs on this batch do little', 'The critic is broken', 'The reward is negative'], ans: 0, why: 'Clipped samples contribute zero gradient.' }),
    ];
  },
  cards: [
    { q: "Why can PPO get stuck on a small reward hill?", a: "Advantage only scores the policy’s own samples. If σ shrinks before it samples the better region, it never learns that region exists." },
    { q: "What does the entropy bonus do in practice?", a: "It keeps σ from collapsing too early. It can’t rescue a policy that started far from the good region, and too much of it stops the policy from converging." },
  ],
});

/* 7 ─ backprop & Adam */
lesson({
  id: 'w-learn', ch: 'walk', mins: 9,
  title: "Backprop and Adam",
  lede: "The loss is one number and the network has thousands of weights. Backprop works out how much each weight contributed, and Adam decides how far to move it.",
  build() {
    const elu = x => x > 0 ? x : Math.exp(x) - 1, delu = x => x > 0 ? 1 : Math.exp(x);

    const race = Widget('Plain gradient descent vs Adam on a narrow ravine', body => {
      let n = 25; const holder = h('div'), out = h('div.readout');
      const f = (x, y) => .02 * x * x + y * y, gr = (x, y) => [.04 * x, 2 * y];
      const run = () => {
        const sg = [[-4, 1.5]], ad = [[-4, 1.5]]; let x = -4, y = 1.5, m = [0, 0], v = [0, 0];
        for (let t = 1; t <= 60; t++) {
          let g = gr(x, y); x -= .9 * g[0]; y -= .9 * g[1]; sg.push([x, y]);
        }
        x = -4; y = 1.5;
        for (let t = 1; t <= 60; t++) {
          const g = gr(x, y);
          m = m.map((mm, i) => .9 * mm + .1 * g[i]); v = v.map((vv, i) => .999 * vv + .001 * g[i] * g[i]);
          const mh = m.map(mm => mm / (1 - .9 ** t)), vh = v.map(vv => vv / (1 - .999 ** t));
          x -= .3 * mh[0] / (Math.sqrt(vh[0]) + 1e-8); y -= .3 * mh[1] / (Math.sqrt(vh[1]) + 1e-8); ad.push([x, y]);
        }
        return { sg, ad };
      };
      const { sg, ad } = run();
      const draw = () => {
        const p = Plot(660, 230, { x: [-8, 3], y: [-2, 2], xt: [-8, -6, -4, -2, 0, 2], yt: [-2, 0, 2], xl: 'weight 1 (flat direction)', yl: 'weight 2 (steep)', m: { l: 36 } });
        [.16, .5, 1.2, 2.6].forEach(c => p.svg.append(S('ellipse', { cx: p.sx(0), cy: p.sy(0), rx: Math.abs(p.sx(Math.sqrt(c / .02)) - p.sx(0)), ry: Math.abs(p.sy(Math.sqrt(c)) - p.sy(0)), fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.5 })));
        const tr = (arr, col, dash) => { p.svg.append(Path(path(arr.slice(0, n + 1).map(q => [p.sx(clamp(q[0], -8, 3)), p.sy(clamp(q[1], -2, 2))])), col, 2.5, 'none', dash ? { 'stroke-dasharray': dash } : {})); const e = arr[Math.min(n, arr.length - 1)]; p.svg.append(Circle(p.sx(clamp(e[0], -8, 3)), p.sy(clamp(e[1], -2, 2)), 5.5, col, { stroke: 'var(--surface)', 'stroke-width': 2 })); };
        tr(sg, 'var(--ink3)', '5 3'); tr(ad, 'var(--c)');
        p.svg.append(T(p.m.l + 8, p.m.t + 14, '- - plain gradient descent', 'lbl', 'start', { style: 'fill:var(--ink3)' }), T(p.m.l + 8, p.m.t + 30, '━ Adam', 'lbl', 'start', { style: 'fill:var(--c)' }), Circle(p.sx(0), p.sy(0), 4, 'var(--ink)'));
        holder.innerHTML = ''; holder.append(p.svg);
        out.innerHTML = ''; out.append(Chip('step', n), Chip('descent loss', fx(f(...sg[n]), 3)), Chip('Adam loss', fx(f(...ad[n]), 3), 'c'));
      };
      body.append(holder, Slider({ label: 'steps taken', min: 0, max: 60, step: 1, value: n, fmt: v => v, on: v => { n = v; draw(); } }), out,
        h('p.cap', null, 'The loss is steep in one direction and nearly flat in the other. Plain descent has to use a small step to stay stable in the steep direction, so it crawls along the flat one. Adam gives each weight its own step size.')); draw();
    });

    return [
      H2('Backprop'),
      P('Backprop asks how much the loss would change if you nudged one weight a tiny bit. That’s the gradient ∂L/∂w. If it’s positive, raising the weight raises the loss, so the weight should go down. Gradient descent moves every weight the opposite way from its gradient.'),
      P('The chain rule gives you the gradients, working backward one layer at a time. Take a layer computing out = w·h + b, where h comes from the previous layer, and say the gradient arriving from ahead is g = ∂L/∂out. Then:'),
      Eq('∂L/∂w = <span class="k">g · h</span>       incoming gradient times the input that weight multiplied\n∂L/∂b = <span class="k">g</span>           the bias only adds, so its gradient is just g\n∂L/∂h = <span class="k">g · w</span>       passed back to the previous layer'),
      P('The previous layer treats ∂L/∂h as its incoming gradient and does the same thing. Each layer gets a gradient, computes its weight gradients from its own inputs, and passes a gradient back through its weights.'),
      H2('Adam'),
      P('Plain gradient descent is w ← w − α·∂L/∂w. Adam tracks two running averages for each weight. One is m, the average gradient, which says which way the weight has been going. The other is v, the average squared gradient, which says how noisy that gradient has been. The update is roughly:'),
      Eq('w ← w − α · <span class="k">m / (√v + ε)</span>'),
      race,
      P('A weight whose gradient keeps pointing the same way takes big steps, and one with a jumpy gradient takes small ones, so every weight ends up with its own effective learning rate. You can treat the internals as a black box if you remember the short version: momentum for direction, variance for caution, per weight.'),
      Eq('the learning chain:  loss → backprop gives every weight’s gradient → Adam turns gradients into updates → repeat for every minibatch and epoch'),
      Check({ q: 'For a layer out = w·h + b with incoming gradient g, what is ∂L/∂w?', opts: ['g · h', 'g · w', 'g'], ans: 0, why: 'A weight’s share of the blame scales with the input it multiplied. ∂L/∂h = g·w is what gets passed backward.' }),
      Check({ q: 'In Adam, what does a weight with a consistent gradient direction do?', opts: ['Takes large steps', 'Takes tiny steps', 'Doesn’t move'], ans: 0, why: 'm is large compared to √v, so the step is large. Noisy gradients give a large v and a small m.' }),
    ];
  },
  cards: [
    { q: "What are the backprop formulas for out = w·h + b with incoming gradient g?", a: "<b>∂L/∂w = g·h</b>, <b>∂L/∂b = g</b>, <b>∂L/∂h = g·w</b>. The last one is passed to the previous layer." },
    { q: "What do Adam’s m and v track?", a: "<b>m</b> is a running average of the gradient (direction). <b>v</b> is a running average of the squared gradient (how noisy it is). The update is about w - α·m/(√v + ε)." },
    { q: "What’s Adam in one line?", a: "Momentum for direction, variance for caution, per weight." },
    { q: "What’s the full learning chain?", a: "Loss, then backprop gradients (chain rule), then Adam updates, repeated for every minibatch and epoch." },
  ],
});

/* 8 ─ position targets & PD */
function simPD(Kp, Kd, grav) {
  const I = .25, mgl = grav ? 6 : 0, c = .04, dt = .002, tmax = 40; let q = 0, w = 0; const qs = [], ts = [];
  for (let i = 0; i <= 1000; i++) {
    const t = i * dt, tgt = t >= .15 ? .6 : 0, tau = clamp(Kp * (tgt - q) + Kd * (0 - w), -tmax, tmax);
    qs.push(q); ts.push(tau); w += (tau - c * w - mgl * Math.sin(q)) / I * dt; q += w * dt;
  } return { qs, ts };
}
lesson({
  id: 'w-pd', ch: 'walk', mins: 7,
  title: "Position targets and PD control",
  lede: "A learned policy outputs joint position targets, not torques. A PD controller underneath turns those targets into torque, and it’s the first piece of the control layer.",
  build() {
    const pd = Widget('A PD controller holding a joint against gravity', body => {
      let Kp = 60, Kd = 6, g = true; const h1 = h('div'), out = h('div.readout');
      const draw = () => {
        const { qs, ts } = simPD(Kp, Kd, g), n = qs.length;
        const p = Plot(660, 200, { x: [0, 2], y: [-.1, .9], xt: [0, .5, 1, 1.5, 2], yt: [0, .3, .6, .9], xl: 'time (s)', yl: 'joint angle (rad)', m: { l: 44, b: 32 } });
        p.svg.append(Path(path([[p.sx(0), p.sy(0)], [p.sx(.15), p.sy(0)], [p.sx(.15), p.sy(.6)], [p.sx(2), p.sy(.6)]]), 'var(--hl)', 2, 'none', { 'stroke-dasharray': '5 4' }), Path(path(qs.filter((_, i) => i % 4 === 0).map((v, i) => [p.sx(i * 4 * .002), p.sy(clamp(v, -.1, .9))])), 'var(--c)', 3), T(p.m.l + 8, p.m.t + 14, '- - target (from the policy)', 'lbl', 'start', { style: 'fill:var(--hl)' }), T(p.m.l + 8, p.m.t + 30, '━ actual joint angle', 'lbl', 'start', { style: 'fill:var(--c)' }));
        const q2 = Plot(660, 110, { x: [0, 2], y: [-40, 40], xt: [], yt: [-40, 0, 40], m: { l: 44, t: 14, b: 8 }, yl: 'torque N·m' });
        q2.svg.append(Path(path(ts.filter((_, i) => i % 4 === 0).map((v, i) => [q2.sx(i * 4 * .002), q2.sy(v)])), 'var(--ink3)', 2));
        h1.innerHTML = ''; h1.append(p.svg, q2.svg);
        const mx = Math.max(...qs), fin = qs[n - 1];
        out.innerHTML = ''; out.append(Chip('overshoot', mx > .6 ? fx((mx - .6) / .6 * 100, 0) + '%' : '0%', mx > .66 ? 'bad' : 'ok'), Chip('final error', fx(.6 - fin, 3) + ' rad', Math.abs(.6 - fin) > .03 ? 'bad' : 'ok'), Chip('peak torque', fx(Math.max(...ts.map(Math.abs)), 0) + ' N·m'));
      };
      const sKp = Slider({ label: 'stiffness Kp', min: 0, max: 200, step: 1, value: Kp, fmt: v => v, on: v => { Kp = v; draw(); } }), sKd = Slider({ label: 'damping Kd', min: 0, max: 14, step: .1, value: Kd, fmt: v => fx(v, 1), on: v => { Kd = v; draw(); } });
      body.append(h1, sKp, sKd,
        h('div.row', null, Seg([{ v: true, label: 'Gravity on' }, { v: false, label: 'Gravity off' }], true, v => { g = v; draw(); }), Btn('Knee gains: Kp 200, Kd 5', () => { Kp = 200; Kd = 5; sKp.set(200); sKd.set(5); draw(); }, 'ghost sm'), Btn('Ankle gains: Kp 20, Kd 0.1', () => { Kp = 20; Kd = .1; sKp.set(20); sKd.set(.1); draw(); }, 'ghost sm')), out,
        h('p.cap', null, 'τ = Kp·(q_target − q) + Kd·(−q̇). With Kd = 0 the joint rings. With low Kp and gravity on, the joint settles short of the target, because gravity has to be balanced by a nonzero error. The two presets are typical leg gains from the papers, applied here to a made-up joint.')); draw();
    });


    return [
      Eq('policy → <span class="k">position targets</span> → control layer (PD for the legs, impedance for the arm) → torques → ros2_control → motors'),
      P('The locomotion policy’s output is roughly in [−1, 1]. You scale it (action_scale is around 0.25) and add it to a default standing pose:'),
      Eq('q_target = q_default + scale · action'),
      P('Everything after that happens below the policy, and the policy never sees it. The simplest controller is a PD loop on each joint, running at roughly 1 to 2 kHz.'),
      Eq('τ = <span class="k">Kp</span>·(q_target − q) + <span class="k">Kd</span>·(−q̇)'),
      P('Kp works like a spring pulling toward the target and Kd like a damper on the velocity. In simulation, MuJoCo’s position actuators play this role with the Kp and Kd from the training config. On hardware, the leg controller has to use those same gains.'),
      pd,
      P('The joint in the widget hangs against gravity, which is the situation an arm is in. With low Kp it settles short of the target. The leg policy gets away with a plain PD because it already learned to command positions that account for gravity. An arm has no learned policy underneath it doing that, so it needs the richer control law in the Controls chapter.'),
      Check({ q: 'What does the locomotion policy output for each joint?', opts: ['A torque', 'A small offset added to a default pose, so a position target', 'A motor current'], ans: 1, why: 'q_target = q_default + scale · action. Torque gets produced below the policy.' }),
      Check({ q: 'What does the Kd term do?', opts: ['Adds stiffness toward the target', 'Damps the velocity so the joint doesn’t ring or overshoot', 'Cancels gravity'], ans: 1, why: 'Kp is the spring and Kd is the damper.' }),
      Check({ q: 'Why does it matter that the policy’s output stays a position target?', opts: ['Torque control isn’t possible', 'The same policy can sit on top of MuJoCo’s actuators in sim or a real torque-control chain on hardware', 'It makes the network smaller'], ans: 1, why: 'The interface stays the same while whatever is underneath changes.' }),
    ];
  },
  cards: [
    { q: "What does the locomotion policy output?", a: "Joint position targets, <code>q_target = q_default + scale·action</code>. Not torques." },
    { q: "What’s the PD formula and what does each term do?", a: "<b>τ = Kp(q_target - q) + Kd(-q̇)</b>. Kp is the spring (stiffness) and Kd is the damper." },
    { q: "What’s the chain from policy to motor?", a: "Policy, position targets, control layer (PD for legs, impedance for the arm), torques, ros2_control, motors." },
    { q: "Why does a low-Kp joint settle short of its target under gravity?", a: "Gravity has to be balanced by Kp × error, so some error is left over." },
  ],
});

/* 9 ─ sim to real */
lesson({
  id: 'w-sim2real', ch: 'walk', mins: 9,
  title: "Sim to real",
  lede: "Two things get a policy trained in simulation to work on hardware: randomizing the physics during training, and keeping the interface the same while you swap out what’s underneath.",
  build() {
    const dr = Widget('Randomization and robustness (illustrative)', body => {
      let rangeW = .25, real = .35; const holder = h('div'), out = h('div.readout');
      const succ = (f, w) => Math.exp(-(Math.max(0, Math.abs(f - .6) - w) ** 2) / (2 * .07 * .07));
      const draw = () => {
        const p = Plot(660, 220, { x: [.1, 1.1], y: [0, 1.05], xt: [.2, .4, .6, .8, 1], yt: [0, .5, 1], yf: v => Math.round(v * 100) + '%', xl: 'real-world floor friction', yl: 'stays upright', m: { l: 46 } });
        p.svg.append(Rect(p.sx(.6 - rangeW), p.m.t, p.sx(.6 + rangeW) - p.sx(.6 - rangeW), 220 - p.m.t - p.m.b, 'var(--c-soft)', { opacity: .8 }), T(p.sx(.6), p.m.t + 12, 'randomized training range', 'axis', 'middle'));
        const cv = w => Path(path(Array.from({ length: 101 }, (_, i) => { const f = .1 + i * .01; return [p.sx(f), p.sy(succ(f, w))]; })), w === 0 ? 'var(--ink3)' : 'var(--c)', 3, 'none', w === 0 ? { 'stroke-dasharray': '6 4' } : {});
        p.svg.append(cv(0), cv(rangeW), Line(p.sx(real), p.m.t, p.sx(real), 220 - p.m.b, 'var(--hl)', 2), T(p.sx(real) + 6, 220 - p.m.b - 8, 'this robot', 'lbl', 'start', { style: 'fill:var(--hl)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        out.innerHTML = ''; out.append(Chip('trained on one fixed sim', Math.round(succ(real, 0) * 100) + '% upright', succ(real, 0) < .6 ? 'bad' : 'ok'), Chip('trained on randomized sims', Math.round(succ(real, rangeW) * 100) + '% upright', succ(real, rangeW) < .6 ? 'bad' : 'ok'));
      };
      body.append(holder, Slider({ label: 'randomization ±', min: 0, max: .5, step: .01, value: rangeW, fmt: v => '±' + fx(v), on: v => { rangeW = v; draw(); } }), Slider({ label: 'real friction', min: .15, max: 1.05, step: .01, value: real, on: v => { real = v; draw(); } }), out,
        h('p.cap', null, '<b>Illustrative curves, not measurements.</b> The dashed policy was trained on friction of exactly 0.6 and fails as soon as reality differs. The randomized policy has seen the whole shaded band, so the real robot is one more condition it has already handled.')); draw();
    });

    const swap = Widget('Simulation and hardware, same interface', body => {
      let mode = 'sim'; const box = (t, d, cls) => h('div', { style: { border: '1px solid ' + (cls === 'shared' ? 'var(--c)' : 'var(--line)'), background: cls === 'shared' ? 'var(--c-soft)' : 'var(--surface)', borderRadius: '10px', padding: '8px 12px' } }, h('b', { style: { fontSize: '14.5px' } }, t), d ? h('div', { style: { fontSize: '13px', color: 'var(--ink3)' } }, d) : null);
      const stack = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } });
      const draw = () => {
        stack.innerHTML = '';
        stack.append(box('Locomotion policy', 'the same weights either way', 'shared'), h('div.mono', { style: { textAlign: 'center', color: 'var(--c)', fontWeight: 600, borderBlock: '1.5px dashed var(--c)', padding: '5px' } }, '▼  THE INTERFACE: joint position targets  ▼'));
        if (mode === 'sim') stack.append(box('MuJoCo’s built-in actuator', 'A PD controller using the training Kp and Kd. You write a control value and the physics moves the joint. This one box stands in for the whole hardware chain.'), box('State: MuJoCo’s data', 'exact ground truth, read straight from the sim'));
        else stack.append(box('Leg controller (control layer)', 'plain PD with the same Kp and Kd as training, 1–2 kHz'), box('ros2_control  →  EtherCAT driver', 'read / update / write at 1 kHz, one frame per millisecond'), box('Drive: field-oriented control', '20–30 kHz, torque → phase currents'), box('Motors + encoders', 'real torque control on real hardware'), box('State: encoders + IMU → state estimator (IEKF)', 'noisy sensors fused into a clean estimate'));
      };
      body.append(Seg([{ v: 'sim', label: 'In simulation' }, { v: 'hw', label: 'On the real robot' }], 'sim', v => { mode = v; draw(); }), stack, h('p.cap', null, 'The low-level team’s job is to replace MuJoCo’s built-in actuation with the real chain, behind the same interface. The arm side has the same shape: the arm policy’s targets go to the arm controller.')); draw();
    });

    return [
      H2('Domain randomization'),
      P('The sim is never an exact copy of reality. Train on one fixed set of physics and the policy overfits to it, then falls over on the real robot because friction and mass are a little off. So between rollouts you randomize the physics: friction, link masses, motor strength, control latency and random shoves. The policy has to find something that works across the whole range, and the real robot ends up being one more variation it has already dealt with. Of everything used for sim-to-real, this matters most.'),
      dr,
      H2('Running on the real robot'),
      UL('The <b>command</b> comes in as [vx, vy, vyaw] in the body frame, from a joystick or from Nav2.', 'The <b>observation</b> is built from the state estimator (the IEKF), which fuses the noisy IMU and encoder readings, since there’s no ground truth on hardware. Its format and meaning have to match training exactly, or the policy gets garbage.', 'The <b>action</b> is the mean of the actor’s distribution, with no sampling.'),
      P('During training the critic can be given privileged information the real robot can’t sense, like true contact forces, friction and masses. The actor never sees any of it, so it can’t come to depend on it. A better-informed critic gives better advantages, which means better actor training, and then it gets discarded. To be clear, the actor’s observation isn’t masked. The critic just gets extra inputs.'),
      H2('Swapping what’s underneath'),
      P('In simulation MuJoCo is the entire low-level stack. There’s no EtherCAT, no drives, no FOC and no real-time kernel. You write a control value to a MuJoCo actuator, which is a PD controller using the Kp and Kd from the training config, and the physics moves the joint. On hardware, the leg controller from the control layer, ros2_control, EtherCAT, the drive and the motor take over that job. The controller has to use the same Kp and Kd the policy was trained with.'),
      swap,
      P('The interface stays the same. The policy outputs position targets and something makes the joints follow them, and only the implementation below changes. That’s why you can build the locomotion policy before any low-level hardware exists, and it’s the reason sim-to-real works at all. Manipulation policies work the same way. They also output position targets, and the arm controller sits underneath them.'),
      Check({ q: 'Why train across a range of frictions if the real robot has just one friction value?', opts: ['Friction can’t be measured', 'A policy trained on one exact set of physics overfits and fails when anything differs a little', 'Randomization makes training faster'], ans: 1, why: 'Across a range, the real robot is one more case the policy has already seen.' }),
      Check({ q: 'Who sees the privileged information (true contact forces, friction, masses) during training?', opts: ['The actor', 'Only the critic', 'Both'], ans: 1, why: 'The critic gets thrown away afterward, and the actor never depends on it.' }),
      Check({ q: 'What has to be the same between simulation and hardware?', opts: ['The physics engine', 'The interface (observation format in, position targets out) and the PD gains', 'The motor brand'], ans: 1, why: 'Everything else below the interface is free to change, but the gains are part of what the policy was trained on.' }),
    ];
  },
  cards: [
    { q: "What is domain randomization?", a: "Randomizing friction, link masses, motor strength, latency and shoves across envs, so the policy learns something that works across the whole range. The real robot is one more variation." },
    { q: "Who gets privileged information in training?", a: "The <b>critic</b> (true contact forces, friction, masses). The actor only sees the observation you can build on hardware. The critic is discarded." },
    { q: "At inference, where does the observation come from?", a: "The state estimator (IEKF) fusing IMU and encoders, formatted exactly like the training observation." },
    { q: "Why does sim-to-real work structurally?", a: "The interface is preserved: position targets out, same observation in. MuJoCo’s actuators can be replaced by the real chain and the policy doesn’t know. The PD gains have to match what it was trained with." },
    { q: "What replaces the whole hardware stack in simulation?", a: "MuJoCo’s built-in actuators, which act as PD controllers with the training Kp and Kd." },
  ],
});

/* 10 ─ what you actually write */
lesson({
  id: 'w-yours', ch: 'walk', mins: 7,
  title: "What you write",
  lede: "The framework already implements PPO. Your part is the environment and the reward, and the math from the earlier lessons is what you use to figure out why something isn’t working.",
  build() {
    return [
      P('Frameworks like rsl_rl and Stable-Baselines3 handle PPO and GAE for you. What you actually work on falls into four groups.'),
      UL('<b>The environment</b> covers the observation function, the reward, termination conditions, action scaling and the domain randomization ranges. This is about 90% of the work.', '<b>PPO hyperparameters</b> are things you tune but never implement.', '<b>Network architecture</b> is a few small choices, and the defaults are proven.', '<b>Training logistics</b> means iteration count, checkpoints, logging, and CPU or GPU.'),
      H3('Reward terms'),
      h('div.scroll-x', null, h('table.tt', null, h('thead', null, h('tr', null, ['term', 'sign', 'typical weight', 'what it’s for'].map(x => h('th', null, x)))), h('tbody', null,
        [['tracking_lin_vel', '+', '~1.0', 'match the commanded velocity'], ['tracking_ang_vel', '+', '~0.5', 'match the commanded turn rate'], ['orientation', '−', '~1–5', 'stay upright'], ['torque', '−', '~1e-5', 'efficiency'], ['action_rate', '−', '~0.01', 'smooth motion'], ['feet_air_time', '+', '~1.0', 'get a real gait instead of shuffling']].map(r => h('tr', null, r.map((c, i) => h('td', { style: i === 3 ? { textAlign: 'left' } : {} }, c))))))),
      Trap('Don’t add reward terms ahead of time. Start with velocity tracking, staying upright and a torque penalty, and only add a term when you can name the bad behavior it fixes.'),
      Match({ q: 'Match each bad behavior to the reward term that addresses it.', rows: [{ l: 'The robot shuffles its feet instead of stepping', r: 'feet_air_time' }, { l: 'Joints jitter every step', r: 'action_rate penalty' }, { l: 'The torso leans and wobbles', r: 'orientation penalty' }, { l: 'The robot ignores your speed command', r: 'tracking_lin_vel' }, { l: 'Motors work harder than they need to', r: 'torque penalty' }, { l: 'It doesn’t turn when told to', r: 'tracking_ang_vel' }], why: 'The loop is to watch a play rollout, spot a bad behavior, add one term and retrain.' }),
      H3('Hyperparameters'),
      h('div.scroll-x', null, h('table.tt', null, h('thead', null, h('tr', null, ['setting', 'default', 'change it?'].map(x => h('th', null, x)))), h('tbody', null,
        [['num_envs', '4096', 'yes, to fit your hardware'], ['learning_rate', '~1e-3 (adaptive)', 'if training is unstable'], ['steps per rollout', '24', 'sometimes'], ['gamma / lambda', '0.99 / 0.95', 'rarely'], ['clip ε', '0.2', 'rarely'], ['entropy / value coefficient', '0.01 / 0.5', 'rarely'], ['epochs / minibatch', '5 / a few hundred', 'rarely'], ['actor / critic layers', '[512,256,128] / [1024,512,256], ELU', 'no']].map(r => h('tr', null, r.map((c, i) => h('td', { style: i ? { textAlign: 'left' } : {} }, c))))))),
      P('The PD gains (Kp and Kd) for each joint also come from the training config, and the real controller has to use the same values. The Controls chapter covers that.'),
      H2('Workflow'),
      Order({ q: 'Put the workflow in order.', items: ['Clone a working repo (unitree_rl_gym or berkeley_humanoid). Don’t start from scratch.', 'Run their example training unchanged, to confirm your setup works', 'Swap in your robot’s URDF or MJCF, fix joint names and counts, and get it to render and stand', 'Write your observation function and action scaling', 'Write a minimal reward and train, watching reward and episode length in W&B', 'Watch a play rollout, find a bad behavior, add one term and retrain', 'Once it walks, add domain randomization ranges and retrain', 'Export the policy to ONNX for deployment'], why: 'The framework owns PPO. You own the environment and the reward.' }),
      Real('Running their example unchanged first shows your setup works before you touch anything, so if something breaks later you know it was your change.'),
      Check({ q: 'Where does about 90% of the effort go on a locomotion project?', opts: ['Implementing PPO and GAE', 'The environment: observations, reward, terminations and randomization', 'Designing a new network architecture'], ans: 1, why: 'Most of the time goes into shaping the reward and working out why the robot is doing something wrong.' }),
    ];
  },
  cards: [
    { q: "What are the four groups of things you work on in RL locomotion?", a: "1) <b>The environment</b> (observation, reward, terminations, action scaling, randomization), about 90% of the work. 2) PPO hyperparameters. 3) Network architecture. 4) Training logistics." },
    { q: "What’s the rule for adding reward terms?", a: "Start minimal (velocity tracking, upright, torque penalty) and add a term only when you can name the bad behavior it fixes." },
    { q: "Which hyperparameters do you actually change?", a: "num_envs (to fit your hardware), learning_rate (if it’s unstable) and sometimes the rollout length. Most of the rest stay at their defaults." },
    { q: "What’s the workflow in short?", a: "Clone a working repo, run its example unchanged, swap in your robot, write observations and action scaling, add a minimal reward, iterate one term at a time, add domain randomization, export ONNX." },
  ],
});
