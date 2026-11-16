'use strict';
/* ═════════ Chapter 1a — Learning to Walk ═════════ */

/* 1 ─ worlds & vocabulary */
lesson({
  id: 'w-worlds', ch: 'walk', mins: 7,
  title: "Parallel worlds and terminology",
  lede: "Locomotion policies get trained on thousands of simulated robots at once. This page pins down the terms, because “rollout” and “episode” get used loosely.",
  build() {
    const gridW = Widget('One rollout, drawn', body => {
      let seed = 11;
      const holder = h('div'), note = h('p.cap');
      const draw = () => {
        const r = rng(seed), envs = 12, steps = 24, W = 660, lm = 58, rowH = 19, top = 26, cw = (W - lm - 10) / steps;
        const svg = svgRoot(W, top + envs * rowH + 34, { 'aria-label': 'Twelve environments running 24 steps each, with episodes ending at falls' });
        let falls = 0, eps = 0;
        for (let e = 0; e < envs; e++) {
          const pf = .015 + .14 * Math.pow(r(), 2); let ep = 0;
          svg.append(T(lm - 8, top + e * rowH + 13, 'env ' + (e + 1), 'axis', 'end'));
          for (let s = 0; s < steps; s++) {
            svg.append(Rect(lm + s * cw + 1, top + e * rowH + 2, cw - 2, rowH - 5, 'var(--c)', { opacity: [.9, .5, .28, .7][ep % 4], rx: 3 }));
            if (r() < pf && s < steps - 1) { falls++; svg.append(T(lm + s * cw + cw / 2, top + e * rowH + 13, '×', 'lbl-b', 'middle', { style: 'fill:var(--bad)' })); ep++; }
          }
          eps += ep + 1;
        }
        svg.append(Rect(lm - 2, top - 3, steps * cw + 4, envs * rowH + 2, 'none', { stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4', rx: 6 }));
        svg.append(T(lm, 15, 'ONE ROLLOUT = every env × 24 steps, pooled', 'lbl-b'));
        svg.append(T(lm, top + envs * rowH + 20, 'time (steps) →', 'axis'), T(W - 10, top + envs * rowH + 20, '× = fell, episode ends, env resets', 'axis', 'end'));
        holder.innerHTML = ''; holder.append(svg);
        note.innerHTML = 'In this draw, <b>' + eps + ' episodes</b> started and <b>' + falls + ' ended in a fall</b> within the same rollout. Shades mark different episodes. Real training has 4096 rows.';
      };
      body.append(holder, note, h('div.row', null, Btn('Draw another rollout', () => { seed += 7; draw(); }, 'ghost')));
      draw();
    });


    return [
      P('You don’t train one robot. You start thousands of identical simulated ones, each in its own little world with slightly different physics, and 4096 is a typical count. It comes down to throughput: with 4096 robots you collect experience 4096 times faster than with one.'),
      H2('Terms'),
      UL('An <b>environment</b> is one simulated robot in one world.', 'An <b>episode</b> is a single attempt. The robot starts reset to a standing pose, and the episode ends when it falls (termination) or hits a maximum step count (truncation), whichever happens first. If it falls early, that env resets right away and starts a new episode.', 'A <b>rollout</b> is one chunk of collected experience. Every env runs for a fixed number of steps, say 24, and all that data gets pooled. With 4096 envs that’s 4096 × 24 = 98,304 transitions.', 'An <b>iteration</b> is one collect-then-train cycle: gather a rollout, then train on it.', 'The <b>training run</b> is every iteration, roughly 2000 of them.'),
      gridW,
      Trap('People sometimes call the whole training run a rollout, but a rollout is just one fixed-length window of collection. Inside it the episodes start and end at different times. One env might get through three short episodes because the robot keeps falling while another is halfway through one long one. It all gets pooled anyway.'),
      H2('Actor and critic'),
      P('Two networks are involved. The <b>actor</b> is the policy, the thing that decides what to do, and it’s the only part that ends up on the robot. The <b>critic</b> (the value function) predicts how good a situation is. It only exists to help training and gets thrown away afterward. While a rollout is being collected, neither network is learning. The weights are frozen and you just record what the current policy does.'),
      Match({ q: 'Match each definition to its term.', rows: [{ l: 'One simulated robot in one world', r: 'Environment' }, { l: 'Reset to standing, run until it falls or times out', r: 'Episode' }, { l: 'All envs × N steps, pooled', r: 'Rollout' }, { l: 'Collect a rollout, then train on it', r: 'Iteration' }, { l: 'All ~2000 of those cycles', r: 'Training run' }], why: 'Episodes live inside rollouts and don’t line up with their edges.' }),
      Check({ q: '4096 envs run for 24 steps each. How many transitions is that one rollout?', opts: ['24', '4,096', 'About 98,000', 'About 8 million'], ans: 2, why: '4096 × 24 = 98,304.' }),
      Check({ q: 'Which network is on the robot after training?', opts: ['The actor', 'The critic', 'Both'], ans: 0, why: 'The critic only exists to help training.' }),
    ];
  },
  cards: [
    { q: "Define environment, episode, rollout, iteration and training run.", a: "<b>Env</b>: one robot in one world. <b>Episode</b>: reset until it falls or times out. <b>Rollout</b>: all envs × N steps, pooled. <b>Iteration</b>: collect a rollout, then train on it. <b>Run</b>: all ~2000 iterations." },
    { q: "How many transitions are in a rollout of 4096 envs × 24 steps?", a: "About 98,000. A rollout is a chunk of experience, not the whole run." },
    { q: "What’s the difference between termination and truncation?", a: "Termination means the robot fell. Truncation means it reached the maximum step count. Whichever comes first ends the episode." },
    { q: "Which of the actor and the critic gets deployed?", a: "Only the actor. The critic is just for training and gets discarded." },
  ],
});

/* 2 ─ senses & actions */
lesson({
  id: 'w-senses', ch: 'walk', mins: 8,
  title: "Observations and actions",
  lede: "The actor is only given what a real robot could measure, and instead of one action it outputs a distribution to sample from.",
  build() {

    const grav = Widget('Projected gravity', body => {
      let pitch = 0, roll = 0; const holder = h('div'), out = h('div.readout');
      const fig = (cx, cy, ang, label, front) => {
        const g = S('g', { transform: `translate(${cx} ${cy})` });
        g.append(Line(0, 0, 0, 70, 'var(--ink3)', 3, { 'stroke-dasharray': '2 5' }), S('polygon', { points: '-6,64 6,64 0,76', fill: 'var(--ink3)' }), T(12, 78, 'gravity (world)', 'axis'));
        const b = S('g', { transform: `rotate(${ang})` });
        b.append(Rect(-16, -62, 32, 62, 'var(--c-soft)', { stroke: 'var(--c)', 'stroke-width': 2, rx: 7 }), Circle(0, -76, 11, 'var(--c-soft)', { stroke: 'var(--c)', 'stroke-width': 2 }), Line(0, 0, 0, -100, 'var(--c)', 1.5, { 'stroke-dasharray': '3 3' }));
        if (!front) b.append(Line(0, -76, 14, -76, 'var(--c)', 3));
        g.append(b, T(0, -112, label, 'axis', 'middle'), Circle(0, 0, 3.5, 'var(--ink)'));
        return g;
      };
      const draw = () => {
        const th = pitch * Math.PI / 180, ph = roll * Math.PI / 180;
        const gx = Math.sin(th), gy = -Math.sin(ph) * Math.cos(th), gz = -Math.cos(ph) * Math.cos(th);
        const svg = svgRoot(660, 220, { 'aria-label': 'Side and front view of a torso with the gravity vector in its own frame' });
        svg.append(fig(90, 130, pitch, 'side view (pitch)', false), fig(240, 130, -roll, 'front view (roll)', true));
        const bars = [['x forward', gx], ['y left', gy], ['z up', gz]], bx = 430, sc = 62;
        svg.append(T(bx, 24, 'gravity in the robot’s own frame', 'lbl-b'));
        bars.forEach(([nm, v], i) => {
          const y = 52 + i * 46;
          svg.append(Line(bx, y + 12, bx + sc * 2 + 60, y + 12, 'var(--line)', 1), Line(bx + sc + 30, y, bx + sc + 30, y + 24, 'var(--ink3)', 1), T(bx, y + 4, nm, 'axis'),
            Rect(v >= 0 ? bx + sc + 30 : bx + sc + 30 + v * sc, y + 6, Math.max(2, Math.abs(v) * sc), 12, i === 2 ? 'var(--ink3)' : 'var(--c)', { rx: 3 }), T(650, y + 17, fx(v), 'lbl', 'end'));
        });
        holder.innerHTML = ''; holder.append(svg);
        out.innerHTML = ''; out.append(Chip('vector', '(' + fx(gx) + ', ' + fx(gy) + ', ' + fx(gz) + ')', 'c'), Chip('|g|', '1.00'));
      };
      body.append(holder, Slider({ label: 'pitch (lean forward)', min: -45, max: 45, step: 1, value: 0, fmt: v => v + '°', on: v => { pitch = v; draw(); } }), Slider({ label: 'roll (lean sideways)', min: -45, max: 45, step: 1, value: 0, fmt: v => v + '°', on: v => { roll = v; draw(); } }), out,
        h('p.cap', null, 'Upright reads (0, 0, −1). As soon as the body leans, x or y becomes nonzero, before the robot has fallen. This is the early warning for balance.')); draw();
    });

    const gaus = Widget('Sampling an action', body => {
      let mu = .3, sig = .35, inf = false, zs = Array.from({ length: 14 }, (_, i) => gauss(rng(100 + i * 7)));
      const holder = h('div'), out = h('div.readout');
      const draw = () => {
        const p = Plot(660, 200, { x: [-1.5, 1.5], y: [0, 1.3], yt: [], xt: [-1, -.5, 0, .5, 1], xl: 'action for one joint (scaled position offset)', m: { l: 20, r: 20, t: 10, b: 36 } });
        const pdf = a => Math.exp(-.5 * ((a - mu) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
        const mx = pdf(mu), pts = []; for (let a = -1.5; a <= 1.5; a += .02) pts.push([p.sx(a), p.sy(pdf(a) / Math.max(mx, 1.2) * 1.15)]);
        p.svg.append(S('path', { d: path(pts) + ` L${p.sx(1.5)} ${p.sy(0)} L${p.sx(-1.5)} ${p.sy(0)}Z`, fill: 'var(--c-soft)', stroke: 'none' }), Path(path(pts), 'var(--c)', 2.5), Line(p.sx(mu), p.sy(0), p.sx(mu), p.sy(1.25), 'var(--c)', 1.5, { 'stroke-dasharray': '4 3' }), T(p.sx(mu) + 6, p.sy(1.2), 'mean', 'lbl'));
        zs.forEach((z, i) => { const a = inf ? mu : mu + sig * z; p.svg.append(Circle(clamp(p.sx(a), 24, 636), p.sy(0) - 8 - (inf ? i * 0 : (i % 3) * 9), 5.5, inf ? 'var(--ink)' : 'var(--hl)', { stroke: 'var(--surface)', 'stroke-width': 1.5 })); });
        holder.innerHTML = ''; holder.append(p.svg);
        out.innerHTML = ''; out.append(Chip('action', inf ? 'mean' : 'mean + std × noise', 'c'), Chip('μ', fx(mu)), Chip('σ', fx(sig)), Chip('mode', inf ? 'deployment: no sampling' : 'training: explore', inf ? 'ok' : 'hl'));
      };
      body.append(holder, Slider({ label: 'mean μ', min: -1, max: 1, step: .01, value: mu, on: v => { mu = v; draw(); } }), Slider({ label: 'std σ', min: .03, max: .8, step: .01, value: sig, on: v => { sig = v; draw(); } }),
        h('div.row', null, Seg([{ v: false, label: 'Training: sample' }, { v: true, label: 'Deployment: take the mean' }], false, v => { inf = v; draw(); }), Btn('Re-roll the noise', () => { zs = zs.map(() => gauss(Math.random)); draw(); }, 'ghost')), out,
        h('p.cap', null, 'Each dot is one env sampling its own action from the same Gaussian. Learning shifts the mean toward the samples that did better. σ starts large and shrinks as the policy becomes confident.')); draw();
    });

    return [
      H2('The observation'),
      P('MuJoCo knows the exact state of everything in the sim, but the actor doesn’t get all of it. You build the observation out of things real sensors provide: joint positions, joint velocities, base angular velocity from the IMU, projected gravity, the velocity command and the previous action. For a 25-DoF robot that adds up to 25 + 25 + 3 + 3 + 3 + 25 = 84 numbers. You decide what goes in and in what order. Get the order wrong, or leave out projected gravity, and nothing learns.'),
      H3('Projected gravity'),
      P('It’s the direction of “down” written in the robot’s own body frame. Standing upright it reads (0, 0, −1). Once the body leans, x or y becomes nonzero, so the robot can feel itself tilting before it actually falls. For balance it’s the most useful number in the whole observation.'),
      grav,
      H2('What the actor outputs'),
      P('The observation goes through the actor, an MLP with hidden layers like 512, 256 and 128. For each joint it outputs a mean and a standard deviation, which together describe a Gaussian, and the action gets sampled as mean + std × noise.'),
      P('The randomness is the exploration. A policy that did exactly the same thing every time it saw the same input could never find out that a slightly different action works better. Some samples come out better than others, and learning nudges the mean toward the better ones. The standard deviation starts large and shrinks as the policy gets more confident.'),
      gaus,
      P('On the real robot nothing is sampled. It takes the mean and ignores the standard deviation.'),
      H2('One step of collection'),
      P('For each env at each timestep you build the observation, run the actor, sample an action, apply it and step the physics, compute the reward, run the critic to get V(s), and store all of it. Advantage isn’t computed yet. That happens after the rollout.'),
      Eq('stored per step:  observation · action · log-probability of the action · reward · value estimate'),
      Check({ q: 'What does projected gravity read when the robot stands upright?', opts: ['(0, 0, 0)', '(0, 0, −1)', '(1, 0, 0)'], ans: 1, why: 'Down, in a frame where z points up, is (0, 0, −1).' }),
      Check({ q: 'Why does the actor output a mean and a standard deviation instead of a single action?', opts: ['It makes inference faster', 'Sampling around the mean explores, so the policy can find out that a different action is better', 'It cuts the number of parameters'], ans: 1, why: 'On the robot the mean is used and the standard deviation is ignored.' }),
    ];
  },
  cards: [
    { q: "What’s in the 84-number observation (25 DoF)?", a: "Joint positions (25), joint velocities (25), base angular velocity (3), projected gravity (3), velocity command (3), last action (25)." },
    { q: "What is projected gravity?", a: "The direction of down in the robot’s own body frame. Upright it reads (0, 0, -1). It lets the robot feel a tilt before it falls." },
    { q: "What does the actor output?", a: "A mean and a standard deviation for each joint. Training samples mean + std × noise for exploration. Deployment just uses the mean." },
    { q: "What gets stored per step in the rollout buffer?", a: "The observation, the action, the log-probability of the action, the reward and the value estimate." },
  ],
});

/* 3 ─ reward, return, value */
lesson({
  id: 'w-reward', ch: 'walk', mins: 7,
  title: "Reward, return and value",
  lede: "Reward is one number for one timestep. Return adds up everything that comes after it. Value is the critic’s guess at that sum.",
  build() {
    const presets = { steady: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], prize: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10], stumble: [1, 1, 1, 1, 1, -5, 0, 0, 0, 0, 0, 0] };
    const wid = Widget('Reward vs return', body => {
      let pre = 'steady', gm = .9, t = 0; const holder = h('div'), eq = h('div.eq'), out = h('div.readout');
      const draw = () => {
        const r = presets[pre], n = r.length, R = new Array(n + 1).fill(0);
        for (let i = n - 1; i >= 0; i--) R[i] = r[i] + gm * R[i + 1];
        const ymax = Math.max(1, ...R.slice(0, n), ...r) * 1.15, ymin = Math.min(0, ...R.slice(0, n), ...r) * 1.15;
        const p = Plot(660, 230, { x: [-.6, n - .4], y: [ymin, ymax], yt: [0, Math.round(ymax / 2), Math.round(ymax)], xt: r.map((_, i) => i), xl: 'timestep t', yl: 'value', m: { l: 44 } });
        r.forEach((v, i) => p.svg.append(Rect(p.sx(i) - 13, Math.min(p.sy(v), p.sy(0)), 26, Math.abs(p.sy(v) - p.sy(0)), 'var(--line)', { rx: 3 })));
        p.svg.append(Path(path(R.slice(0, n).map((v, i) => [p.sx(i), p.sy(v)])), 'var(--c)', 2.5));
        R.slice(0, n).forEach((v, i) => p.svg.append(Circle(p.sx(i), p.sy(v), i === t ? 7 : 4, i === t ? 'var(--hl)' : 'var(--c)', { stroke: 'var(--surface)', 'stroke-width': 2 })));
        p.svg.append(T(p.m.l + 6, p.m.t + 12, '▮ reward r_t', 'lbl', 'start', { style: 'fill:var(--ink3)' }), T(p.m.l + 6, p.m.t + 28, '● return R_t', 'lbl', 'start', { style: 'fill:var(--c)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        const terms = r.slice(t).map((v, k) => (k === 0 ? '' : ' + ' + fx(Math.pow(gm, k)) + '·') + v);
        eq.innerHTML = '<span class="k">R_' + t + '</span> = ' + terms.slice(0, 6).join('') + (terms.length > 6 ? ' + …' : '') + '  =  <span class="k">' + fx(R[t]) + '</span>';
        out.innerHTML = ''; out.append(Chip('reward r_' + t, fx(r[t]), ''), Chip('return R_' + t, fx(R[t]), 'c'), Chip('effective horizon 1/(1−γ)', gm >= .999 ? '∞' : fx(1 / (1 - gm), 0) + ' steps', 'hl'));
      };
      body.append(Seg([{ v: 'steady', label: 'Walking steadily' }, { v: 'prize', label: 'Prize at the end' }, { v: 'stumble', label: 'A stumble at t=5' }], pre, v => { pre = v; draw(); }), holder,
        Slider({ label: 'discount γ', min: 0, max: .99, step: .01, value: gm, on: v => { gm = v; draw(); } }), Slider({ label: 'look from t', min: 0, max: 11, step: 1, value: 0, fmt: v => 't = ' + v, on: v => { t = v; draw(); } }), eq, out); draw();
    });

    return [
      UL('<b>Reward r_t</b> is a single number for a single timestep, basically “did that step go well.”', '<b>Return R_t</b> is the total discounted reward from step t onward: r_t + γ·r_{t+1} + γ²·r_{t+2} + …, with γ around 0.99 so later rewards count a bit less.', '<b>Value V(s)</b> is what the critic outputs. It predicts the return rather than the reward. For a given state, it’s roughly how much total reward is still coming.'),
      wid,
      Trap('It’s easy to read V(s) as a prediction of the next reward. What the critic actually estimates is the whole discounted sum that comes after, and a good V(s) ends up close to the return the robot really gets.'),
      P('It doesn’t look ahead by simulating anything. After seeing millions of states it has learned what usually happens next, the same way you can watch someone mid-stumble and know they’re going down without working out the physics. That knowledge sits in the weights.'),
      P('With γ = 0.99 the effective horizon is about 1/(1 − γ) = 100 steps, which is around two seconds at a 50 Hz policy rate.'),
      Check({ q: 'What is the critic trained to match?', opts: ['The next reward', 'The return', 'The advantage'], ans: 1, why: 'L_critic = (V(s_t) − R_t)², so it’s regression onto the return.' }),
      Check({ q: 'Every reward is 0 except a 10 at step 11 of 12. With γ = 0.9, what’s the return at step 0?', opts: ['0', '10', 'About 3.1', '120'], ans: 2, why: '0.9^11 × 10 ≈ 3.14. The prize gets discounted once for every step it’s away.' }),
    ];
  },
  cards: [
    { q: "What’s the difference between reward, return and value?", a: "<b>Reward</b> is one timestep’s number. <b>Return</b> is the discounted sum of all future rewards. <b>Value</b> is the critic’s prediction of the return." },
    { q: "What does γ do?", a: "It discounts the future. The effective horizon is about 1/(1 - γ) steps." },
    { q: "How does the critic look ahead without simulating?", a: "It learned from millions of past states what usually happens next, and that’s stored in its weights." },
  ],
});

/* 4 ─ TD, advantage, GAE */
lesson({
  id: 'w-advantage', ch: 'walk', mins: 10,
  title: "TD error, advantage and GAE",
  lede: "The policy gets updated using advantage, which measures how much better or worse an action turned out than what the policy would usually do from that state. Here’s how you get there from the TD error.",
  build() {
    const G = .99;
    const R0 = [1, 1, 1, 1, -3, 1, 1, 1], V0 = [4.0, 3.6, 3.2, 2.5, .5, 5.0, 5.2, 5.1], D0 = [0, 0, 0, 0, 1, 0, 0, 0], VLAST = 5.0;
    const wid = Widget('One env’s rollout, 8 steps, two episodes', body => {
      let lam = .95, cut = true, sel = 3; const tbl = h('div.scroll-x'), msg = h('div.eq'), out = h('div.readout');
      const compute = () => {
        const n = R0.length, dl = [], A = new Array(n).fill(0);
        for (let t = n - 1; t >= 0; t--) {
          const nv = t === n - 1 ? VLAST : V0[t + 1], nt = cut && D0[t] ? 0 : 1;
          dl[t] = R0[t] + G * nv * nt - V0[t]; A[t] = dl[t] + G * lam * nt * (t === n - 1 ? 0 : A[t + 1]);
        } return { dl, A, Rt: A.map((a, i) => a + V0[i]) };
      };
      const draw = () => {
        const { dl, A, Rt } = compute(), mx = Math.max(...A.map(Math.abs), 1);
        const head = h('tr', null, ['t', 'reward', 'V(s)', 'done', 'δ (TD)', 'A (GAE)', 'return = A+V'].map(x => h('th', null, x)));
        const rows = R0.map((r, t) => h('tr', { style: { background: t === sel ? 'var(--c-soft)' : (D0[t] ? 'var(--bad-soft)' : null), cursor: 'pointer' }, on: { click() { sel = t; draw(); } } },
          h('td', null, t), h('td', null, fx(r, 1)), h('td', null, fx(V0[t], 1)), h('td', null, D0[t] ? '✕ fell' : '·'), h('td', null, fx(dl[t])),
          h('td', null, h('span', { style: { display: 'inline-block', width: Math.max(2, Math.abs(A[t]) / mx * 60) + 'px', height: '9px', borderRadius: '3px', background: A[t] >= 0 ? 'var(--ok)' : 'var(--bad)', marginRight: '8px', verticalAlign: 'middle' } }), fx(A[t])), h('td', null, fx(Rt[t]))));
        tbl.innerHTML = ''; tbl.append(h('table.tt', null, h('thead', null, head), h('tbody', null, rows)));
        const nv = sel === R0.length - 1 ? VLAST : V0[sel + 1], nt = cut && D0[sel] ? 0 : 1;
        msg.innerHTML = '<span class="k">δ_' + sel + '</span> = r + γ·V(s′)·' + (D0[sel] ? (cut ? '<b>0</b>' : '1') : '1') + ' − V(s) = ' + fx(R0[sel], 1) + ' + ' + G + '×' + fx(nv, 1) + '×' + nt + ' − ' + fx(V0[sel], 1) + ' = <span class="k">' + fx(dl[sel]) + '</span>\n' +
          '<span class="k">A_' + sel + '</span> = δ_' + sel + ' + γλ·A_' + (sel + 1) + ' = <span class="k">' + fx(A[sel]) + '</span>   →   ' + (A[sel] > .05 ? 'better than expected: do more' : A[sel] < -.05 ? 'worse than expected: do less' : 'about as expected');
        out.innerHTML = ''; out.append(Chip('advantage of the fall step', fx(A[4]), A[4] < 0 ? 'ok' : 'bad'), Chip('advantage of the step before it', fx(A[3]), A[3] < 0 ? 'ok' : 'bad'), Chip(cut ? 'boundary cut' : 'LEAKING', cut ? 'correct' : 'wrong', cut ? 'ok' : 'bad'));
      };
      body.append(h('div.row', null, Seg([{ v: true, label: 'Cut recursion at done flags' }, { v: false, label: 'Don’t cut (leak)' }], true, v => { cut = v; draw(); })),
        Slider({ label: 'GAE λ', min: 0, max: 1, step: .01, value: lam, on: v => { lam = v; draw(); } }), tbl, msg, out, h('p.cap', null, 'Click a row to see its numbers. γ = 0.99. The robot falls at step 4 and a new episode starts at step 5.')); draw();
    });

    return [
      H2('TD error'),
      P('The critic said the return from s_t would be V(s_t). Then the robot got reward r_t and landed in a state the critic values at V(s_{t+1}). That gives a better estimate of the return, r_t + γ·V(s_{t+1}), and the gap between it and the original guess is the TD error.'),
      Eq('<span class="k">δ_t</span> = r_t + γ·V(s_{t+1}) − V(s_t)'),
      P('Positive means things went better than the critic expected, negative means worse. It’s a one-step correction to the critic’s prediction.'),
      H2('Advantage'),
      P('Advantage asks whether an action was better or worse than the policy’s average from that state. If it’s positive you want more of that action, and if it’s negative you want less. This number drives the policy update.'),
      H2('GAE'),
      P('A single TD error is short-sighted and noisy. Adding up every reward to the end of the episode has no bias but a lot of variance. GAE sits between the two, as a discounted sum of future TD errors.'),
      Eq('<span class="k">A_t</span> = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + …\n    = δ_t + γλ·A_{t+1}          computed backward through the rollout, one pass'),
      UL('λ near 0 is close to the one-step TD error, with low variance and more bias.', 'λ near 1 looks far ahead, with less bias and more variance.', 'In practice λ is usually 0.95.'),
      P('Once the backward pass is done, the critic’s training target comes for free: R_t = A_t + V(s_t). The table below is one env’s rollout with a fall at step 4.'),
      wid,
      Trap('The recursion has to stop at episode boundaries, so you zero the bootstrap at done flags. If you don’t, value from the next episode leaks backward and the fall looks fine. Nothing crashes. The reward curve just never improves, which makes it hard to spot.'),
      Order({ q: 'Put the data pipeline after a rollout in order.', items: ['Collect the rollout with frozen networks: observation, action, log-probability, reward, value', 'Compute TD errors from the rewards and values', 'Run GAE backward, zeroing at done flags, to get advantages', 'Compute returns as advantage + value', 'Shuffle into minibatches and train for several epochs'], why: 'Returns come from R_t = A_t + V(s_t).' }),
      Check({ q: 'With the boundary cut, why does the step just before the fall (t = 3) have a strongly negative advantage?', opts: ['Its own TD error is large', 'GAE adds in the very negative future TD errors, so the actions leading up to the fall take the blame', 'The critic is wrong there'], ans: 1, why: 'Its own δ is only about −1, but A_3 = δ_3 + γλ·A_4 inherits the fall’s large negative advantage.' }),
    ];
  },
  cards: [
    { q: "Write the TD error.", a: "<b>δ_t = r_t + γV(s_{t+1}) - V(s_t)</b>. It’s a one-step correction to the critic’s return prediction." },
    { q: "What is advantage?", a: "How much better or worse an action was than the policy’s average from that state. It drives the policy update." },
    { q: "What are the GAE formula and its recursion?", a: "<b>A_t = δ_t + γλ·δ_{t+1} + (γλ)²δ_{t+2} + …</b>, which is the same as <b>δ_t + γλ·A_{t+1}</b>, computed in one backward pass." },
    { q: "What does λ control?", a: "How far ahead the TD errors are weighted. Near 0 it’s low variance but biased, near 1 it’s less biased but noisier. Usually 0.95." },
    { q: "What do you do at episode boundaries in GAE?", a: "Stop the recursion and zero the bootstrap at done flags, so the next episode’s value doesn’t leak backward." },
    { q: "How do you get the critic’s target once you have advantages?", a: "R_t = A_t + V(s_t)." },
  ],
});

/* 5 ─ PPO clip */
lesson({
  id: 'w-ppo', ch: 'walk', mins: 8,
  title: "The PPO objective",
  lede: "PPO takes several gradient steps on one batch of data while keeping the policy from wandering far from the one that collected it.",
  build() {
    const clipW = Widget('The clipped objective', body => {
      let A = 1, eps = .2, r = 1.15; const holder = h('div'), out = h('div.readout'), msg = h('p.cap');
      const draw = () => {
        const p = Plot(660, 250, { x: [.4, 1.6], y: [-3.4, 3.4], xt: [.4, .6, .8, 1, 1.2, 1.4, 1.6], yt: [-3, -2, -1, 0, 1, 2, 3], xl: 'ratio r = π_new(a|s) / π_old(a|s)', yl: 'objective', m: { l: 40 } });
        const cl = x => clamp(x, 1 - eps, 1 + eps), U = x => x * A, C = x => cl(x) * A, M = x => Math.min(U(x), C(x));
        const line = (f, s, w, d) => Path(path(Array.from({ length: 61 }, (_, i) => { const x = .4 + i * .02; return [p.sx(x), p.sy(clamp(f(x), -3.4, 3.4))]; })), s, w, 'none', d ? { 'stroke-dasharray': d } : {});
        p.svg.append(Rect(p.sx(1 - eps), p.m.t, p.sx(1 + eps) - p.sx(1 - eps), 250 - p.m.t - p.m.b, 'var(--c-soft)', { opacity: .6 }), line(U, 'var(--ink3)', 1.5, '5 4'), line(C, 'var(--hl)', 2, '3 3'), line(M, 'var(--c)', 3.5));
        p.svg.append(Circle(p.sx(r), p.sy(clamp(M(r), -3.4, 3.4)), 7, 'var(--ink)', { stroke: 'var(--surface)', 'stroke-width': 2 }), T(p.sx(1 - eps), p.m.t + 12, '1−ε', 'axis', 'middle'), T(p.sx(1 + eps), p.m.t + 12, '1+ε', 'axis', 'middle'));
        p.svg.append(T(p.m.l + 6, 250 - p.m.b - 8, '- - r·A     ┄ clip(r)·A     ━ min(...) = what PPO maximizes', 'lbl'));
        holder.innerHTML = ''; holder.append(p.svg);
        const u = U(r), c = C(r), flat = (A > 0 && r > 1 + eps) || (A < 0 && r < 1 - eps);
        out.innerHTML = ''; out.append(Chip('u = r·A', fx(u)), Chip('c = clip·A', fx(c)), Chip('min(u, c)', fx(Math.min(u, c)), 'c'), Chip('L_actor = −min', fx(-Math.min(u, c))), Chip('gradient', flat ? 'zero (flat)' : 'flows', flat ? 'bad' : 'ok'));
        msg.innerHTML = flat ? 'The policy has already moved as far as it is allowed on this sample. Past the clip the gradient is zero, so there is no incentive to push further.' : (A === 0 ? 'Advantage is zero: nothing to reinforce.' : A > 0 ? 'Positive advantage: the gradient raises this action’s probability until r reaches 1+ε.' : 'Negative advantage: the gradient lowers its probability until r reaches 1−ε.');
      };
      body.append(holder, Slider({ label: 'advantage A', min: -2, max: 2, step: .1, value: A, on: v => { A = v; draw(); } }), Slider({ label: 'clip ε', min: .05, max: .4, step: .01, value: eps, on: v => { eps = v; draw(); } }), Slider({ label: 'ratio r', min: .4, max: 1.6, step: .01, value: r, on: v => { r = v; draw(); } }), out, msg); draw();
    });

    return [
      H2('Epochs, and why clipping exists'),
      P('The rollout (about 98,000 transitions) gets shuffled and cut into minibatches of a few hundred. Neighboring samples come from the same episode and are correlated, which makes gradients biased and unstable, and shuffling breaks that up. One pass through all the transitions is an epoch, and PPO does around 5 epochs on the same rollout before throwing it away. Reusing data like that is what makes it efficient, but it also means the later updates use data from a policy that has already changed, so you have to limit how far it moves.'),
      Eq('r = π_new(a|s) / π_old(a|s)          1 means unchanged, 1.3 means 30% more likely\nu = r · A\nc = clip(r, 1−ε, 1+ε) · A\n<span class="k">L_actor = − min(u, c)</span>            ε ≈ 0.2'),
      P('π_old collected the data and stays frozen, and π_new is the policy being trained. Pushing up the probability of good actions and pushing down bad ones is just r·A. Clipping caps how much any one sample can help. Past 1 + ε (for positive advantage) or below 1 − ε (for negative), the objective goes flat and the gradient is zero. The minus sign is only there because optimizers minimize.'),
      clipW,
      Trap('Clipping doesn’t physically stop the ratio from changing. A clipped sample just stops contributing gradient, and other samples can still push the ratio a bit past the boundary.'),
      H2('Critic loss and total loss'),
      Eq('L_critic = (V(s_t) − R_t)²                     plain regression onto the return\n<span class="k">L = L_actor + w₁·L_critic − w₂·entropy</span>        w₁ ≈ 0.5, w₂ ≈ 0.01'),
      P('Entropy measures how spread out the policy’s distribution is. Subtracting it in the loss rewards a wider distribution, which stops the policy from settling on one deterministic action before it’s done exploring. So one loss trains both networks and keeps exploration going.'),
      Check({ q: 'A sample has positive advantage and the ratio is already 1.5 with ε = 0.2. What’s the gradient from that sample?', opts: ['Large and positive', 'Zero, because the objective is clipped flat there', 'Negative'], ans: 1, why: 'Past 1 + ε with A > 0, min(r·A, clip(r)·A) is a constant.' }),
      Check({ q: 'Why does PPO need clipping?', opts: ['It reuses the same data for several epochs, so later updates use data from an older policy', 'It has two networks', 'It samples actions'], ans: 0, why: 'Clipping limits how far the policy can drift from the one that collected the data.' }),
      Check({ q: 'What does the entropy term do in the total loss?', opts: ['Speeds up the critic', 'Keeps the policy from becoming deterministic too early', 'Clips the ratio'], ans: 1, why: 'Since entropy is subtracted, a wider distribution lowers the loss.' }),
    ];
  },
  cards: [
    { q: "Write the PPO clipped actor loss.", a: "<b>r = π_new/π_old; L = -min(r·A, clip(r, 1-ε, 1+ε)·A)</b>, with ε around 0.2." },
    { q: "What does PPO do, in one sentence?", a: "It takes multiple gradient steps on a batch of data without letting the policy move far from the one that collected it." },
    { q: "Why is there a minus sign in the actor loss?", a: "Optimizers minimize, and the clipped objective is something you want to maximize." },
    { q: "What’s an epoch in PPO?", a: "One pass through every transition of the shuffled rollout, in minibatches. It does about 5 epochs per rollout." },
    { q: "What’s the total loss and what does each term do?", a: "<b>L = L_actor + 0.5·L_critic - 0.01·entropy</b>. The terms improve the policy, improve the value predictions and keep it exploring." },
    { q: "Why shuffle the rollout?", a: "Neighboring samples are correlated. Shuffling breaks that up so gradients are less biased and more stable." },
  ],
});
