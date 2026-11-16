'use strict';
/* ═════════ Chapter 2 — Arms and Hands ═════════ */

/* ── planar IK math ── */
const IKL = [104, 86, 62];
function ikFK(th) { let x = 0, y = 0, a = 0; const pts = [[0, 0]]; th.forEach((t, i) => { a += t; x += IKL[i] * Math.cos(a); y += IKL[i] * Math.sin(a); pts.push([x, y]); }); return pts; }
function ikJ(th) { const pts = ikFK(th), ee = pts[pts.length - 1]; const J = [[], []]; th.forEach((_, j) => { J[0][j] = -(ee[1] - pts[j][1]); J[1][j] = ee[0] - pts[j][0]; }); return J; }
function ikStep(th, tgt, mode, gain = .5, lam = 6) {
  const pts = ikFK(th), ee = pts[pts.length - 1], e = [(tgt[0] - ee[0]) * gain, (tgt[1] - ee[1]) * gain], J = ikJ(th);
  if (mode === 'one') { const c = [J[0][0], J[1][0]], d = c[0] * c[0] + c[1] * c[1] + lam * lam, k = (c[0] * e[0] + c[1] * e[1]) / d; return [th[0] + k, th[1], th[2]]; }
  const a = J[0].reduce((s, v) => s + v * v, 0) + lam * lam, b = J[0].reduce((s, v, i) => s + v * J[1][i], 0), d = J[1].reduce((s, v) => s + v * v, 0) + lam * lam, det = a * d - b * b;
  const y0 = (d * e[0] - b * e[1]) / det, y1 = (-b * e[0] + a * e[1]) / det;
  return th.map((t, j) => t + J[0][j] * y0 + J[1][j] * y1);
}

/* 1 ─ IK */
lesson({
  id: 'a-ik', ch: 'arms', mins: 9,
  title: "Inverse kinematics",
  lede: "The first rung of manipulation doesn’t involve learning. You’re given a target for the hand and you work out what joint angles put it there.",
  build() {
    const ik = Widget('Inverse kinematics on a 3-joint arm', body => {
      let th = [1.05, -.7, -.55], tgt = [190, 105], mode = 'all', stop = null; const BX = 120, BY = 262, W = 660, Hh = 300;
      const svg = svgRoot(W, Hh, { 'aria-label': 'Three-link planar arm reaching for a draggable target' }), tbl = h('div.scroll-x'), out = h('div.readout');
      const X = p => BX + p[0], Y = p => BY - p[1];
      const draw = () => {
        svg.innerHTML = '';
        const pts = ikFK(th), ee = pts[3], reach = IKL.reduce((a, b) => a + b, 0);
        svg.append(S('circle', { cx: BX, cy: BY, r: reach, fill: 'none', stroke: 'var(--line)', 'stroke-dasharray': '4 5', 'stroke-width': 1.5 }), Rect(BX - 30, BY + 6, 60, 10, 'var(--ink3)', { rx: 3 }));
        for (let i = 0; i < 3; i++) svg.append(Line(X(pts[i]), Y(pts[i]), X(pts[i + 1]), Y(pts[i + 1]), 'var(--c)', 11 - i * 2, { opacity: .85 }));
        pts.slice(0, 3).forEach(p => svg.append(Circle(X(p), Y(p), 8, 'var(--surface)', { stroke: 'var(--ink)', 'stroke-width': 2.5 })));
        svg.append(Circle(X(ee), Y(ee), 7, 'var(--ink)'), Line(BX + tgt[0] - 11, BY - tgt[1], BX + tgt[0] + 11, BY - tgt[1], 'var(--hl)', 3), Line(BX + tgt[0], BY - tgt[1] - 11, BX + tgt[0], BY - tgt[1] + 11, 'var(--hl)', 3), Circle(BX + tgt[0], BY - tgt[1], 15, 'none', { stroke: 'var(--hl)', 'stroke-width': 2 }), T(14, 22, 'drag to move the target', 'lbl', 'start', { style: 'fill:var(--ink3)' }));
        const err = Math.hypot(tgt[0] - ee[0], tgt[1] - ee[1]), J = ikJ(th), far = Math.hypot(tgt[0], tgt[1]) > reach;
        tbl.innerHTML = ''; tbl.append(h('table.tt', null, h('thead', null, h('tr', null, ['J (2×3)', 'joint 1', 'joint 2', 'joint 3'].map(x => h('th', null, x)))), h('tbody', null, ['ẋ', 'ẏ'].map((nm, r) => h('tr', null, h('td', null, nm), J[r].map(v => h('td', null, fx(v, 0))))))));
        out.innerHTML = ''; out.append(Chip('error', fx(err, 1) + ' px', err < 2 ? 'ok' : 'c'), far ? Chip('target out of reach', 'no solution', 'bad') : Chip('DOF vs task dims', '3 vs 2: many solutions', 'hl'));
      };
      const setT = p => { tgt = [clamp(p.x - BX, -110, 500), clamp(BY - p.y, -30, 270)]; draw(); };
      drag(svg, setT);
      const step = () => { th = ikStep(th, tgt, mode); draw(); };
      const solve = () => { if (stop) stop(); let n = 0; stop = loop(() => { for (let i = 0; i < 2; i++) th = ikStep(th, tgt, mode); draw(); return ++n < 160; }); };
      body.append(svg, tbl, out, h('div.row', null, Seg([{ v: 'all', label: 'Pseudoinverse: all joints, together' }, { v: 'one', label: 'Nudge joint 1 only' }], 'all', v => { mode = v; }), Btn('One step', step), Btn('Solve', solve), Btn('Reset arm', () => { if (stop) stop(); th = [1.05, -.7, -.55]; draw(); }, 'ghost')),
        h('p.cap', null, 'Each step: forward kinematics, the error to the target, the Jacobian at the current angles, then the pseudoinverse turns the hand error into a change across every joint. In “joint 1 only” mode a single joint cannot get the hand there.')); draw();
    });

    return [
      H2('Forward and inverse'),
      P('<b>Forward kinematics</b> goes from joint angles to hand pose. You chain the transform for each joint (a rotation and a translation) from the base out to the end effector and you get the hand’s position and orientation. Every joint has an encoder, so you always have the angles.'),
      P('<b>Inverse kinematics</b> goes the other way. Given where you want the hand, find joint angles that get it there. There’s no clean formula in general. Different configurations can give the same hand pose, and sometimes nothing reaches the target at all, so you solve it iteratively.'),
      H3('The Jacobian'),
      P('The Jacobian relates joint motion to hand motion. If each degree of freedom moves at some rate, how fast and in what direction does the hand move? For a 7-DOF arm controlling a 6-D pose (3 for position, 3 for orientation) it’s a 6×7 matrix. The arm below has 3 joints and a 2-D target, so its Jacobian is 2×3.'),
      Trap('A joint and a degree of freedom aren’t the same thing. A hinge has 1 DOF, while a ball joint is a single joint with 3. The Jacobian has one column per DOF, not per joint. They only match when everything is a hinge, which is why people mix the two words up.'),
      Eq('loop:  <span class="k">FK</span> gives the hand position → error = target − hand → <span class="k">J</span> at the current angles → Δq = <span class="k">J⁺</span> · error → small step → repeat'),
      P('The pseudoinverse J⁺ converts the hand error into a change for all the joints at once, picking the smallest total joint motion that moves the hand toward the target. It doesn’t move a single joint. All of them move a little, together.'),
      ik,
      H2('What IK needs to sense'),
      P('You need joint encoders (every motor has one) and a known base pose. For an arm bolted to a table the base never moves, so its pose is a constant and you don’t need an IMU at all. You only need an IMU, or really a full state estimator, when the base floats, like a humanoid’s torso. Even then IK just uses the base pose the estimator already produces.'),
      P('The camera is a separate need. A pose estimator like FoundationPose takes a camera image plus a 3D model of the object and reports where the object is. That finds the target, not the arm. The encoders tell you where the arm is, the camera tells you where the target is, and IK works out the joint motion that closes the gap.'),
      Match({ q: 'What supplies each of these?', rows: [{ l: 'Where the arm’s joints currently are', r: 'Joint encoders' }, { l: 'Where the object to grasp is', r: 'Camera and pose estimator (FoundationPose)' }, { l: 'Base pose of an arm bolted to a table', r: 'A known constant, no IMU' }, { l: 'Base pose of a humanoid’s moving torso', r: 'The state estimator (IMU + kinematics)' }], why: 'FoundationPose finds the target, not the arm.' }),
      H2('Where it stops'),
      P('IK gets the hand to a pose, but it can’t grasp anything. When to close the fingers, how to adapt to the object’s shape and what to do when it slips are hard to write as geometry, and that gap is what the learning methods are for.'),
      Note('IK works for any target position because it’s only math, but it has nothing to say about contact. Learned policies handle contact and struggle with new positions. Real systems mix the two, which the hybrid lesson gets to.'),
      P('The joint angles IK produces are position targets for the arm. They don’t go straight to the motors. The arm controller in the Controls chapter turns them into torques, and it has to account for gravity, because an arm in free space sags without that.'),
      Check({ q: 'For a 7-joint arm and a 6-D target pose, the pseudoinverse update changes:', opts: ['One joint per iteration', 'All the joints at once, with the smallest total joint motion', 'Only the wrist'], ans: 1, why: 'It’s a coordinated change across all the DOFs.' }),
    ];
  },
  cards: [
    { q: "What’s the difference between forward and inverse kinematics?", a: "<b>FK</b> goes from joint angles to hand pose, using encoders, and is always available. <b>IK</b> goes from a desired hand pose to joint angles. There’s no clean formula, so it’s solved iteratively." },
    { q: "What’s the Jacobian, and what shape is it for a 7-DOF arm with a 6-D pose?", a: "It relates joint velocities to hand velocity. It’s <b>6×7</b>, with one column per DOF rather than per joint." },
    { q: "What does the pseudoinverse IK step do?", a: "It changes all the DOFs together, using the smallest total joint motion that moves the hand toward the target. Take a small step, recompute, repeat." },
    { q: "What sensors does IK need?", a: "Encoders always, plus a known base pose. A fixed arm has a constant base, so no IMU. A floating base uses the state estimator. The camera finds the target." },
    { q: "What are IK’s strength and limit?", a: "It works for any position but can’t do dexterous contact. Learning is the reverse. Use both." },
  ],
});

/* 2 ─ ACT */
lesson({
  id: 'a-act', ch: 'arms', mins: 9,
  title: "ACT",
  lede: "Imitation learning has no reward. You show the robot the task and it learns to copy you. ACT predicts a chunk of actions at a time and uses a CVAE built from transformers.",
  build() {

    const cvae = Widget('The CVAE: what runs when', body => {
      let mode = 'train'; const stage = h('div');
      const box = (t, d, dim, strong) => h('div', { style: { border: '1.5px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'var(--surface)', opacity: dim ? .35 : 1, borderRadius: '10px', padding: '9px 12px', textDecoration: dim ? 'line-through' : 'none' } }, h('b', { style: { fontSize: '14.5px' } }, t), h('div', { style: { fontSize: '13px', color: 'var(--ink3)', textDecoration: 'none' } }, d));
      const draw = () => {
        const tr = mode === 'train';
        stage.innerHTML = ''; stage.append(h('div.grid2', null,
          h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } },
            box('Encoder', 'looks at the real 100-action demo and the state and squeezes them into a small latent', !tr, tr),
            h('div.mono', { style: { textAlign: 'center' } }, tr ? '↓ z, about 32 numbers describing the style of this demo (a fast grab, a slow approach curving left, and so on)' : '↓ there’s no demo to encode, so z = 0, the center of N(0, 1)'),
            box('z', tr ? 'computed from the demonstration' : 'set to zero, which asks for the average style', false, false)),
          h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } },
            box('Decoder (transformer)', 'camera images (ResNet) + joint positions + z → 100 actions', false, true),
            h('div.mono', { style: { textAlign: 'center' } }, '↓'),
            box('Predicted action chunk', tr ? 'compared with the demonstrated chunk' : 'executed on the robot', false, false))),
          tr ? h('div.eq', { html: 'loss = <span class="k">reconstruction</span> + β · <span class="k">KL</span>\n  reconstruction = |predicted chunk − demonstrated chunk|    (L1: crisper actions than L2)\n  KL             = how far z strays from a standard normal' }) : h('div.eq', { html: 'deployment = decoder(observation, <span class="k">z = 0</span>)   → one forward pass, the <span class="k">average</span> style' }));
      };
      body.append(Seg([{ v: 'train', label: 'Training' }, { v: 'infer', label: 'Inference' }], 'train', v => { mode = v; draw(); }), stage); draw();
    });

    return [
      H2('Demonstrations'),
      P('You teleoperate the robot through the task, often with leader arms (you move a small master arm and the real one mirrors it), and record camera images and joint positions at every timestep. One demonstration is one complete run, and you collect somewhere from 50 to a couple hundred.'),
      H2('Chunking'),
      P('If a policy only predicts the next single action, small errors pile up. The robot drifts a little off the demonstrated path, ends up in a state it never saw in training, and falls apart. Human demos also have pauses and jitter, and single-step prediction turns those into shaky motion. ACT predicts the next 100 actions in one go. Sticking to one multi-step plan instead of reacting step by step cuts down the compounding error.'),
      H2('The CVAE'),
      P('A conditional variational autoencoder has an encoder and a decoder that do different jobs, and only the decoder is used on the robot. Switch between the two phases below.'),
      cvae,
      P('The part that confuses everyone is this: if z is set to zero at inference, why compute it during training at all? Human demonstrations are inconsistent, since the same task gets done a little differently each time. If the decoder had to map an observation straight to a chunk, it would see the same input with different outputs and produce a blur. z soaks up that variation by telling the decoder which style this particular demo used, so it can explain each demo cleanly.'),
      P('The loss has two parts. Reconstruction is the L1 (absolute) difference between the predicted chunk and the demonstrated one. L1 is used over L2 because it averages less aggressively and gives crisper actions. The KL term keeps z close to a standard normal distribution, and that’s what makes zero a sensible stand-in at inference, when there’s no demonstration to compute z from.'),
      Eq('loss = reconstruction + β · KL'),
      Trap('Training needs z. At inference you swap it for its mean because there’s no demo to compute it from, and that swap is exactly why ACT averages.'),
      Note('ACT’s output is a chunk of joint position targets. Like the locomotion policy’s output, they go to the arm controller in the control layer, which turns them into torques.'),
      H2('The weakness'),
      P('At deployment ACT always asks for the average style. Small variations are fine. Approaches from 8 cm and 12 cm average out to 10 cm, and that works. But if half the demos come in from the left and half from the right, both valid, ACT averages them and drives down the middle, which might miss completely. That’s the multimodality problem, and it’s the reason for the next method.'),
      Check({ q: 'Why does ACT predict 100 actions at once?', opts: ['To run faster', 'Sticking to a multi-step plan cuts compounding error and jitter', 'To need fewer demonstrations'], ans: 1, why: 'A one-step predictor can drift into states it never saw, and it amplifies jitter from the demos.' }),
      Check({ q: 'Why compute z in training if it’s set to zero at inference?', opts: ['It’s a leftover and could be removed', 'Demos are inconsistent, and z absorbs the variation so the decoder doesn’t average into a blur', 'To speed up inference'], ans: 1, why: 'KL keeps z near N(0, 1), which makes z = 0 a typical value at inference.' }),
      Check({ q: 'Half the demos grab from the left and half from the right. What does ACT do at deployment?', opts: ['Picks one at random', 'Averages them and approaches down the middle', 'Refuses to move'], ans: 1, why: 'One forward pass with z = 0 gives the average action.' }),
    ];
  },
  cards: [
    { q: "What does ACT stand for, and what do the two halves handle?", a: "<b>Action Chunking with Transformers</b>. Chunking predicts ~100 actions at once to fight compounding error and jitter. The CVAE decoder takes the observation plus a latent." },
    { q: "What does ACT’s CVAE do in training and at inference?", a: "<b>Training</b>: the encoder squeezes the demo chunk and state into z, and the decoder rebuilds the chunk from the observation and z. <b>Inference</b>: no encoder, z = 0, and the decoder gives an average-style chunk." },
    { q: "Why compute z in training?", a: "Demos are inconsistent. z absorbs the demo-to-demo variation so the decoder doesn’t average into a blur." },
    { q: "What’s ACT’s loss?", a: "<b>Reconstruction (L1) + β·KL</b>. L1 gives crisper actions than L2, and KL keeps z near N(0, 1) so z = 0 makes sense at inference." },
    { q: "What’s ACT’s main weakness?", a: "<b>Multimodality</b>. It averages distinct strategies (left and right) into one that can fail." },
  ],
});

/* 3 ─ Diffusion */
function ddimTwoMode(rand) {
  const K = 10, ab = k => Math.cos(k / K * Math.PI / 2 * .995) ** 2; let x = gauss(rand); const tr = [x];
  for (let k = K; k >= 1; k--) {
    const a = ab(k), a2 = ab(k - 1), x0 = Math.tanh(Math.sqrt(a) * x / (1 - a)), e = (x - Math.sqrt(a) * x0) / Math.sqrt(1 - a);
    x = Math.sqrt(a2) * x0 + Math.sqrt(1 - a2) * e; tr.push(x);
  } return tr;
}
lesson({
  id: 'a-diff', ch: 'arms', mins: 9,
  title: "Diffusion policy",
  lede: "Same demos, same goal, but the policy builds each action chunk by denoising random noise. Different noise gives a different valid trajectory, so nothing gets averaged.",
  build() {

    const two = Widget('Two valid ways around an obstacle', body => {
      let mode = 'diff', seed = 3, passes = 10; const scene = h('div'), traces = h('div'), out = h('div.readout');
      const draw = () => {
        const r = rng(seed * 101), N = 300, runs = Array.from({ length: N }, () => ddimTwoMode(r)), finals = runs.map(t => t[t.length - 1]);
        const act = Array.from({ length: N }, () => .03 * gauss(r)), xs = mode === 'act' ? act : finals;
        const sc = svgRoot(660, 190, { 'aria-label': 'Top view: an obstacle in the middle with paths curving around it' });
        sc.append(Circle(60, 95, 10, 'var(--ink)'), T(60, 123, 'start', 'axis', 'middle'), Circle(600, 95, 10, 'var(--hl)'), T(600, 123, 'object', 'axis', 'middle'), Rect(300, 63, 60, 64, 'var(--ink3)', { rx: 8, opacity: .8 }), T(330, 100, 'obstacle', 'lbl', 'middle', { style: 'fill:var(--bg)' }));
        for (let i = 0; i < 14; i++) { const x = xs[i], cy = 95 + x * 96; sc.append(Path(`M60 95 Q330 ${cy} 600 95`, Math.abs(x) < .5 ? 'var(--bad)' : 'var(--c)', 2.5, 'none', { opacity: .6 })); }
        scene.innerHTML = ''; scene.append(sc);
        const p = Plot(660, 150, { x: [-1.6, 1.6], y: [0, 1], xt: [-1, 0, 1], yt: [], xf: v => v === -1 ? 'left' : v === 1 ? 'right' : 'middle', xl: 'lateral approach of 300 attempts', m: { l: 20, r: 20, t: 8, b: 34 } });
        const bins = new Array(32).fill(0); xs.forEach(x => { const b = Math.floor((x + 1.6) / 3.2 * 32); if (b >= 0 && b < 32) bins[b]++; });
        const mxb = Math.max(...bins); bins.forEach((c, i) => p.svg.append(Rect(p.sx(-1.6 + i * .1) + 1, p.sy(c / mxb * .95), (p.sx(0) - p.sx(-.1)) - 2, p.sy(0) - p.sy(c / mxb * .95), 'var(--c)', { rx: 2 })));
        traces.innerHTML = ''; traces.append(p.svg);
        const crash = xs.filter(x => Math.abs(x) < .5).length;
        out.innerHTML = ''; out.append(Chip(mode === 'act' ? 'ACT: 1 forward pass' : 'Diffusion: 10 denoising passes', mode === 'act' ? 'the average' : 'a sample', 'c'), Chip('attempts that hit the obstacle', Math.round(crash / N * 100) + '%', crash / N > .2 ? 'bad' : 'ok'));
      };
      body.append(h('div.row', null, Seg([{ v: 'act', label: 'ACT (mean, z = 0)' }, { v: 'diff', label: 'Diffusion Policy (sample)' }], 'diff', v => { mode = v; draw(); }), Btn('New random draws', () => { seed++; draw(); }, 'ghost')), scene, traces, out,
        h('p.cap', null, '14 of the 300 attempts are drawn as paths (red goes through the obstacle). The histogram shows all 300. The demos are half left and half right. The denoiser here is exact for this two-mode task, so it shows what a well-trained diffusion policy does, not how it is trained.')); draw();
    });

    const chain = Widget('Denoising passes', body => {
      let passes = 0; const holder = h('div'); const runs = Array.from({ length: 10 }, (_, i) => ddimTwoMode(rng(40 + i * 17)));
      const draw = () => {
        const p = Plot(660, 220, { x: [0, 10], y: [-2, 2], xt: [0, 2, 4, 6, 8, 10], yt: [-1, 0, 1], yf: v => v === -1 ? 'left' : v === 1 ? 'right' : 'mid', xl: 'denoising pass', yl: 'lateral offset', m: { l: 64 } });
        runs.forEach((t, i) => { const pts = t.slice(0, passes + 1).map((v, k) => [p.sx(k), p.sy(clamp(v, -2, 2))]); if (pts.length > 1) p.svg.append(Path(path(pts), 'var(--c)', 2, 'none', { opacity: .7 })); const e = pts[pts.length - 1]; p.svg.append(Circle(e[0], e[1], 4.5, 'var(--c)', { stroke: 'var(--surface)', 'stroke-width': 1.5 })); });
        holder.innerHTML = ''; holder.append(p.svg);
      };
      body.append(holder, Slider({ label: 'passes so far', min: 0, max: 10, step: 1, value: 0, fmt: v => v + ' / 10', on: v => { passes = v; draw(); } }), h('p.cap', null, 'Ten different starting noises. Early on they look like noise. By pass ten each has settled on a clean left or right chunk. Each pass is one network evaluation (DDIM, a fast sampler), conditioned on the camera image.')); draw();
    });

    return [
      H2('Training'),
      P('The denoiser is a neural network, usually a U-Net. Diffusion is just a particular way of training and running it. Training never runs a denoising chain. For each example you take a clean demonstrated chunk, pick a random noise level, add that much noise, and train the network to predict the noise you added. Because the level is random, the network sees everything from lightly corrupted to pure static. It’s always one prediction step, since you already know the answer, which is the noise you put in.'),
      Eq('q_τ = √ᾱ_τ · q₀ + √(1 − ᾱ_τ) · ε          corrupt the clean chunk q₀ with noise ε\nloss = <span class="k">‖ ε − ε_θ(q_τ, τ, image) ‖²</span>        predict the noise, squared error'),
      H2('Inference'),
      P('On the robot there’s no clean chunk to start from, because producing one is the goal. So you start with pure random noise and run the denoiser about 10 times using DDIM, a fast sampler. Each pass removes some of the predicted noise, conditioned on the current camera image. After around 10 passes you have a clean action chunk.'),
      chain,
      H2('Why it doesn’t average'),
      P('Every run starts from fresh noise, and different noise ends up at different valid trajectories. The model learned the full distribution of good actions, not just the mean. If half the demos went left and half went right, one run gives a clean left chunk and the next gives a clean right one.'),
      two,
      Note('ACT gives you the average action in one forward pass. Diffusion gives you a sample from the whole distribution of good actions in about 10 passes, so handling multiple modes costs roughly 10 times the inference compute.'),
      Trap('ACT and diffusion policy are both imitation learning, and neither generalizes past its demos. If you only showed it grabbing a cube at position A, it won’t grab one at position C. The next lesson covers how people handle that.'),
      Check({ q: 'What is a diffusion policy network trained to predict?', opts: ['The next action', 'The noise that was added to a clean action chunk', 'The average of all the demonstrations'], ans: 1, why: 'Loss = ‖ε − ε_θ(q_τ, τ, image)‖². You know the noise because you added it.' }),
      Check({ q: 'What does diffusion cost compared with ACT?', opts: ['More demonstrations', 'About 10 times the inference compute (around 10 network evaluations per chunk)', 'It can’t use images'], ans: 1, why: 'ACT uses one forward pass per chunk.' }),
    ];
  },
  cards: [
    { q: "How is a diffusion policy trained?", a: "Take a clean demo chunk, pick a <b>random noise level</b>, corrupt it, and train the network to <b>predict the added noise</b>. It’s always one step, never the full chain." },
    { q: "How does a diffusion policy run at inference?", a: "Start from pure noise and denoise about <b>10 passes</b> (DDIM), conditioned on the camera image, until it’s a clean action chunk." },
    { q: "Why doesn’t diffusion policy average left and right demos?", a: "It learned the whole distribution. Different starting noise gives different valid trajectories, so it lands on one mode." },
    { q: "What’s ACT versus diffusion in one line?", a: "ACT does one pass and gives the average. Diffusion does about 10 passes and gives a sample. Roughly 10× the inference cost." },
    { q: "What limit do ACT and diffusion share?", a: "Both copy their demonstrations and don’t generalize past them." },
  ],
});

/* 4 ─ Hybrid */
lesson({
  id: 'a-hybrid', ch: 'arms', mins: 6,
  title: "Geometry plus learning",
  lede: "IK can reach anywhere but can’t grasp, and learned policies can grasp but only near what they were shown. Using both is what works in practice.",
  build() {
    const wid = Widget('Where the cube is', body => {
      let strat = 'hybrid', cube = [430, 120]; const W = 660, Hh = 300, BX = 330, BY = 290, R = 250, DC = [200, 150]; const svg = svgRoot(W, Hh, { 'aria-label': 'Table top view with arm base, demo region and cube' }), out = h('div.readout'), stats = h('div.readout');
      const inReach = p => Math.hypot(p[0] - BX, p[1] - BY) < R && p[1] < BY - 20, inDemo = p => Math.hypot(p[0] - DC[0], p[1] - DC[1]) < 62;
      const result = p => strat === 'learn' ? (inDemo(p) ? 'ok' : 'fail-gen') : strat === 'ik' ? (inReach(p) ? 'fail-grasp' : 'fail-reach') : (inReach(p) ? 'ok' : 'fail-reach');
      const draw = () => {
        svg.innerHTML = ''; svg.append(Rect(4, 4, W - 8, Hh - 8, 'var(--surface2)', { rx: 14 }), S('path', { d: `M${BX - R} ${BY} A${R} ${R} 0 0 1 ${BX + R} ${BY}`, fill: 'none', stroke: 'var(--line)', 'stroke-width': 2, 'stroke-dasharray': '5 5' }), T(BX, 24, 'table, top view (dashed arc = arm reach)', 'axis', 'middle'), Circle(DC[0], DC[1], 62, 'var(--c-soft)', { stroke: 'var(--c)', 'stroke-dasharray': '4 3', 'stroke-width': 2, opacity: .9 }), T(DC[0], DC[1] + 78, 'where you collected demos', 'axis', 'middle'), Rect(BX - 26, BY - 16, 52, 24, 'var(--ink)', { rx: 6 }), T(BX, BY - 22, 'arm', 'axis', 'middle'));
        const r = result(cube);
        if (strat !== 'learn' && inReach(cube)) svg.append(Line(BX, BY - 16, cube[0], cube[1], 'var(--c)', 3, { 'stroke-dasharray': strat === 'hybrid' ? '' : '2 6' }));
        svg.append(Rect(cube[0] - 12, cube[1] - 12, 24, 24, 'var(--hl)', { rx: 4, stroke: 'var(--ink)', 'stroke-width': 2 }), T(cube[0], cube[1] - 20, r === 'ok' ? '✓ grasped' : r === 'fail-gen' ? '✗ never saw this position' : r === 'fail-grasp' ? '✗ arrives, cannot grasp' : '✗ out of reach', 'lbl-b', 'middle', { style: 'fill:' + (r === 'ok' ? 'var(--ok)' : 'var(--bad)') }));
        let win = 0, n = 200; const rr = rng(77); for (let i = 0; i < n; i++) { const p = [rr() * 620 + 20, rr() * 260 + 20]; if (inReach(p) && result(p) === 'ok') win++; }
        const tot = Array.from({ length: 200 }, (_, i) => 0).length; let reachN = 0; const r2 = rng(77); for (let i = 0; i < n; i++) { const p = [r2() * 620 + 20, r2() * 260 + 20]; if (inReach(p)) reachN++; }
        out.innerHTML = ''; out.append(Chip('this placement', r === 'ok' ? 'success' : 'fails', r === 'ok' ? 'ok' : 'bad'));
        stats.innerHTML = ''; stats.append(Chip('success across the reachable table', Math.round(win / reachN * 100) + '%', win / reachN > .7 ? 'ok' : 'bad'));
      };
      drag(svg, p => { cube = [clamp(p.x, 20, W - 20), clamp(p.y, 40, Hh - 30)]; draw(); });
      body.append(Seg([{ v: 'learn', label: 'Learned policy only' }, { v: 'ik', label: 'IK only' }, { v: 'hybrid', label: 'FoundationPose + IK + learned grasp' }], 'hybrid', v => { strat = v; draw(); }), svg, out, stats, h('p.cap', null, '<b>A toy model, not a benchmark.</b> Click or drag on the table to move the cube. The learned-only policy works near the demonstrations. IK reaches everywhere but cannot grasp. The hybrid uses each where it is strong.')); draw();
    });

    return [
      P('Say you have 100 demos of picking up a cube, all from roughly the same spot on the table. A learned policy (ACT or diffusion) copies what it saw, so it fails once the cube is somewhere else. Try moving the cube in the widget below under each strategy.'),
      wid,
      H2('The pick-and-place pipeline'),
      OL('<b>FoundationPose</b> takes a camera image and a 3D model of the object and reports where the object is in space. It finds the target, not the arm.', '<b>IK</b> moves the arm to a consistent pre-grasp pose relative to that object. This is the getting-there step.', 'The <b>learned policy</b> only handles the final grasp, starting from that consistent pose.'),
      Note('The learned policy only needs to cover the small variations in the grasp itself, not the whole workspace.'),
      Check({ q: 'Why does the hybrid make the learned policy’s job easier?', opts: ['It starts from a consistent pose relative to the object every time, so it only learns the small variations in the grasp', 'It gets rid of the camera', 'It makes the network smaller'], ans: 0, why: 'FoundationPose and IK absorb the variation in where the object is.' }),
    ];
  },
  cards: [
    { q: "What’s the standard pick-and-place pipeline in practice?", a: "FoundationPose finds the object, IK moves to a consistent pre-grasp pose relative to it, and a learned policy does only the final grasp." },
    { q: "What does FoundationPose do?", a: "It takes a camera image and a 3D model of the object and outputs the object’s pose in space. It finds the target, not the arm." },
    { q: "Why is the hybrid better than a learned policy alone?", a: "The learned part only has to cover the small variations in the grasp, not the whole workspace." },
  ],
});

/* 5 ─ VLAs */
lesson({
  id: 'a-vla', ch: 'arms', mins: 10,
  title: "Vision-language-action models",
  lede: "ACT and diffusion policy only know the task you showed them. A VLA starts from a big pretrained vision-language model and gets fine-tuned to output robot actions.",
  build() {
    const quant = Widget('Approach A: actions as tokens', body => {
      let bins = 16; const holder = h('div'), out = h('div.readout');
      const f = t => .8 * Math.sin(2 * Math.PI * t) + .18 * Math.sin(6 * Math.PI * t + .6);
      const draw = () => {
        const p = Plot(660, 210, { x: [0, 1], y: [-1.1, 1.1], xt: [0, .25, .5, .75, 1], yt: [-1, 0, 1], xl: 'time within an action chunk', yl: 'shoulder target', m: { l: 44 } });
        const q = v => { const b = clamp(Math.floor((v + 1) / 2 * bins), 0, bins - 1); return -1 + (b + .5) * 2 / bins; };
        const N = 200; let mx = 0, se = 0; const cont = [], stair = [];
        for (let i = 0; i <= N; i++) { const t = i / N, v = f(t), qv = q(v); cont.push([p.sx(t), p.sy(v)]); stair.push([p.sx(t), p.sy(qv)]); mx = Math.max(mx, Math.abs(v - qv)); se += (v - qv) ** 2; }
        p.svg.append(Path(path(stair), 'var(--c)', 3), Path(path(cont), 'var(--hl)', 2, 'none', { 'stroke-dasharray': '5 4' }), T(p.m.l + 8, p.m.t + 14, '- - the smooth motion you want', 'lbl', 'start', { style: 'fill:var(--hl)' }), T(p.m.l + 8, p.m.t + 30, '━ what the token vocabulary can say', 'lbl', 'start', { style: 'fill:var(--c)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        out.innerHTML = ''; out.append(Chip('vocabulary size', bins + ' bins / dimension', 'c'), Chip('worst-case error', fx(mx, 3), mx > .1 ? 'bad' : 'ok'), Chip('RMS error', fx(Math.sqrt(se / (N + 1)), 3)));
      };
      const opts = [4, 8, 16, 32, 64, 128, 256];
      body.append(holder, Slider({ label: 'bins per dimension', min: 0, max: 6, step: 1, value: 2, fmt: v => opts[v], on: v => { bins = opts[v]; draw(); } }), out, h('p.cap', null, 'RT-2 split each action dimension into 256 buckets and treated each bucket as a new word. The language model is unchanged, but the motion is quantized, which is rough for fast, precise control.')); draw();
    });

    const arch = Widget('Two ways to get from a VLM to motor commands', body => {
      let m = 'A'; const box = (t, d, strong) => h('div', { style: { border: '1.5px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'var(--surface)', borderRadius: '10px', padding: '9px 12px' } }, h('b', { style: { fontSize: '14.5px' } }, t), h('div', { style: { fontSize: '13px', color: 'var(--ink3)' } }, d));
      const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } }), eq = h('div.eq');
      const draw = () => {
        st.innerHTML = '';
        st.append(box('Inputs', '[image tokens] + [text: “pick up the cube”]'), box('Pretrained VLM backbone', 'billions of parameters trained on internet-scale data, so it knows what a cup is and what “the red one” means', true), h('div.mono', { style: { textAlign: 'center' } }, '↓'));
        if (m === 'A') { st.append(box('Predicts action tokens', 'each action dimension is binned; each bin is a new word in the vocabulary'), h('div.mono', { style: { textAlign: 'center' } }, '↓ de-tokenize'), box('Continuous joint commands', 'loses detail, awkward for fast precise motion')); eq.innerHTML = 'fine-tuning loss: <span class="k">token prediction</span>, same as language modeling'; }
        else { st.append(box('Action expert (small separate network)', 'turns the VLM’s scene and task representation into smooth motion using flow matching, a close relative of diffusion', true), h('div.mono', { style: { textAlign: 'center' } }, '↓'), box('Continuous actions, high frequency', 'no discretization')); eq.innerHTML = 'fine-tuning loss: <span class="k">flow matching</span> on the continuous action (moving from noise toward the real action)'; }
      };
      body.append(Seg([{ v: 'A', label: 'A · Tokenize actions (RT-2)' }, { v: 'B', label: 'B · Action expert (π0)' }], 'A', v => { m = v; draw(); }), st, eq); draw();
    });


    return [
      P('A VLA starts from a pretrained vision-language model, the kind behind image-understanding chatbots, which has been trained on huge amounts of internet images and text. It already knows what a cup is, what “pick up the red one” means and how objects sit relative to each other. That’s something like 3 billion parameters of world knowledge for free. You attach a way to produce actions and fine-tune the whole thing on robot demos. This is why VLAs handle new objects and new instructions where ACT can’t.'),
      P('A VLM outputs text tokens and the robot needs joint commands, so there are two ways to bridge that.'),
      H3('A. Tokenize the actions'),
      P('This is the original RT-2 approach. Each action dimension is split into bins (256, for example) and every bin becomes a new word in the vocabulary. Actions turn into more tokens, predicted the same way as the next word in a sentence, and the model barely changes. The downside is that chopping smooth motion into buckets loses information and is awkward for fast, precise control.'),
      quant,
      H3('B. A separate action expert'),
      P('This is what π0 does. The pretrained VLM still does the understanding, turning the image and instruction into a representation of the scene and the task. A separate small network, the action expert, takes that representation and outputs continuous actions using flow matching, a close relative of diffusion. Nothing is discretized, so control stays smooth and can run at high frequency.'),
      arch,
      P('Fine-tuning feeds the image and instruction through the model and trains its output to match the demonstrated action. Approach A uses a token-prediction loss and approach B uses a flow-matching loss on the continuous action. You can fine-tune with LoRA (freeze the big backbone and train small adapters, which is cheap) or go with fuller fine-tuning. It’s the same idea as fine-tuning an LLM, except you’re adding a new kind of output.'),
      H2('The three big players'),
      UL('<b>NVIDIA GR00T</b> splits the model into two systems. System 2 is a slow VLM that reasons about the task, and System 1 is a fast diffusion action expert with about 10 ms latency. It’s the same slow-planner, fast-controller split you see across the whole stack, just inside the manipulation policy. They’re betting on data scale, with 20,000+ hours of human egocentric video plus sim.', '<b>Google Gemini Robotics</b> reasons in language first, then translates that plan into actions. It merges data from different robot types using motion transfer.', '<b>Physical Intelligence (the π series)</b> uses a single flow-based stream that generates actions through diffusion or flow matching. They introduced approach B for VLAs.'),
      P('They differ in how cleanly they separate thinking from acting (NVIDIA the most, PI the least), whether they reason in language first (Google does), and how they get around the cost of robot demos (all three are moving toward video, sim and transfer between robots). Underneath, each one is a pretrained understanding model plus an action generator, fine-tuned on robot data.'),
      Check({ q: 'Where does a VLA’s ability to follow “pick up the red one” for an object it never saw come from?', opts: ['Your robot demonstrations', 'The pretrained vision-language model', 'IK'], ans: 1, why: 'The backbone already knows about objects and language.' }),
      Check({ q: 'What’s the main downside of tokenizing actions?', opts: ['It needs a new model', 'Splitting continuous motion into bins loses information and is rough for fast, precise control', 'It can’t use language'], ans: 1, why: 'Fewer bins means bigger steps. Approach B keeps the output continuous.' }),
    ];
  },
  cards: [
    { q: "What is a VLA, and why does it generalize better than ACT?", a: "A pretrained vision-language model plus an action output, fine-tuned on robot demos. It brings internet-scale knowledge of objects and language that your own demos couldn’t provide." },
    { q: "What’s the difference between approach A and B for getting actions out of a VLM?", a: "<b>A</b> (RT-2) bins each action dimension (e.g. 256) and predicts tokens, with a token-prediction loss, and loses information. <b>B</b> (π0) uses a separate action expert with flow matching, so the output is continuous and smooth." },
    { q: "How do NVIDIA, Google and Physical Intelligence differ?", a: "<b>GR00T</b> has System 2 (slow VLM) plus System 1 (fast diffusion, ~10 ms). <b>Google</b> reasons in language first, then acts. <b>PI</b> uses one flow-based stream and is the most unified." },
    { q: "What is LoRA?", a: "Freezing the large backbone and training small adapters. It’s cheap fine-tuning that keeps the general knowledge." },
  ],
});

/* 6 ─ Frontier */
lesson({
  id: 'a-frontier', ch: 'arms', mins: 8,
  title: "Learning from experience",
  lede: "Imitation can only be as good as its demos. The newest fix adds reinforcement learning on top of the VLA, and it reuses advantage from PPO.",
  build() {
    const rungs = [
      { n: 'IK', adds: 'Geometry. It reaches any pose, anywhere, instantly.', fails: 'No dexterity. It can’t grasp, close on an object or handle slip.', needs: 'Encoders always, a camera for the target, and an IMU only if the base floats.' },
      { n: 'ACT', adds: 'Dexterity from demos, with 100-action chunks against compounding error.', fails: 'Averages across demonstrations (z = 0), so it breaks when a task has clearly different modes. Copies its demos.', needs: '50 to 200 teleoperated demos.' },
      { n: 'Diffusion Policy', adds: 'Samples from the full distribution of good actions, so it handles multiple modes.', fails: 'About 10× the inference compute, and it still doesn’t generalize past its demos.', needs: 'The same demos, plus ~10 denoising passes.' },
      { n: 'VLA', adds: 'World knowledge from a pretrained VLM, so it handles new objects and language instructions.', fails: 'Still imitation, so it’s limited by its demos and can’t learn from failure. It also has no memory and only reacts to the current frame.', needs: 'A large pretrained backbone plus robot demos to fine-tune on (LoRA or full).' },
      { n: 'VLA + RL from experience', adds: 'Goes past its demonstrations and recovers from mistakes by learning from its own successes and failures (π*0.6, RECAP).', fails: 'Needs real deployment, human interventions, and a way to use RL with flow-based actions.', needs: 'Demos, on-policy collection and human interventions.' },
    ];
    const ladder = Widget('The ladder', body => {
      let cur = 0; const list = h('div', { style: { display: 'flex', flexDirection: 'column-reverse', gap: '6px' } }), panel = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } });
      const draw = () => {
        list.innerHTML = ''; rungs.forEach((r, i) => list.append(h('button', { type: 'button', style: { textAlign: 'left', border: '1.5px solid ' + (i === cur ? 'var(--c)' : 'var(--line)'), background: i === cur ? 'var(--c-soft)' : 'var(--surface)', borderRadius: '10px', padding: '9px 14px', fontWeight: 600, marginLeft: (i * 14) + 'px' }, on: { click() { cur = i; draw(); } } }, 'Rung ' + (i + 1) + ' · ' + r.n)));
        const r = rungs[cur]; panel.innerHTML = ''; panel.append(h('p', { style: { margin: 0 }, html: '<b style="color:var(--ok)">Adds:</b> ' + r.adds }), h('p', { style: { margin: 0 }, html: '<b style="color:var(--bad)">Still breaks:</b> ' + r.fails }), h('p', { style: { margin: 0 }, html: '<b>Needs:</b> ' + r.needs }));
      };
      body.append(h('div.grid2', null, list, panel)); draw();
    });

    const loop = Widget('Locomotion and manipulation compared', body => {
      const rows = [['Rollout of experience', 'The robot deploys in the real world and collects its own successes and failures'], ['Advantage: better or worse than average?', 'Still the signal. The policy improves using advantage, learned from real deployment'], ['PPO clipped update', 'Flow-matching action generation doesn’t fit standard PPO math cleanly, so they use <b>advantage-conditioning</b>'], ['Exploration vs exploitation', 'Human interventions when it gets stuck, plus on-policy attempts'], ['Reward from the environment', 'Success and failure outcomes, alongside demonstrations']];
      body.append(h('div.scroll-x', null, h('table.tt', { style: { whiteSpace: 'normal' } }, h('thead', null, h('tr', null, h('th', { style: { textAlign: 'left' } }, 'Locomotion (PPO)'), h('th', { style: { textAlign: 'left' } }, 'Manipulation frontier (RECAP)'))), h('tbody', null, rows.map(r => h('tr', null, h('td', { style: { textAlign: 'left', whiteSpace: 'normal', verticalAlign: 'top' } }, r[0]), h('td', { style: { textAlign: 'left', whiteSpace: 'normal' }, html: r[1] })))))));
    });

    return [
      P('ACT, diffusion policy and a fine-tuned VLA all copy demonstrations. They can’t do better than the demos, and they can’t learn from failures, because failures aren’t in the demo set.'),
      P('Physical Intelligence’s π*0.6 (late 2025) puts reinforcement learning on top of an imitation-trained VLA. The robot gets deployed, collects its own successes and failures, and improves from them using a method called RECAP. RECAP combines demonstrations, the robot’s own on-policy data and human interventions for when it gets stuck. What it uses to improve the policy is advantage, the same “better or worse than average” number from PPO.'),
      P('There’s one wrinkle. Flow-matching VLAs can’t use standard PPO directly, because flow-based action generation doesn’t fit the usual RL math cleanly, so they use advantage-conditioning instead. The idea is the same. Push the policy toward good actions and away from bad ones, learned from real deployment as well as from demos.'),
      loop,
      Note('Manipulation is heading toward the same recipe locomotion already uses: pretrain on demonstrations, then improve with RL from real experience.'),
      H2('The ladder'),
      ladder,
      Match({ q: 'Which method fits each situation?', rows: [{ l: 'Move the hand to wherever a camera says the object is, with no learning', r: 'IK' }, { l: '50 demos of one grasp style, one fast forward pass', r: 'ACT' }, { l: 'Demos go left and right around an obstacle and must not be averaged', r: 'Diffusion Policy' }, { l: 'Follow “pick up the red one” for objects that weren’t in your demos', r: 'VLA' }, { l: 'Get better than the demos using its own successes and failures', r: 'VLA + RL from experience' }], why: 'Each rung adds something the one below it doesn’t have.' }),
      H3('Other frontier topics'),
      UL('<b>Memory.</b> Standard VLAs have none and only react to the current frame. Newer work adds memory for long tasks that need earlier steps remembered.', '<b>Force and tactile sensing.</b> Vision alone misses contact forces, which matter a lot for delicate manipulation.', '<b>Mixture of experts.</b> Route different sub-tasks to specialized sub-networks so capacity can grow efficiently.'),
      Check({ q: 'What limits ACT, diffusion policy and a fine-tuned VLA?', opts: ['Backbone size', 'They’re imitation learners, so they can’t beat their demos or learn from failure', 'Camera resolution'], ans: 1, why: 'Getting past that needs learning from the robot’s own experience.' }),
      Check({ q: 'Why can’t a flow-matching VLA use standard PPO directly?', opts: ['PPO needs images', 'Flow-based action generation doesn’t fit the standard RL math cleanly, so advantage-conditioning is used', 'PPO only works for legs'], ans: 1, why: 'Advantage survives as an idea even though PPO’s ratio and clip don’t.' }),
    ];
  },
  cards: [
    { q: "What limits ACT, diffusion and a fine-tuned VLA?", a: "They’re imitation. They can’t exceed their demos and can’t learn from failure." },
    { q: "What are π*0.6 and RECAP?", a: "Physical Intelligence’s method (late 2025) for improving an imitation-trained VLA with RL from real deployment. It uses demos, on-policy experience and human interventions, driven by advantage." },
    { q: "Why is advantage-conditioning used for flow-based VLAs?", a: "Flow-matching action generation doesn’t fit standard PPO math cleanly. It’s the same idea with a different mechanism." },
    { q: "What’s each rung of the ladder in one line?", a: "IK is geometry with no dexterity. ACT is imitation and averages. Diffusion samples modes at ~10× the compute. A VLA adds world knowledge. VLA + RL can go past the demos." },
    { q: "What are three other frontier topics?", a: "Memory (VLAs have none), force and tactile sensing, and mixture of experts." },
  ],
});
