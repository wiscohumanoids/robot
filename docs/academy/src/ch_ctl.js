'use strict';
/* ═════════ Chapter — Controls ═════════ */

/* ── small simulations for the widgets ── */
function simTrack(kp, kd, hz) { // one leg joint following a policy that updates its target at `hz`, PD running at 1 kHz
  const I = .1, c = .02, dt = .001, N = 1000; let q = 0, w = 0; const qs = [], tg = [], rf = [];
  for (let i = 0; i <= N; i++) {
    const t = i * dt, r = .4 * Math.sin(2 * Math.PI * 1.5 * t), tk = Math.floor(t * hz + 1e-9) / hz, tgt = .4 * Math.sin(2 * Math.PI * 1.5 * tk);
    const tau = kp * (tgt - q) + kd * (0 - w);
    qs.push(q); tg.push(tgt); rf.push(r); w += (tau - c * w) / I * dt; q += w * dt;
  } return { qs, tg, rf };
}
function simArm(mode, kp, kd, err) { // one arm joint, horizontal at q = 0, gravity torque = mgl·cos(q)
  const I = .4, c = .3, mgl = 6, dt = .002, N = 1000; let q = mode === 'teach' ? .6 : 0, w = 0; const qs = [], ts = [];
  for (let i = 0; i <= N; i++) {
    const t = i * dt, qd = t >= .1 ? .6 : 0, gh = mode === 'pd' ? 0 : (1 + err) * mgl * Math.cos(q);
    const tau = mode === 'teach' ? gh : gh + kp * (qd - q) + kd * (0 - w);
    qs.push(q); ts.push(tau); w += (tau - mgl * Math.cos(q) - c * w) / I * dt; q = clamp(q + w * dt, -1.5, 1.5);
    if (q === -1.5 || q === 1.5) w = 0;
  } return { qs, ts };
}

/* 1 ─ the layer itself */
lesson({
  id: 'c-layer', ch: 'ctl', mins: 8,
  title: 'The control layer',
  lede: 'Between the policies and ros2_control there’s a layer that turns target positions into torques. ros2_control, EtherCAT and the drives move data around, and this layer decides what torque to send.',
  build() {
    const route = Widget('Both policies, both controllers', body => {
      let sel = 'both'; const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '10px' } }), info = h('p.cap');
      const box = (t, d, on) => h('div', { style: { flex: '1 1 150px', minWidth: 0, border: '1px solid ' + (on ? 'var(--c)' : 'var(--line)'), background: on ? 'var(--c-soft)' : 'transparent', opacity: on ? 1 : .35, borderRadius: '8px', padding: '8px 12px', transition: 'opacity .2s' } }, h('b', { style: { fontSize: '14.5px' } }, t), h('div', { style: { fontSize: '12.5px', color: 'var(--ink2)' } }, d));
      const arr = () => h('span.mono', { style: { flex: 'none', color: 'var(--ink3)', alignSelf: 'center' } }, '→');
      const row = (a, b, c2, on) => h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'stretch' } }, box(a[0], a[1], on), arr(), box(b[0], b[1], on), arr(), box(c2[0], c2[1], on));
      const draw = () => {
        const armOn = sel !== 'legs', legOn = sel !== 'arm';
        st.innerHTML = '';
        st.append(row(['Manipulation policy', 'camera and joint state in, arm position targets out'], ['Arm controller', 'impedance control: gravity + Coriolis + PD, 1–2 kHz'], ['τ arm', 'a torque for every arm joint'], armOn),
          row(['Locomotion policy', '/cmd_vel and observation in, leg position targets out, 50–100 Hz'], ['Leg controller', 'plain PD, 1–2 kHz'], ['τ legs', 'a torque for every leg joint'], legOn),
          h('div', { style: { border: '1px solid var(--line)', borderRadius: '8px', padding: '8px 12px', fontSize: '14px' } }, h('b', null, 'ros2_control'), ' effort interface → EtherCAT / CAN → drives and motors'));
        info.textContent = sel === 'both' ? 'Both controllers run all the time, each on its own joints. They share the same ros2_control effort interface and the same drives.'
          : sel === 'arm' ? 'The arm sticks out into free space, so its controller needs the arm’s dynamics: gravity and Coriolis terms from a dynamics library, plus PD and a joint-limit barrier.'
            : 'The leg policy was trained in a simulator that already had gravity and contact in it, so the controller underneath is just PD, tracking whatever positions the policy asks for.';
      };
      body.append(Seg([{ v: 'both', label: 'Both' }, { v: 'arm', label: 'Arm' }, { v: 'legs', label: 'Legs' }], 'both', v => { sel = v; draw(); }), st, info); draw();
    });
    return [
      H2('Where it sits'),
      P('The locomotion policy and the manipulation policy both output joint position targets. Nothing below them wants position targets, because a motor drive takes torque. The control layer sits in between. It takes the targets and the current joint state and computes a torque for every joint, quickly and smoothly and without hurting anything.'),
      P('ros2_control, the EtherCAT driver and the drives are the transport. Libraries handle most of that, and the hardware chapter covers it. The control layer is the math that runs on top of that plumbing and makes the joints move where the policy asked. It’s its own piece of software, and the robot doesn’t work without it.'),
      route,
      H2('Two policies, two controllers'),
      P('The arm and the legs get different controllers, and the reason is the main point of this chapter. The arm sticks out into free space with gravity pulling it down and nothing to push against, so its controller has to know the arm’s physics. The legs are touching the ground through the whole gait, and the locomotion policy was trained in a simulator that already had gravity and contact forces, so a plain fast PD loop underneath is enough.'),
      UL('<b>Arm:</b> impedance control with gravity and Coriolis compensation. The arm lessons in this chapter cover it.', '<b>Legs:</b> a fast PD controller. That’s the next lesson.'),
      H2('Two rates'),
      P('Both controllers run much faster than the policies feeding them. A policy puts out a new target every 10 to 20 ms (50 to 100 Hz). The controllers run at 1 to 2 kHz, which is 10 to 40 times faster, and keep recomputing torque from the latest target and the current joint state. It’s the same slow-brain, fast-reflex pattern as the rest of the stack.'),
      H2('Joint-level control now, a whole-body controller later'),
      P('What this layer does is joint-level torque control, and it’s the ancestor of a whole-body controller (WBC). A WBC solves one optimization across the whole body at once, coordinating balance, several contacts and competing priorities under the full coupled dynamics. That’s more than a first version needs. Both controllers are always running, each on its own joints. What the first version doesn’t attempt is coordinating them, like reaching out while balancing on one foot. A real WBC can take over the same slot later, and the last lesson in this chapter covers when.'),
      Check({ q: 'What does the control layer take in and put out?', opts: ['Position targets from the policies and the current joint state in, a torque per joint out', 'Torques from the policies in, currents out', 'Camera images in, position targets out'], ans: 0, why: 'The drive takes torque, so something has to convert the policies’ position targets into torques.' }),
      Check({ q: 'Why do the arm and the legs get different controllers?', opts: ['The arm is in free space and needs a controller that knows its dynamics, while the leg policy already learned the dynamics in simulation', 'The arm motors are faster', 'The legs use a different network'], ans: 0, why: 'The leg policy was trained in a simulator with gravity and contact, so a plain PD is enough underneath it.' }),
      Check({ q: 'How much faster do the controllers run than the policies?', opts: ['About 10 to 40 times', 'About the same rate', 'About 1000 times'], ans: 0, why: 'Policies run at 50 to 100 Hz and the controllers at 1 to 2 kHz.' }),
    ];
  },
  cards: [
    { q: 'What does the control layer do and where does it sit?', a: 'It turns the policies’ joint position targets into torques. It sits below both policies and above ros2_control.' },
    { q: 'Which controller does each side use?', a: '<b>Arm</b>: impedance control with gravity and Coriolis compensation. <b>Legs</b>: a plain fast PD controller.' },
    { q: 'What’s the rate structure of the control layer?', a: 'Policies output a target every 10-20 ms (50-100 Hz). The controllers run at 1-2 kHz, 10 to 40 times faster, using the latest target and the current joint state.' },
    { q: 'How does this layer relate to a whole-body controller?', a: 'It’s joint-level torque control, the ancestor of a WBC. A WBC would take over the same slot later and replace both controllers.' },
  ],
});

/* 2 ─ legs */
lesson({
  id: 'c-legs', ch: 'ctl', mins: 9,
  title: 'Legs: a plain PD controller',
  lede: 'The locomotion papers the notes mention all use the same low-level controller, a plain PD loop with no dynamics model at all.',
  build() {
    const legsW = Widget('Slow targets, fast torque', body => {
      let hz = 50, kp = 100; const holder = h('div'), out = h('div.readout');
      const rms = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length);
      const draw = () => {
        const tr = simTrack(100, 2.5, hz), cu = simTrack(kp, 2.5, hz), p = Plot(660, 230, { x: [0, 1], y: [-.6, 1.05], xt: [0, .25, .5, .75, 1], yt: [-.4, 0, .4], xl: 'time (s)', yl: 'joint angle (rad)', m: { l: 48 } });
        const ln = (a, col, w, dash) => Path(path(a.filter((_, i) => i % 4 === 0).map((v, i) => [p.sx(i * .004), p.sy(clamp(v, -.6, 1.05))])), col, w, 'none', dash ? { 'stroke-dasharray': dash } : {});
        p.svg.append(ln(cu.rf, 'var(--line)', 1.5), ln(cu.tg, 'var(--hl)', 1.5, '3 3'), ln(tr.qs, 'var(--ink3)', 2, '6 4'), ln(cu.qs, 'var(--c)', 3),
          T(p.m.l + 8, p.m.t + 14, 'yellow steps: policy target, updated ' + hz + ' times a second', 'lbl', 'start', { style: 'fill:var(--hl)' }), T(p.m.l + 8, p.m.t + 30, 'grey dashed: joint with the trained gains (Kp 100)', 'lbl', 'start', { style: 'fill:var(--ink3)' }), T(p.m.l + 8, p.m.t + 46, 'colored: joint with the deployed gains (Kp ' + kp + ')', 'lbl', 'start', { style: 'fill:var(--c)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        const et = rms(tr.qs, tr.rf), ec = rms(cu.qs, cu.rf);
        out.innerHTML = ''; out.append(Chip('PD loop', '1000 updates / s'), Chip('policy', hz + ' updates / s'), Chip('error, trained gains', fx(et, 3) + ' rad'), Chip('error, deployed gains', fx(ec, 3) + ' rad', ec > et * 1.4 ? 'bad' : 'ok'));
      };
      body.append(holder, Slider({ label: 'policy rate', min: 10, max: 100, step: 5, value: hz, fmt: v => v + ' Hz', on: v => { hz = v; draw(); } }), Slider({ label: 'deployed Kp', min: 20, max: 300, step: 5, value: kp, fmt: v => v, on: v => { kp = v; draw(); } }), out,
        h('p.cap', null, 'One leg joint follows a target the policy updates at the rate above, while the PD loop underneath recomputes torque every millisecond. Kd stays at 2.5 and the trained gains are Kp 100. Move the deployed Kp away from 100 and the joint stops responding the way the policy expected.')); draw();
    });
    return [
      H2('The whole controller'),
      Eq('τ = <span class="k">Kp</span>·(q_target − q) + <span class="k">Kd</span>·(0 − q̇)'),
      P('That’s all of it. There’s no gravity compensation, no Coriolis term and no dynamics library. The 0 in the damping term is the desired velocity, since the policy only asks for positions. The team’s lowlevel_control package already computes the general form τ = τ_ff + Kp(q_d − q) + Kd(q̇_d − q̇). For locomotion you set τ_ff = 0 and leave the desired velocity at 0, and you have the standard locomotion PD law.'),
      H2('Why this is enough'),
      P('The policy was trained inside a physics simulator that already had gravity, inertia and contact forces. It learned to output joint targets that account for all of that. So the gravity compensation lives in the network’s weights, and the PD controller only has to follow whatever positions the policy commands. The legs are also in contact with the ground through the whole gait, so the dynamics are dominated by contact forces the policy learned to use, and not by the free-space sag an arm gets.'),
      P('An arm needs an explicit physics model because a scripted or IK motion knows nothing about the arm’s dynamics. The leg policy is that physics model, learned.'),
      H2('Two rates'),
      P('The policy runs at 50 to 100 Hz and the PD loop runs underneath it at 1000 to 2000 Hz, 10 to 40 times faster. The policy drops a new target every 10 to 20 ms. Between those updates the inner loop keeps recomputing torque from the latest target and the current joint state, which is what keeps the joints tracking smoothly.'),
      legsW,
      H2('Gains have to match training'),
      P('During training the policy assumed particular Kp and Kd values. It commands positions expecting the joints to respond the way those gains make them respond. Use different gains on the real robot and the learned behavior doesn’t transfer. Treat the gains like the observation format and copy them from the training config.'),
      h('div.scroll-x', null, h('table.tt', null, h('thead', null, h('tr', null, ['joint group', 'Kp', 'Kd'].map(x => h('th', null, x)))), h('tbody', null, [['hips', '100', '2.5'], ['knees', '200', '5'], ['ankles', '20', '0.1'], ['waist', '400', '5']].map(r => h('tr', null, r.map(c => h('td', null, c))))))),
      P('Those are typical starting values from the papers, and they’re close to what the repo’s defaults already are. The pattern is the same as the arm’s: stiff on the big joints and soft on the distal ones.'),
      H2('What’s left to do'),
      P('The controller itself is nearly finished in the scaffold. The remaining work is matching the gains to the trained policy and running the loop at 1 kHz.'),
      Trap('Kp and Kd in the deployed controller have to be the same values used in training. If the policy was trained with one set and the hardware runs another, the policy is commanding positions for a robot that doesn’t exist.'),
      Check({ q: 'Why does a locomotion policy not need gravity compensation in its controller?', opts: ['It was trained in a simulator with gravity and contact, so the compensation is already in its weights', 'Legs don’t have weight', 'The motors compensate automatically'], ans: 0, why: 'The PD controller only has to track the positions the policy commands.' }),
      Check({ q: 'Roughly how do the two loops’ rates compare?', opts: ['Policy 50-100 Hz, PD 1000-2000 Hz', 'Both at 1 kHz', 'Policy 1 kHz, PD 50 Hz'], ans: 0, why: 'The PD loop runs 10 to 40 times faster and fills in between the policy’s updates.' }),
      Check({ q: 'What happens if the deployed PD gains differ from the training gains?', opts: ['The joints respond differently than the policy expects and the behavior doesn’t transfer', 'Nothing, the policy adapts', 'The controller refuses to start'], ans: 0, why: 'The policy learned to command positions assuming a specific response.' }),
    ];
  },
  cards: [
    { q: 'What’s the leg control law?', a: '<b>τ = Kp·(q_target - q) + Kd·(0 - q̇)</b>. No gravity compensation, no Coriolis, no dynamics library. In lowlevel_control it’s the general form with τ_ff = 0.' },
    { q: 'Why is a plain PD enough for the legs?', a: 'The policy was trained in a simulator with gravity, inertia and contact, so it already outputs targets that account for them. The compensation is in the weights. The legs are also in contact with the ground through the gait.' },
    { q: 'What are the two rates for the legs?', a: 'Policy: 50-100 Hz, a new target every 10-20 ms. PD loop: 1000-2000 Hz, 10 to 40 times faster.' },
    { q: 'Why must deployed gains match training gains?', a: 'The policy learned to command positions assuming a specific joint response. Different gains mean different behavior.' },
    { q: 'Typical starting leg gains?', a: 'Hips Kp 100 / Kd 2.5. Knees 200 / 5. Ankles 20 / 0.1. Waist 400 / 5. Stiff on big joints, soft on distal ones.' },
  ],
});

/* 3 ─ arm */
lesson({
  id: 'c-arm', ch: 'ctl', mins: 10,
  title: 'Arm: impedance control',
  lede: 'An arm sticks out into free space, so a plain spring toward the target sags under the arm’s own weight. The arm controller adds terms that come from the arm’s physics.',
  build() {
    const armW = Widget('Gravity compensation on one joint', body => {
      let mode = 'pd', kp = 60, kd = 8, err = 0; const holder = h('div'), out = h('div.readout');
      const draw = () => {
        const { qs, ts } = simArm(mode, kp, kd, err), n = qs.length, p = Plot(660, 230, { x: [0, 2], y: [-.5, 1], xt: [0, .5, 1, 1.5, 2], yt: [-.4, 0, .4, .8], xl: 'time (s)', yl: 'joint angle (rad)', m: { l: 48 } });
        const tgt = mode === 'teach' ? [[0, .6], [2, .6]] : [[0, 0], [.1, 0], [.1, .6], [2, .6]];
        p.svg.append(Path(path(tgt.map(a => [p.sx(a[0]), p.sy(a[1])])), 'var(--hl)', 2, 'none', { 'stroke-dasharray': '5 4' }), Path(path(qs.filter((_, i) => i % 4 === 0).map((v, i) => [p.sx(i * 4 * .002), p.sy(clamp(v, -.5, 1))])), 'var(--c)', 3),
          T(p.m.l + 8, p.m.t + 14, mode === 'teach' ? 'yellow: where you left it' : 'yellow: target', 'lbl', 'start', { style: 'fill:var(--hl)' }), T(p.m.l + 8, p.m.t + 30, 'colored: joint angle (0 = arm horizontal)', 'lbl', 'start', { style: 'fill:var(--c)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        const end = qs[n - 1], dev = end - .6;
        out.innerHTML = ''; out.append(mode === 'teach' ? Chip('after 2 s', Math.abs(dev) < .02 ? 'holds in place' : (dev < 0 ? 'sinks ' : 'floats up ') + fx(Math.abs(dev), 2) + ' rad', Math.abs(dev) < .02 ? 'ok' : 'bad') : Chip('final error', fx(dev, 3) + ' rad', Math.abs(dev) > .03 ? 'bad' : 'ok'),
          Chip('gravity model', mode === 'pd' ? 'none' : err === 0 ? 'exact' : (err > 0 ? '+' : '') + Math.round(err * 100) + '% mass error', mode !== 'pd' && err !== 0 ? 'hl' : ''), mode === 'teach' ? Chip('Kp, Kd', '0, 0') : Chip('Kp, Kd', kp + ', ' + fx(kd, 1)));
      };
      body.append(holder, Seg([{ v: 'pd', label: 'PD only' }, { v: 'gc', label: 'Gravity compensation + PD' }, { v: 'teach', label: 'Teach mode' }], 'pd', v => { mode = v; draw(); }),
        Slider({ label: 'stiffness Kp', min: 0, max: 200, step: 1, value: kp, fmt: v => v, on: v => { kp = v; draw(); } }), Slider({ label: 'damping Kd', min: 0, max: 20, step: .5, value: kd, fmt: v => fx(v, 1), on: v => { kd = v; draw(); } }), Slider({ label: 'URDF mass error', min: -.5, max: .5, step: .05, value: err, fmt: v => (v > 0 ? '+' : '') + Math.round(v * 100) + '%', on: v => { err = v; draw(); } }), out,
        h('p.cap', null, 'PD only: the joint settles below the target because the spring has to carry the arm’s weight. Gravity compensation + PD: the arm’s weight is cancelled and the spring only does the moving. Teach mode: Kp = Kd = 0 and only the gravity term is left. With a correct model the arm holds wherever you leave it. With wrong inertials it sinks or floats. The model error slider changes the mass in the URDF that the controller believes.')); draw();
    });
    return [
      H2('Why plain PD isn’t enough'),
      P('Gravity pulls the arm down and there’s nothing to push against. Command “go to 45 degrees” with only a spring and the arm sags below 45, and the controller has to fight gravity constantly, which feels stiff and twitchy. The arm needs a richer control law, a sum of terms that each do one job.'),
      Eq([['τ = ', 'g(q)', 'gravity compensation'], ['  + ', 'C(q, q̇)·q̇', 'Coriolis'], ['  + ', 'Kp·(q_des − q)', 'position spring'], ['  + ', 'Kd·(q̇_des − q̇)', 'velocity damper'], ['  + ', 'barrier(q)', 'soft push away from joint limits']].map(([pre, term, note]) => pre + '<span class="k">' + term + '</span>' + ' '.repeat(Math.max(2, 26 - (pre + term).replace(/\u0307/g, '').length)) + note).join('\n')),
      UL('<b>g(q)</b> is the torque needed just to hold the arm up at this pose.', '<b>C(q, q̇)·q̇</b> covers the velocity-dependent forces when the arm moves fast.', '<b>Kp·(q_des − q)</b> is the spring pulling toward the target.', '<b>Kd·(q̇_des − q̇)</b> is the damper that prevents overshoot.', '<b>barrier(q)</b> is a soft repulsive push near the joint limits, for safety.'),
      P('Where each term comes from matters. Gravity and Coriolis are pure physics, so you don’t tune them. You compute them, exactly, from the arm’s dynamics model. Kp and Kd can’t be computed. You tune them by hand on the real hardware, watching for oscillation on one side and sluggishness on the other.'),
      armW,
      Trap('The gravity term is only as good as the arm’s model. If the mass or center of mass in the URDF is off, the controller cancels the wrong amount of gravity and no choice of Kp and Kd fully fixes it. The next lesson covers where the model comes from.'),
      Note('Like the leg controller, this one takes joint position targets from a policy (or from IK) and runs at 1 to 2 kHz. It sits in the same slot in the stack. What differs is that it adds the arm’s own physics, because a free-space arm has no learned policy absorbing gravity for it.'),
      Check({ q: 'Why does a plain spring sag an arm?', opts: ['The spring has to carry the arm’s weight, so a nonzero error remains', 'The damper is too weak', 'The policy is too slow'], ans: 0, why: 'At equilibrium Kp times the error equals the gravity torque, so the arm rests below the target.' }),
      Check({ q: 'Which terms are computed and which are tuned?', opts: ['Gravity and Coriolis are computed from the dynamics model, and Kp and Kd are tuned by hand', 'All four are tuned by hand', 'All four are computed'], ans: 0, why: 'Gravity and Coriolis are pure physics. Kp and Kd depend on the hardware and get tuned.' }),
      Check({ q: 'What does the barrier term do?', opts: ['Gives a soft repulsive push near the joint limits for safety', 'Cancels gravity', 'Damps velocity'], ans: 0, why: 'It keeps the arm away from its hard limits.' }),
    ];
  },
  cards: [
    { q: 'Write the arm control law.', a: '<b>τ = g(q) + C(q,q̇)q̇ + Kp(q_des - q) + Kd(q̇_des - q̇) + barrier(q)</b>: gravity compensation, Coriolis, spring, damper, and a soft joint-limit barrier.' },
    { q: 'Why does the arm need gravity compensation?', a: 'It’s in free space with gravity pulling it down. A plain spring sags below the target and fights gravity constantly.' },
    { q: 'Which arm terms are computed and which are tuned?', a: 'g(q) and C(q,q̇)q̇ are computed from the dynamics model. Kp and Kd are tuned by hand on the hardware.' },
    { q: 'What does the barrier term do?', a: 'A soft repulsive push near the joint limits, for safety.' },
  ],
});

/* 4 ─ tools, URDF, teach mode */
lesson({
  id: 'c-tools', ch: 'ctl', mins: 10,
  title: 'Pinocchio, CRISP and the URDF',
  lede: 'Gravity and Coriolis torques come from a dynamics library, and a controller that uses one already exists. What’s left for you is correct inertials, gains and a teach-mode test.',
  build() {
    const pathW = Widget('Two ways to get the arm controller', body => {
      let m = 'A'; const box = (t, d, strong) => h('div', { style: { border: '1px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'transparent', borderRadius: '8px', padding: '9px 13px' } }, h('b', { style: { fontSize: '14.5px' } }, t), d ? h('div', { style: { fontSize: '13px', color: 'var(--ink2)' } }, d) : null);
      const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } }), out = h('div.readout'), info = h('p.cap');
      const arrow = () => h('div.mono', { style: { textAlign: 'center', color: 'var(--ink3)' } }, '↓');
      const draw = () => {
        st.innerHTML = '';
        st.append(box('What you provide', m === 'A' ? 'the URDF with correct inertials, the Kp and Kd gains, and the position targets from the policy' : 'the URDF with correct inertials, the Kp and Kd gains, the position targets, and the controller code'), arrow());
        if (m === 'A') st.append(box('CRISP: a ready-made ros2_control controller', 'implements the whole law: gravity + Coriolis + Kp + Kd + barrier', true), h('div', { style: { marginLeft: '28px' } }, box('Pinocchio, inside it', 'RNEA gives g(q) and C(q, q̇)·q̇ from the URDF')));
        else st.append(box('Your own controller', 'a ros2_control controller you write: computes the law and writes efforts', true), h('div', { style: { marginLeft: '28px' } }, box('Pinocchio, called by your code', 'RNEA gives g(q) and C(q, q̇)·q̇ from the URDF')));
        st.append(arrow(), box('ros2_control effort interface', 'torque per arm joint'));
        out.innerHTML = ''; out.append(m === 'A' ? Chip('you write', 'config only', 'ok') : Chip('you write', 'a controller', 'hl'), m === 'A' ? Chip('recommended', 'yes, start here', 'ok') : Chip('use when', 'CRISP doesn’t fit'));
        info.textContent = m === 'A' ? 'CRISP is built for learning-based manipulation policies and teleoperation, which is this project’s exact use case. You never touch Pinocchio directly.' : 'More work and more to learn. Only worth it if CRISP can’t do what you need.';
      };
      body.append(Seg([{ v: 'A', label: 'Path A: adopt CRISP' }, { v: 'B', label: 'Path B: write your own' }], 'A', v => { m = v; draw(); }), st, out, info); draw();
    });
    return [
      H2('Where gravity and Coriolis come from'),
      P('You don’t derive the dynamics yourself. A dynamics library does it. Pinocchio is the modern one and KDL is the older, simpler one. You give Pinocchio the robot’s URDF and the current joint angles, and it runs an algorithm called RNEA (Recursive Newton-Euler) that returns g(q) and C(q, q̇)·q̇.'),
      Trap('Pinocchio is only as right as the inertial parameters in your URDF: the mass, center of mass and inertia tensor of every link. If those are wrong, or left as placeholder zeros, which happens a lot, the gravity compensation is garbage and no amount of gain tuning saves it. Getting them right is mechanical and CAD work upstream of the software, and it’s the real prerequisite.'),
      H2('Pinocchio and CRISP'),
      P('People often treat these as alternatives, but one lives inside the other. Pinocchio is only a math engine. It knows nothing about ROS or controllers. CRISP (and the Cartesian-Impedance-Controller it builds on) is a complete ros2_control controller that implements the whole sum of terms and calls Pinocchio internally for the gravity and Coriolis terms.'),
      pathW,
      P('So there are two paths. Path A is to adopt CRISP: give it your URDF, tune the gains, and never touch Pinocchio directly because it’s already in there. Path B is to write your own controller and call Pinocchio yourself, which is more work and more learning. Start with Path A and fall back to B only if CRISP doesn’t fit.'),
      H2('Tuning Kp and Kd'),
      P('Kp and Kd can’t be computed. You tune them by hand on real hardware, watching for oscillation versus sluggishness. It takes patience. Like the legs, the big proximal joints get stiff gains and the distal ones get soft gains.'),
      H2('Teach mode: the first milestone'),
      P('Before tuning a single gain, run the controller with gravity compensation only, so Kp = 0 and Kd = 0. If the URDF inertials are correct, the arm becomes weightless. You can grab it, move it anywhere by hand, let go, and it stays where you left it, holding against gravity with no sag. That one test checks your whole gravity model. If the arm sags or drifts, the inertials are wrong and you fix those before anything else.'),
      P('Teach mode is also how you hand-guide the arm to collect teleoperation demonstrations for the manipulation policy. The widget in the previous lesson has a teach-mode setting where you can see what a wrong mass does.'),
      Real('Do teach mode before any policy or gain tuning. It’s the cheapest test there is, and it fails loudly if the URDF is wrong.'),
      Check({ q: 'What does Pinocchio compute for the arm controller?', opts: ['g(q) and C(q, q̇)·q̇, from the URDF and the current joint angles, using RNEA', 'Kp and Kd', 'The policy’s targets'], ans: 0, why: 'Pinocchio is only the dynamics engine.' }),
      Check({ q: 'How are Pinocchio and CRISP related?', opts: ['CRISP is a full ros2_control controller that calls Pinocchio internally', 'They’re competing libraries', 'Pinocchio is a controller and CRISP is a URDF parser'], ans: 0, why: 'Adopting CRISP means never touching Pinocchio directly.' }),
      Check({ q: 'What does teach mode test?', opts: ['The URDF inertials, through the whole gravity model', 'The Kp and Kd values', 'The policy'], ans: 0, why: 'With Kp = Kd = 0 a correct model makes the arm weightless. Sagging or drifting means the inertials are wrong.' }),
    ];
  },
  cards: [
    { q: 'How do you get g(q) and C(q,q̇)q̇?', a: 'From a dynamics library (Pinocchio, or the older KDL). It takes the URDF and the joint angles and runs RNEA (Recursive Newton-Euler).' },
    { q: 'What does Pinocchio need to be correct?', a: 'The URDF’s inertial parameters: mass, center of mass and inertia tensor of every link. Placeholder zeros make gravity compensation garbage. Fixing them is upstream mechanical and CAD work.' },
    { q: 'How do Pinocchio and CRISP relate?', a: 'Pinocchio is only the math engine. CRISP is a complete ros2_control controller that implements the whole law and calls Pinocchio inside. Path A: adopt CRISP. Path B: write your own controller and call Pinocchio.' },
    { q: 'What’s teach mode and why do it first?', a: 'Run with gravity compensation only (Kp = Kd = 0). With correct inertials the arm is weightless and stays wherever you leave it. It validates the whole gravity model, and it’s how you hand-guide the arm for teleoperation demos.' },
  ],
});

/* 5 ─ WBC */
lesson({
  id: 'c-wbc', ch: 'ctl', mins: 6,
  title: 'Where a whole-body controller fits',
  lede: 'The two controllers are enough while the robot walks and manipulates one after the other. A whole-body controller takes over when it has to do the two together.',
  build() {
    const wbcW = Widget('The same slot, two ways', body => {
      let m = 'v0'; const box = (t, d, strong) => h('div', { style: { border: '1px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'transparent', borderRadius: '8px', padding: '9px 13px', flex: '1 1 200px', minWidth: 0 } }, h('b', { style: { fontSize: '14.5px' } }, t), h('div', { style: { fontSize: '13px', color: 'var(--ink2)' } }, d));
      const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } }), info = h('p.cap');
      const draw = () => {
        st.innerHTML = '';
        st.append(h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } }, box('Manipulation policy', 'arm targets'), box('Locomotion policy', 'leg targets')), h('div.mono', { style: { textAlign: 'center', color: 'var(--ink3)' } }, '↓'),
          m === 'v0' ? h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } }, box('Arm controller', 'impedance control with gravity and Coriolis terms', true), box('Leg controller', 'plain fast PD', true)) : h('div', { style: { display: 'flex', gap: '8px' } }, box('Whole-body controller', 'one optimization over the whole body: balance, contacts, priorities, coupled dynamics', true)),
          h('div.mono', { style: { textAlign: 'center', color: 'var(--ink3)' } }, '↓'), h('div', { style: { display: 'flex' } }, box('ros2_control effort interface', 'torque per joint, unchanged')));
        info.textContent = m === 'v0' ? 'Each controller handles its own joints. There’s nothing to trade off between them yet.' : 'One solve coordinates arm and legs. It replaces both controllers and nothing above or below it has to change.';
      };
      body.append(Seg([{ v: 'v0', label: 'First version' }, { v: 'wbc', label: 'With a WBC' }], 'v0', v => { m = v; draw(); }), st, info); draw();
    });
    return [
      H2('What joint-level control is'),
      P('What this chapter built is joint-level torque control. Each joint’s torque comes from that joint’s own target, with an arm controller and a leg controller running side by side. It’s the ancestor of a whole-body controller.'),
      H2('What a WBC does differently'),
      P('A whole-body controller solves one optimization (typically a QP) across the entire body at once. It coordinates balance, multiple contacts and competing priorities under the full coupled dynamics, and it can push lower-priority tasks through the null space of higher-priority ones.'),
      P('You need that when the tasks compete inside a single solve, like reaching out while balancing on one foot, where “keep my balance” and “extend my arm” pull against each other.'),
      H2('Why the first version doesn’t need one'),
      P('A first version locomotes, then manipulates, and never does both at the same time. It never reaches while balancing on one foot, and it never has to trade off balance against arm extension in a single solve. So it doesn’t need the QP, MPC or contact optimization. Two simpler controllers side by side are enough: impedance control for the arm, because the arm is in free space, and plain PD for the legs, because the RL policy already absorbed the dynamics.'),
      wbcW,
      H2('Growing into one'),
      P('When the humanoid has to do real loco-manipulation, with arm and legs coordinating, a WBC goes in this same slot and replaces both controllers. The policies above it and ros2_control and the drives below it stay as they are.'),
      Note('Whichever controller sits in this slot, the layer takes position targets in and puts torques out, so the policies above it and ros2_control below it never need to know which one it is.'),
      Check({ q: 'What does a whole-body controller do that the two joint-level controllers don’t?', opts: ['Solves one optimization across the whole body, coordinating balance, contacts and competing priorities', 'Runs faster', 'Replaces the policies'], ans: 0, why: 'The joint-level controllers each handle their own joints and don’t coordinate.' }),
      Check({ q: 'Why does the first version get by without one?', opts: ['It locomotes and then manipulates, and never reaches while balancing, so nothing has to be traded off in one solve', 'WBCs aren’t compatible with ROS', 'The policies do the balancing'], ans: 0, why: 'Trade-offs between balance and arm extension only come up when both happen at once.' }),
    ];
  },
  cards: [
    { q: 'What does a whole-body controller do?', a: 'Solves one optimization (typically a QP) across the whole body, coordinating balance, multiple contacts and competing priorities under the full coupled dynamics.' },
    { q: 'Why doesn’t the first version need a WBC?', a: 'It locomotes and then manipulates, never both at once, and never reaches while balancing. So nothing has to be traded off in one solve. Arm impedance control plus leg PD is enough.' },
    { q: 'Where does a WBC go when the robot needs one?', a: 'In the same slot, between the policies and ros2_control, replacing both joint-level controllers. Nothing above or below changes.' },
  ],
});
