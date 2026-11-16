'use strict';
/* ═════════ pages: shell, home, lesson, review, router ═════════ */

const DAY = 86400000, INTERVALS = [0, 0, 1, 3, 7, 14, 30]; // days by Leitner box (1..6)
const allCards = () => LESSONS.flatMap(L => (L.cards || []).map((c, i) => ({ id: L.id + ':' + i, q: c.q, a: c.a, ch: L.ch, lesson: L })));
const isDue = c => ST.cards[c.id] && ST.cards[c.id].due <= Date.now();
const dueCount = () => allCards().filter(isDue).length;
const orderIn = L => LESSONS.filter(x => x.ch === L.ch).indexOf(L) + 1;
const SHORT = { descent: 'How the pieces fit together', 'w-worlds': 'Parallel worlds', 'w-senses': 'Observations and actions', 'w-reward': 'Reward, return, value', 'w-advantage': 'TD error and GAE', 'w-ppo': 'The PPO objective', 'w-sandbox': 'PPO on one joint', 'w-learn': 'Backprop and Adam', 'w-pd': 'Targets and PD control', 'w-sim2real': 'Sim to real', 'w-yours': 'What you write', 'a-ik': 'Inverse kinematics', 'a-act': 'ACT', 'a-diff': 'Diffusion policy', 'a-hybrid': 'Geometry plus learning', 'a-vla': 'VLAs', 'a-frontier': 'Learning from experience', 'p-slam': 'What SLAM is', 'p-orb': 'ORB-SLAM3', 'p-loop': 'Drift and loop closure', 'p-frames': 'Objects, frames, TF', 'p-team': 'Build vs use', 'c-layer': 'The control layer', 'c-legs': 'Legs: PD control', 'c-arm': 'Arm: impedance control', 'c-tools': 'Pinocchio, CRISP, URDF', 'c-wbc': 'Where a WBC fits', 'b-planner': 'Task planner', 'b-bt': 'Behavior tree', 'b-cmdvel': 'Nav2 and /cmd_vel', 'b-ros2c': 'ros2_control', 'b-ethercat': 'EtherCAT', 'b-foc': 'FOC and CAN', 'b-capstone': 'The full chain' };

/* ───────── the descent ladder (used in the Map lesson) ───────── */
const LAYERS = [
  { nm: 'You', sub: 'A joystick, or a goal like “bring me the cube”', n: 1, unit: 'per goal', ch: 'map', to: 'descent' },
  { nm: 'Task planner (LLM)', sub: 'Runs once per goal and outputs a skill list', n: 1, unit: 'per goal', ch: 'plan', to: 'b-planner' },
  { nm: 'Behavior tree', sub: 'Ticks the safety guards and the active skill', n: 20, unit: 'ticks / s', ch: 'plan', to: 'b-bt' },
  { nm: 'Camera + SLAM tracking', sub: 'ORB feature tracking on every frame', n: 30, unit: 'frames / s', ch: 'see', to: 'p-orb' },
  { nm: 'Nav2 local controller', sub: 'Turns the path into /cmd_vel', n: 20, unit: 'cmds / s', ch: 'plan', to: 'b-cmdvel' },
  { nm: 'Locomotion policy', sub: '/cmd_vel in, leg targets out (50 to 100 Hz)', n: 50, unit: 'actions / s', ch: 'walk', to: 'w-senses' },
  { nm: 'Manipulation policy', sub: 'Camera and joints in, arm targets out (about 50 Hz is typical)', n: 50, unit: 'actions / s', ch: 'arms', to: 'a-act' },
  { nm: 'Leg controller (PD)', sub: 'Leg targets to torques, 1 to 2 kHz', n: 1000, unit: 'torques / s', ch: 'ctl', to: 'c-legs' },
  { nm: 'Arm controller (impedance)', sub: 'Arm targets to torques, 1 to 2 kHz', n: 1000, unit: 'torques / s', ch: 'ctl', to: 'c-arm' },
  { nm: 'ros2_control', sub: 'Read state, run controllers, write commands', n: 1000, unit: 'loops / s', ch: 'hw', to: 'b-ros2c' },
  { nm: 'EtherCAT frame', sub: 'One frame per cycle, through every drive', n: 1000, unit: 'frames / s', ch: 'hw', to: 'b-ethercat' },
  { nm: 'FOC inside each drive', sub: 'Phase currents at 20 to 30 kHz', n: 30000, unit: 'cycles / s', ch: 'hw', to: 'b-foc' },
];
const logw = n => 100 * Math.max(.05, Math.log10(n + 1) / Math.log10(30001));
const fmtN = n => Math.round(n).toLocaleString('en-US');

function Ladder() {
  const rows = LAYERS.map(L => {
    const bar = h('i', { style: { width: logw(L.n) + '%' } }), ct = h('div.ct', null, fmtN(L.n), h('small', null, L.unit));
    return { L, bar, ct, a: h('a.layer', { href: '#/l/' + L.to, 'data-ch': L.ch }, h('div.nm', null, h('b', null, L.nm), h('small', null, L.sub)), h('div.bar', null, bar), ct) };
  });
  const el = h('div.ladder', null, rows.map(r => r.a)); let t0 = 0, stop = null;
  el.run = () => {
    if (stop) stop(); t0 = 0; rows.forEach(r => { r.bar.style.width = '0%'; });
    stop = loop(dt => {
      t0 += dt; const t = Math.min(1, t0 / 2.4), e = 1 - Math.pow(1 - t, 3);
      rows.forEach(r => { const v = r.L.n * e; r.ct.firstChild.textContent = fmtN(r.L.n <= 1 ? Math.ceil(v) : v); r.bar.style.width = (v < 1 ? 0 : logw(v)) + '%'; });
      return t < 1;
    });
  };
  return el;
}

/* ───────── home graphics: three small living pictures ───────── */
function miniWalk() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'Simulated robots falling less often as the reward curve climbs' }), r = rng(4);
  const thr = Array.from({ length: 60 }, () => r()), dots = thr.map((_, i) => Circle(36 + (i % 10) * 24, 44 + Math.floor(i / 10) * 30, 8, 'var(--line)'));
  const cx0 = 320, cy0 = 200, cw = 260, ch = 150, curve = Path('M0 0', 'var(--c)', 3), head = Circle(0, 0, 5.5, 'var(--c)');
  svg.append(T(24, 22, 'SIMULATED ROBOTS (lit = standing)', 'axis'), ...dots, Line(cx0, cy0, cx0 + cw, cy0, 'var(--line)', 1.5), Line(cx0, cy0, cx0, cy0 - ch, 'var(--line)', 1.5), T(cx0, 22, 'REWARD', 'axis'), T(cx0 + cw, cy0 + 18, 'iterations →', 'axis', 'end'), curve, head);
  let t = 0;
  loop(dt => {
    t += dt; const cyc = 9.5, u = Math.min(1, (t % cyc) / 7.5), f = 1 - Math.exp(-3.4 * u), pts = [];
    for (let k = 0; k <= 80; k++) { const v = k / 80 * u; pts.push([cx0 + v * cw, cy0 - (1 - Math.exp(-3.4 * v) + .05 * Math.sin(v * 46) * (1 - v)) * (ch - 8)]); }
    curve.setAttribute('d', path(pts)); const e = pts[pts.length - 1]; head.setAttribute('cx', e[0]); head.setAttribute('cy', e[1]);
    dots.forEach((d, i) => { const up = thr[i] < .1 + .88 * f; d.setAttribute('fill', up ? 'var(--c)' : 'var(--line)'); d.setAttribute('r', up ? 8 : 5); });
  });
  return svg;
}
function miniArms() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'A three-joint arm continuously chasing a moving target' }), BX = 300, BY = 208, K = .6;
  const X = p => BX + K * p[0], Y = p => BY - K * p[1], links = [0, 1, 2].map(i => Line(0, 0, 0, 0, 'var(--c)', 11 - i * 2.5)), joints = [0, 1, 2].map(() => Circle(0, 0, 7, 'var(--surface)', { stroke: 'var(--ink)', 'stroke-width': 2.5 })), tip = Circle(0, 0, 6, 'var(--ink)');
  const trail = Array.from({ length: 26 }, () => Circle(0, 0, 3, 'var(--hl)')), tgt = Circle(0, 0, 13, 'none', { stroke: 'var(--hl)', 'stroke-width': 2.5 });
  svg.append(Line(BX - 190, BY + 8, BX + 190, BY + 8, 'var(--line)', 2), S('path', { d: `M${BX - 151} ${BY} A151 151 0 0 1 ${BX + 151} ${BY}`, fill: 'none', stroke: 'var(--line)', 'stroke-dasharray': '4 6', 'stroke-width': 1.5 }), ...trail, ...links, ...joints, tip, tgt, T(20, 22, 'IK, FOLLOWING A TARGET', 'axis'));
  let th = [1.1, -.8, -.5], t = 0; const hist = [];
  loop(dt => {
    t += dt; const tg = [30 + 125 * Math.sin(t * .75), 105 + 58 * Math.sin(t * 1.15 + 1)];
    for (let i = 0; i < 3; i++) th = ikStep(th, tg, 'all', .3); const pts = ikFK(th);
    links.forEach((l, i) => { l.setAttribute('x1', X(pts[i])); l.setAttribute('y1', Y(pts[i])); l.setAttribute('x2', X(pts[i + 1])); l.setAttribute('y2', Y(pts[i + 1])); });
    joints.forEach((j, i) => { j.setAttribute('cx', X(pts[i])); j.setAttribute('cy', Y(pts[i])); });
    tip.setAttribute('cx', X(pts[3])); tip.setAttribute('cy', Y(pts[3])); tgt.setAttribute('cx', X(tg)); tgt.setAttribute('cy', Y(tg));
    hist.unshift(tg); hist.length = Math.min(hist.length, trail.length); trail.forEach((c, i) => { const p = hist[i]; if (p) { c.setAttribute('cx', X(p)); c.setAttribute('cy', Y(p)); c.setAttribute('opacity', (1 - i / trail.length) * .55); } });
  });
  return svg;
}
function miniStack() {
  const W = 600, Hh = 236, svg = svgRoot(W, Hh, { 'aria-label': 'Twelve layers ticking at very different rates' }), rows = [['you', 1, 'map'], ['planner', 1, 'plan'], ['tree', 20, 'plan'], ['SLAM', 30, 'see'], ['Nav2', 20, 'plan'], ['loco policy', 50, 'walk'], ['arm policy', 50, 'arms'], ['leg PD', 1000, 'ctl'], ['arm ctrl', 1000, 'ctl'], ['ros2_control', 1000, 'hw'], ['EtherCAT', 1000, 'hw'], ['FOC', 30000, 'hw']], lines = [];
  const hz = n => n >= 1000 ? (n / 1000) + ' kHz' : n + ' Hz';
  rows.forEach(([nm, n, ch], i) => {
    const y = 16 + i * 18.4, period = 440 / (3 + 55 * Math.log10(n + 1) / 4.48);
    svg.append(T(20, y + 4, nm, 'axis'), T(W - 14, y + 4, i ? hz(n) : '1 goal', 'axis', 'end'));
    const l = Line(150, y, 520, y, ch === 'map' ? 'var(--ink)' : 'var(--' + ch + ')', 9, { 'stroke-dasharray': `3 ${(period - 3).toFixed(1)}`, 'stroke-linecap': 'butt' }); lines.push([l, period]); svg.append(l);
  });
  let t = 0; loop(dt => { t += dt; lines.forEach(([l, p]) => l.setAttribute('stroke-dashoffset', (-(t * 80) % p).toFixed(2))); });
  return svg;
}
function miniPlan() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'A behavior tree ticking: guards checked every tick, one skill running' });
  const skills = ['navigate', 'pick', 'navigate', 'place'], sx = [80, 220, 360, 500], gx = [60, 130, 200];
  const root = Rect(215, 24, 170, 30, 'var(--surface2)', { rx: 6, stroke: 'var(--line)' });
  const edges = [...gx.map(x => Line(300, 54, x + 100, 112, 'var(--line)', 1.5)), ...sx.map(x => Line(300, 54, x + 20 + 40, 168, 'var(--line)', 1.5))];
  const guards = gx.map((x, i) => Circle(x + 100, 122, 11, 'var(--surface2)', { stroke: 'var(--line)', 'stroke-width': 1.5 }));
  const boxes = sx.map(x => Rect(x, 168, 120, 34, 'var(--surface2)', { rx: 6, stroke: 'var(--line)' })), fills = sx.map(x => Rect(x, 168, 0, 34, 'var(--c)', { rx: 6, opacity: .55 }));
  const pulse = Circle(300, 54, 5, 'var(--hl)');
  svg.append(...edges, root, T(300, 44, 'ReactiveSequence', 'axis', 'middle'), ...guards, ...['battery', 'upright', 'faults'].map((n, i) => T(gx[i] + 100, 150, n, 'axis', 'middle')), ...boxes, ...fills, ...skills.map((n, i) => T(sx[i] + 60, 189, n, 'lbl', 'middle')), pulse, T(20, 22, 'TICKS 20 TIMES A SECOND (SLOWED DOWN)', 'axis'));
  let t = 0;
  loop(dt => {
    t += dt; const tick = Math.floor(t / .5), ph = (t % .5) / .5, k = tick % 4, cyc = t % 10, act = Math.min(3, Math.floor(cyc / 2.4)), prog = clamp((cyc - act * 2.4) / 2.4, 0, 1);
    const tgt = k < 3 ? [gx[k] + 100, 122] : [sx[act] + 60, 168]; pulse.setAttribute('cx', lerp(300, tgt[0], ph)); pulse.setAttribute('cy', lerp(54, tgt[1], ph));
    guards.forEach((g, i) => g.setAttribute('fill', i === k && ph > .8 ? 'var(--ok)' : 'var(--surface2)'));
    boxes.forEach((b, i) => b.setAttribute('stroke', i === act ? 'var(--c)' : 'var(--line)')); fills.forEach((f, i) => f.setAttribute('width', i < act ? 120 : i === act ? 120 * prog : 0));
  });
  return svg;
}
function miniControls() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'A slow staircase of policy targets and a joint following them smoothly' }), x0 = 40, wpx = 540, top = 40, bot = 200, WIN = 5;
  const tgt = Path('M0 0', 'var(--hl)', 2, 'none', { 'stroke-dasharray': '4 3' }), resp = Path('M0 0', 'var(--c)', 3);
  svg.append(Line(x0, bot, x0 + wpx, bot, 'var(--line)', 1.5), Line(x0, (top + bot) / 2, x0 + wpx, (top + bot) / 2, 'var(--line)', 1), T(20, 22, 'POLICY TARGETS (STEPS) AND THE JOINT FOLLOWING THEM', 'axis'), T(x0 + wpx, Hh - 12, 'torque recomputed 1000+ times a second', 'axis', 'end'), tgt, resp);
  let t = 0, q = 0, w = 0, acc = 0; const hist = []; const I = .08, kp = 60, kd = 2.4;
  const step = dt => { t += dt; acc += dt; const tk = Math.floor(t / .3) * .3, target = .8 * Math.sin(tk * 1.2 + .4); while (acc >= .002) { acc -= .002; const tau = kp * (target - q) - kd * w; w += tau / I * .002; q += w * .002; } hist.push([t, target, q]); while (hist.length && hist[0][0] < t - WIN) hist.shift(); };
  for (let i = 0; i < WIN * 60; i++) step(1 / 60);
  loop(dt => {
    step(dt);
    const px = s => x0 + (s - (t - WIN)) / WIN * wpx, py = v => (top + bot) / 2 - v * 70;
    tgt.setAttribute('d', path(hist.map(a => [px(a[0]), py(a[1])]))); resp.setAttribute('d', path(hist.map(a => [px(a[0]), py(a[2])])));
  });
  return svg;
}
function miniChain() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'One frame travelling through a chain of drives and back' }), N = 9, x0 = 110, gap = 56;
  const master = Rect(20, 96, 60, 40, 'var(--ink)', { rx: 6 }), drives = Array.from({ length: N }, (_, i) => Rect(x0 + i * gap, 100, 34, 32, 'var(--surface2)', { rx: 6, stroke: 'var(--line)' }));
  const dot = Circle(80, 116, 6, 'var(--hl)');
  svg.append(Line(80, 116, x0 + (N - 1) * gap + 34, 116, 'var(--line)', 3), master, T(50, 121, 'master', 'lbl', 'middle', { style: 'fill:var(--bg)' }), ...drives, dot, T(20, 22, 'ONE FRAME PER MILLISECOND, THROUGH EVERY DRIVE (SLOWED DOWN)', 'axis'), T(x0 + (N - 1) * gap + 17, 176, 'each drive reads its part as the frame passes', 'axis', 'end'));
  let t = 0;
  loop(dt => {
    t += dt; const u = (t % 2.4) / 2.4, span = x0 + (N - 1) * gap + 34 - 80, pos = u < .5 ? u * 2 : 2 - u * 2, x = 80 + span * pos;
    dot.setAttribute('cx', x); drives.forEach((d, i) => { const cx = x0 + i * gap + 17; d.setAttribute('fill', Math.abs(cx - x) < 20 ? 'var(--c)' : 'var(--surface2)'); });
  });
  return svg;
}

function miniSee() {
  const W = 600, Hh = 230, svg = svgRoot(W, Hh, { 'aria-label': 'A camera path drifting around a loop, then snapping back when the loop closes' });
  const N = 100, X0 = 130, Y0 = 34, RW = 340, RH = 150, R = 30;
  const P = u => { // rounded-rectangle loop, u in [0,1]
    const per = 2 * (RW + RH), s = u * per; let x, y;
    if (s < RW) { x = s; y = 0; } else if (s < RW + RH) { x = RW; y = s - RW; } else if (s < 2 * RW + RH) { x = RW - (s - RW - RH); y = RH; } else { x = 0; y = RH - (s - 2 * RW - RH); }
    return [X0 + x, Y0 + y];
  };
  const rr = rng(9), lm = Array.from({ length: 26 }, () => [X0 - 20 + rr() * (RW + 40), Y0 - 16 + rr() * (RH + 32)]);
  const lmEls = lm.map(p => Circle(p[0], p[1], 2.4, 'var(--ink3)'));
  const truth = Path(path(Array.from({ length: N + 1 }, (_, k) => P(k / N))), 'var(--line)', 2, 'none', { 'stroke-dasharray': '4 5' });
  const est = Path('M0 0', 'var(--c)', 3), cam = Circle(0, 0, 6, 'var(--c)'), link = Line(0, 0, 0, 0, 'var(--hl)', 2, { 'stroke-dasharray': '3 3' }), lab = T(20, 22, 'ESTIMATED PATH', 'axis');
  svg.append(...lmEls, truth, link, est, cam, lab, T(W - 20, Hh - 12, 'dashed: actual path', 'axis', 'end'));
  const DR = [52, -38]; let t = 0;
  loop(dt => {
    t += dt; const cyc = 10, tt = t % cyc, u = Math.min(1, tt / 6), s = tt < 6.8 ? 0 : Math.min(1, (tt - 6.8) / 1.2), pts = [];
    for (let k = 0; k <= N * u; k++) { const v = k / N, p = P(v), d = Math.pow(v, 1.3) * (1 - s); pts.push([p[0] + DR[0] * d, p[1] + DR[1] * d]); }
    if (pts.length < 2) pts.push(P(0), P(0));
    est.setAttribute('d', path(pts)); const e = pts[pts.length - 1]; cam.setAttribute('cx', e[0]); cam.setAttribute('cy', e[1]);
    const gap = tt >= 6 && tt < 8, st = P(0);
    link.setAttribute('x1', st[0]); link.setAttribute('y1', st[1]); link.setAttribute('x2', e[0]); link.setAttribute('y2', e[1]); link.setAttribute('opacity', gap ? 1 : 0);
    lab.textContent = tt < 6 ? 'DRIFTING' : tt < 6.8 ? 'LOOP CLOSURE' : 'ERROR SPREAD OUT';
  });
  return svg;
}

/* ───────── shell ───────── */
const BRAND = '<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="4" fill="var(--walk)"/><path d="M7 24 L14 12 L25 9" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="square"/><circle cx="14" cy="12" r="3" fill="#0B0909"/></svg>';
function buildRail(activeId) {
  const rail = $('#rail'); rail.innerHTML = '';
  rail.append(h('a.tbtn', { href: '#/', style: { marginBottom: '6px' } }, '← Home'));
  CHAPTERS.forEach(ch => {
    const wrap = h('div', { 'data-ch': ch.id }, h('h4', null, ch.title));
    LESSONS.filter(l => l.ch === ch.id).forEach(l => wrap.append(h('a.lnk' + (ST.done[l.id] ? '.done' : ''), { href: '#/l/' + l.id, 'aria-current': l.id === activeId ? 'page' : null }, SHORT[l.id] || l.title)));
    rail.append(wrap);
  });
  const due = dueCount(), rv = $('#reviewLink'); if (rv) rv.innerHTML = 'Review' + (due ? ' <i>' + due + '</i>' : '');
}
function shell() {
  const root = $('#root'); root.innerHTML = '';
  const scrim = h('div.scrim', { on: { click: () => toggle(false) } }), toggle = on => { $('#rail').classList.toggle('open', on); scrim.classList.toggle('open', on); };
  const theme = h('button.tbtn', { type: 'button', 'aria-label': 'Toggle light or dark theme', on: { click() {
    const r = document.documentElement, cur = r.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'), nx = cur === 'dark' ? 'light' : 'dark';
    r.setAttribute('data-theme', nx); try { localStorage.setItem(KEY + ':theme', nx); } catch (e) { /* noop */ }
  } } }, 'Theme');
  root.append(scrim, h('nav.rail', { id: 'rail', 'aria-label': 'All lessons' }),
    h('div.topbar', null, h('a.brand', { href: '#/', html: BRAND + '<span>WisCortex</span>' }), h('div.crumb#crumb'), h('a.tbtn#reviewLink', { href: '#/review' }, 'Review'), h('button.tbtn', { type: 'button', on: { click: () => toggle(true) } }, 'Lessons'), theme),
    h('main.main#view', { tabindex: '-1' }));
  document.addEventListener('click', e => { if (e.target.closest('#rail a')) toggle(false); });
}

/* ───────── home ───────── */
function home() {
  const TR = [
    { ch: 'see', title: 'Perception', line: 'Visual SLAM, loop closure, and how detected objects end up in the map.', fig: miniSee },
    { ch: 'plan', title: 'Planning and navigation', line: 'The task planner, the behavior tree and Nav2: how a goal becomes skills and then a velocity command.', fig: miniPlan, flip: true },
    { ch: 'walk', title: 'Learning to walk', line: 'Reinforcement learning on thousands of simulated robots at once, and how the trained policy gets onto hardware.', fig: miniWalk },
    { ch: 'arms', title: 'Arms and hands', line: 'Manipulation methods in the order they build on each other: IK, ACT, diffusion policy, VLAs, then RL on top.', fig: miniArms, flip: true },
    { ch: 'ctl', title: 'Controls', line: 'The layer that turns both policies’ position targets into torques: PD for the legs, impedance control for the arm.', fig: miniControls },
    { ch: 'hw', title: 'Hardware interface', line: 'ros2_control, EtherCAT and the drive: from a torque command to current in a motor winding.', fig: miniChain, flip: true },
  ];
  const node = h('div.page.wide', { 'data-ch': 'map' },
    h('section.hero', null,
      h('div.eyebrow', null, 'Study notes'),
      h('h1', null, 'The WiscoHumanoids software stack', h('em', null, '.')),
      h('p.lede', null, 'Short lessons on the software behind the humanoid: perception, planning, the walking and manipulation policies, the control layer that turns their targets into torques, and the hardware interface down to the motors. Most of the diagrams move when you push on them.'),
      h('div.row', null, h('a.btn', { href: '#/l/descent' }, 'Start here'), h('a.btn.ghost', { href: '#/review' }, 'Review deck')), h('div.hero-fig', null, miniStack())),
    h('div', { style: { marginTop: '34px' } }, TR.map(t => {
      const ls = LESSONS.filter(l => l.ch === t.ch);
      return h('section.band' + (t.flip ? '.flip' : ''), { 'data-ch': t.ch },
        h('div.txt', null, h('div.eyebrow', null, ls.length + ' lessons'), h('h2', null, h('a', { href: '#/l/' + ls[0].id }, t.title)), h('p', null, t.line),
          h('ol', null, ls.map(l => h('li', null, h('a' + (ST.done[l.id] ? '.done' : ''), { href: '#/l/' + l.id }, SHORT[l.id] || l.title, h('small', null, l.mins + ' min')))))),
        h('div.fig', null, t.fig()));
    })));
  return { node, crumb: '' };
}

/* ───────── lesson ───────── */
function lessonPage(id) {
  const L = lessonById(id); if (!L) return { node: h('div.page', null, h('h1', null, 'Lesson not found'), h('p', null, h('a', { href: '#/' }, 'Back home'))), crumb: '' };
  const ch = chapterOf(L.ch), list = LESSONS.filter(x => x.ch === L.ch), ORD = CHAPTERS.flatMap(c => LESSONS.filter(x => x.ch === c.id)), i = ORD.indexOf(L), cards = L.cards || [];
  ST.seen[id] = 1; save(); LC.reset();
  const prose = h('div.prose');
  try { prose.append(...L.build().filter(Boolean)); } catch (e) { prose.append(Callout('trap', 'Lesson failed to render', String(e && e.message))); console.error(e); }
  const nxt = ORD[i + 1], prv = ORD[i - 1];
  const save1 = Btn(ST.done[id] ? 'Saved to review deck ✓' : 'Save to review deck', () => {
    ST.done[id] = 1; cards.forEach((_, k) => { const cid = id + ':' + k; if (!ST.cards[cid]) ST.cards[cid] = { box: 1, due: Date.now() }; }); save(); buildRail(id); save1.textContent = 'Saved to review deck ✓'; toast(cards.length + ' cards added to your review deck');
  }, 'ghost sm');
  const node = h('article.page', { 'data-ch': L.ch },
    h('div.eyebrow', null, ch.title, h('i', null, '·'), orderIn(L) + ' of ' + list.length, h('i', null, '·'), '~' + L.mins + ' min'),
    h('h1', null, L.title), h('p.lede', null, L.lede), prose,
    cards.length ? h('section.short', null, h('h3', null, 'The short version'), h('ul', null, cards.map(c => h('li', null, h('b', { html: c.q }), h('span', { html: c.a })))), h('div.row', null, save1)) : null,
    LC.practice.length ? h('details.practice', null, h('summary', null, 'Practice questions · ' + LC.practice.length), LC.practice.map(p => h('details', null, h('summary', { html: p.q }), h('div', { html: p.a })))) : null,
    h('div.lesson-nav', null, prv ? h('a.btn.ghost', { href: '#/l/' + prv.id }, '← ' + (SHORT[prv.id] || prv.title)) : h('span'), nxt ? h('a.btn', { href: '#/l/' + nxt.id }, (SHORT[nxt.id] || nxt.title) + ' →') : h('a.btn', { href: '#/review' }, 'Review deck →')));
  return { node, crumb: ch.title + ' / ' + (SHORT[id] || L.title), active: id };
}

/* ───────── review (Leitner boxes) ───────── */
function reviewPage() {
  const root = h('div.page', { 'data-ch': 'map' });
  const render = () => {
    root.innerHTML = '';
    const cards = allCards(), learned = cards.filter(c => ST.cards[c.id]), due = learned.filter(isDue);
    root.append(h('div.eyebrow', null, 'Spaced review', h('i', null, '·'), learned.length + ' of ' + cards.length + ' cards saved'), h('h1', null, 'Review deck'),
      h('p.lede', null, 'Cards come back on a schedule. A missed card returns the same day. A correct one waits 1, 3, 7, 14, then 30 days. Chapters are mixed together.'));
    const counts = [1, 2, 3, 4, 5, 6].map(b => learned.filter(c => ST.cards[c.id].box === b).length), mx = Math.max(1, ...counts);
    if (learned.length) root.append(h('div.widget', null, h('div.wbody', null, h('div.bars', null, counts.map(n => h('div', { style: { height: Math.max(6, 44 * n / mx) + 'px' }, title: n + ' cards' }))), h('div.bars-l', null, counts.map((_, i) => h('span', null, 'Box ' + (i + 1) + ' · ' + counts[i]))))));
    const actions = h('div.row', { style: { marginTop: '22px' } });
    if (!learned.length) actions.append(Btn('Add every card to the deck', () => { cards.forEach(c => { ST.cards[c.id] = { box: 1, due: Date.now() }; }); save(); buildRail(); render(); }), h('span.cap', null, 'You can also save a lesson’s cards from the bottom of its page.'));
    else actions.append(Btn(due.length ? 'Start review · ' + due.length + ' cards' : 'Nothing due. Review everything anyway', () => session(due.length ? due : learned)));
    root.append(actions);
  };
  const session = pool => {
    let q = shuffled(pool, Date.now() % 1e6), total = q.length, done = 0, missed = 0;
    const step = () => {
      root.innerHTML = '';
      if (!q.length) { root.append(h('div.eyebrow', null, 'Session complete'), h('h1', null, (done - missed) + ' of ' + done + ' first try'), h('p.lede', null, missed ? 'The ones you missed will come back today.' : 'Every card was right on the first try.'), h('div.row', null, Btn('Back to deck', () => { buildRail(); render(); }))); return; }
      const c = q[0], ans = h('div.a', { html: c.a, hidden: true }), acts = h('div.row');
      const grade = ok => { const s = ST.cards[c.id] || { box: 1 }; q.shift(); if (ok) { s.box = Math.min(6, s.box + 1); s.due = Date.now() + INTERVALS[s.box] * DAY; done++; } else { s.box = 1; s.due = Date.now(); missed++; q.push(c); total++; } ST.cards[c.id] = s; save(); step(); };
      acts.append(Btn('Show answer', () => { ans.hidden = false; acts.innerHTML = ''; acts.append(Btn('Missed it', () => grade(false), 'ghost'), Btn('Got it', () => grade(true))); }));
      root.append(h('div.eyebrow', { 'data-ch': c.ch }, chapterOf(c.ch).title, h('i', null, '·'), done + ' / ' + total), h('div.flash', { 'data-ch': c.ch, style: { marginTop: '16px' } }, h('div.q', { html: c.q }), ans, acts), h('p.cap', { style: { marginTop: '12px' } }, 'From ', h('a', { href: '#/l/' + c.lesson.id }, SHORT[c.lesson.id] || c.lesson.title)));
    };
    step();
  };
  render();
  return { node: root, crumb: 'Review deck' };
}

/* ───────── router ───────── */
function route() {
  runLeave();
  const parts = (location.hash.replace(/^#/, '') || '/').split('/').filter(Boolean);
  let res;
  try { res = parts[0] === 'l' ? lessonPage(parts[1]) : parts[0] === 'review' ? reviewPage() : home(); }
  catch (e) { console.error(e); res = { node: h('div.page', null, h('h1', null, 'Something broke'), h('pre', null, String(e && e.stack))), crumb: '' }; }
  const view = $('#view'); view.innerHTML = ''; view.append(res.node); $('#crumb').textContent = res.crumb || ''; buildRail(res.active);
  window.scrollTo(0, 0);
}
function boot() {
  try { const t = localStorage.getItem(KEY + ':theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) { /* noop */ }
  shell(); route(); window.addEventListener('hashchange', route);
}
