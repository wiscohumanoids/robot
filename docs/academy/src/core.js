'use strict';
/* ═════════ core: DOM helpers, store, teaching blocks ═════════ */
const NS = 'http://www.w3.org/2000/svg';
const $ = (s, r = document) => r.querySelector(s);

function apply(el, props, svg) {
  for (const k in props || {}) {
    const v = props[k];
    if (v == null || v === false) continue;
    if (k === 'class') el.setAttribute('class', ((el.getAttribute('class') || '') + ' ' + v).trim());
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'on') for (const e in v) el.addEventListener(e, v[e]);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (!svg && k === 'value') el.value = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
}
function addKids(el, list) {
  for (const c of list.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}
function h(tag, props, ...kids) {
  const name = (tag.match(/^[a-z0-9]*/i)[0]) || 'div', el = document.createElement(name), cls = [];
  for (const m of tag.slice(name.length).matchAll(/([.#])([^.#]+)/g)) { if (m[1] === '#') el.id = m[2]; else cls.push(m[2]); }
  if (cls.length) el.setAttribute('class', cls.join(' '));
  apply(el, props, false);
  if (cls.includes('cap') && kids.length && kids.every(k => typeof k === 'string')) el.innerHTML = kids.join(''); else addKids(el, kids);
  return el;
}
function S(tag, props, ...kids) {
  const el = document.createElementNS(NS, tag);
  apply(el, props, true); addKids(el, kids); return el;
}
const svgRoot = (w, hgt, extra) => S('svg', Object.assign({ viewBox: `0 0 ${w} ${hgt}`, role: 'img' }, extra));
function svgPt(svg, e) {
  const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
function drag(svg, onMove, onStart) {
  let on = false;
  svg.style.touchAction = 'none';
  svg.addEventListener('pointerdown', e => { on = true; svg.setPointerCapture(e.pointerId); onStart && onStart(svgPt(svg, e)); onMove(svgPt(svg, e)); });
  svg.addEventListener('pointermove', e => { if (on) onMove(svgPt(svg, e)); });
  const off = () => { on = false; };
  svg.addEventListener('pointerup', off); svg.addEventListener('pointercancel', off);
}

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const fx = (x, d = 2) => (Math.abs(x) < 1e-9 ? 0 : x).toFixed(d);
function rng(seed) { // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function shuffled(arr, seed) {
  const r = rng(seed || (Math.random() * 1e9) | 0), a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function path(pts) { return pts.map((p, i) => (i ? 'L' : 'M') + fx(p[0], 1) + ' ' + fx(p[1], 1)).join(' '); }

/* ───────── cleanup registry (timers etc. die when you leave a lesson) ───────── */
const _leave = [];
const onLeave = fn => _leave.push(fn);
function runLeave() { while (_leave.length) { try { _leave.pop()(); } catch (e) { /* noop */ } } }
function loop(fn) { // rAF loop bound to the page lifetime; fn(dt) returns false to stop
  let id, last = performance.now(), dead = false;
  const tick = t => { if (dead) return; const dt = Math.min(.05, (t - last) / 1000); last = t; if (fn(dt) === false) return; id = requestAnimationFrame(tick); };
  id = requestAnimationFrame(tick);
  const stop = () => { dead = true; cancelAnimationFrame(id); };
  onLeave(stop); return stop;
}

/* ───────── store ───────── */
const KEY = 'sentence-to-servo:v1';
let ST = { done: {}, seen: {}, cards: {} };
try { ST = Object.assign(ST, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { /* storage unavailable */ }
function save() { try { localStorage.setItem(KEY, JSON.stringify(ST)); } catch (e) { /* noop */ } }
function toast(msg) {
  const t = h('div.toast', { role: 'status' }, msg); document.body.append(t);
  setTimeout(() => t.remove(), 2400);
}

/* ───────── registry ───────── */
const CHAPTERS = [
  { id: 'map', title: 'The map', blurb: 'The layers between a command and a motor, and how fast each one runs.' },
  { id: 'see', title: 'Perception', blurb: 'Visual SLAM, loop closure, putting objects in the map, and how it plugs into ROS.' },
  { id: 'plan', title: 'Planning and navigation', blurb: 'The task planner, the behavior tree and Nav2, from a goal to a velocity command.' },
  { id: 'walk', title: 'Learning to walk', blurb: 'Reinforcement learning for locomotion, from parallel simulation through PPO to getting a policy onto hardware.' },
  { id: 'arms', title: 'Arms and hands', blurb: 'Manipulation methods in the order they build on each other.' },
  { id: 'ctl', title: 'Controls', blurb: 'Turning the policies’ position targets into torques: PD for the legs, impedance control for the arm, and where a whole-body controller fits.' },
  { id: 'hw', title: 'Hardware interface', blurb: 'ros2_control, EtherCAT and the drive, from a torque command to current in a motor winding.' },
];
const LESSONS = [];
const lessonById = id => LESSONS.find(l => l.id === id);
const chapterOf = id => CHAPTERS.find(c => c.id === id);
function lesson(def) { LESSONS.push(def); }

/* ───────── lesson context: practice questions collect here, off the reading path ───────── */
const LC = { practice: [], reset() { this.practice = []; } };

/* ───────── content blocks ───────── */
const P = html => h('p', { html });
const H2 = t => h('h2', null, t);
const H3 = t => h('h3', null, t);
const UL = (...items) => h('ul', null, items.map(i => h('li', { html: i })));
const OL = (...items) => h('ol', null, items.map(i => h('li', { html: i })));
const Eq = html => h('div.eq', { html });
const Callout = (kind, title, html) => h('div.callout.' + kind, null, h('span.ct', null, title), h('p', { html }));
const Key = html => Callout('key', 'In short', html);
const Trap = html => Callout('trap', 'Common mistake', html);
const Note = html => Callout('note', 'Note', html);
const Real = html => Callout('real', 'On a real robot', html);
const Deeper = (title, html) => h('details.deeper', null, h('summary', null, title), h('p', { html }));

function Widget(title, build, o = {}) {
  const body = h('div.wbody');
  const el = h('section.widget', null, h('header', null, h('b', null, title), o.hint ? h('span', null, o.hint) : null), body);
  build(body); return el;
}
function Slider({ label, min = 0, max = 1, step = .01, value = 0, fmt = v => fx(v, 2), on }) {
  const out = h('output'); const inp = h('input', { type: 'range', min, max, step, value, 'aria-label': label });
  const show = () => { out.textContent = fmt(+inp.value); };
  inp.addEventListener('input', () => { show(); on && on(+inp.value); }); show();
  const el = h('label.ctl', null, h('span.lab', null, label), inp, out);
  el.get = () => +inp.value; el.set = v => { inp.value = v; show(); };
  return el;
}
function Seg(options, value, on) {
  const el = h('div.seg', { role: 'group' }); let cur = value;
  const btns = options.map(o => {
    const v = typeof o === 'object' ? o.v : o, lab = typeof o === 'object' ? o.label : o;
    const b = h('button', { type: 'button', 'aria-pressed': String(v === cur), on: { click() { cur = v; btns.forEach(x => x.b.setAttribute('aria-pressed', String(x.v === cur))); on(v); } } }, lab);
    return { b, v };
  });
  btns.forEach(x => el.append(x.b)); el.get = () => cur; return el;
}
const Btn = (label, on, cls = '') => h('button.btn' + (cls ? '.' + cls.split(' ').join('.') : ''), { type: 'button', on: { click: on } }, label);
const Chip = (label, val, cls = '') => h('span.chip' + (cls ? '.' + cls : ''), null, label + ' ', h('b', null, val));

/* ───────── practice blocks: not rendered inline; gathered into a collapsed section at the end ───────── */
function Check(o) { LC.practice.push({ q: o.q, a: '<b>' + o.opts[o.ans] + '</b><br>' + o.why }); return null; }
const Predict = Check;
function Order(o) { LC.practice.push({ q: o.q, a: '<ol>' + o.items.map(i => '<li>' + i + '</li>').join('') + '</ol>' + (o.why || '') }); return null; }
function Match(o) { LC.practice.push({ q: o.q, a: o.rows.map(r => r.l + ' → <b>' + r.r + '</b>').join('<br>') + '<br><br>' + (o.why || '') }); return null; }
