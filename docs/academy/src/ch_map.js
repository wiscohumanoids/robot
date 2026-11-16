'use strict';
/* ═════════ shared plot helper + Chapter 0 ═════════ */

function Plot(W, Hh, o) {
  const m = Object.assign({ l: 46, r: 14, t: 14, b: 36 }, o.m), svg = svgRoot(W, Hh, { 'aria-label': o.label || 'chart' });
  const sx = v => m.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * (W - m.l - m.r);
  const sy = v => Hh - m.b - (v - o.y[0]) / (o.y[1] - o.y[0]) * (Hh - m.t - m.b);
  (o.yt || []).forEach(v => svg.append(
    S('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: 'var(--line)', 'stroke-width': 1 }),
    S('text', { x: m.l - 8, y: sy(v) + 4, 'text-anchor': 'end', class: 'axis' }, o.yf ? o.yf(v) : v)));
  (o.xt || []).forEach(v => svg.append(
    S('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: Hh - m.b, stroke: 'var(--line)', 'stroke-width': 1, opacity: .5 }),
    S('text', { x: sx(v), y: Hh - m.b + 16, 'text-anchor': 'middle', class: 'axis' }, o.xf ? o.xf(v) : v)));
  if (o.xl) svg.append(S('text', { x: (m.l + W - m.r) / 2, y: Hh - 4, 'text-anchor': 'middle', class: 'axis' }, o.xl));
  if (o.yl) svg.append(S('text', { x: 11, y: (m.t + Hh - m.b) / 2, 'text-anchor': 'middle', class: 'axis', transform: `rotate(-90 11 ${(m.t + Hh - m.b) / 2})` }, o.yl));
  return { svg, sx, sy, m, W, H: Hh };
}
const T = (x, y, txt, cls = 'lbl', anchor = 'start', extra) => S('text', Object.assign({ x, y, class: cls, 'text-anchor': anchor }, extra), txt);
const Line = (x1, y1, x2, y2, stroke = 'var(--ink3)', w = 1.5, extra) => S('line', Object.assign({ x1, y1, x2, y2, stroke, 'stroke-width': w, 'stroke-linecap': 'round' }, extra));
const Circle = (cx, cy, r, fill, extra) => S('circle', Object.assign({ cx, cy, r, fill }, extra));
const Rect = (x, y, w, hh, fill, extra) => S('rect', Object.assign({ x, y, width: w, height: hh, fill }, extra));
const Path = (d, stroke, w = 2, fill = 'none', extra) => S('path', Object.assign({ d, stroke, 'stroke-width': w, fill, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, extra));

/* ───────── The Map ───────── */
lesson({
  id: 'descent', ch: 'map', mins: 6,
  title: "How the pieces fit together",
  lede: "A command goes in at the top of the stack and a motor turns at the bottom. This page lists the layers in between, how often each one runs, and how the two policies fit side by side.",
  build() {
    const all = ['tele', 'nav', 'pick'];
    const stages = [
      { t: 'Input', d: 'Joystick or keyboard, or a spoken goal', use: all },
      { t: 'Whisper', d: 'Speech to text, runs locally, a few hundred ms', use: ['nav', 'pick'] },
      { t: 'Task planner (LLM)', d: 'Goal, world state and skill library in, ordered skills out. Runs once per goal.', use: ['nav', 'pick'] },
      { t: 'Behavior tree', d: 'Runs the plan at 20 Hz and hands each skill to the right place', use: ['nav', 'pick'] },
      { t: 'Nav2', d: 'Makes a path, then a velocity command', use: ['nav'] },
      { t: '/cmd_vel  [vx, vy, vyaw]', d: 'Three numbers in the robot’s body frame. Teleop and Nav2 meet here.', use: ['tele', 'nav'], seam: true },
      { t: 'Locomotion policy', d: 'Command and observation in, leg position targets out', use: ['tele', 'nav'] },
      { t: 'Manipulation policy', d: 'FoundationPose finds the object, IK moves to a pre-grasp pose, the learned policy grasps. Arm position targets out.', use: ['pick'] },
      { t: 'Control layer', d: '', dm: { tele: 'Leg controller: plain PD. Position targets in, torques out, 1 to 2 kHz.', nav: 'Leg controller: plain PD. Position targets in, torques out, 1 to 2 kHz.', pick: 'Arm controller: impedance control with gravity compensation. Position targets in, torques out, 1 to 2 kHz.' }, use: all },
      { t: 'ros2_control → EtherCAT', d: 'Torque goes into a frame, one frame per millisecond', use: all },
      { t: 'Drive: field-oriented control', d: 'Torque to phase currents at 20-30 kHz', use: all },
      { t: 'Motor', d: 'The shaft turns and the encoder reports the angle back up', use: all },
    ];
    const trace = Widget('Trace a command', body => {
      let mode = 'nav', timer = null, at = -1;
      const list = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } }), note = h('p.cap');
      const rows = stages.map(s => {
        const det = h('div', { style: { fontSize: '13px', color: 'var(--ink3)' } }, s.d);
        const r = h('div', { style: { display: 'grid', gridTemplateColumns: '10px minmax(0,1fr)', gap: '12px', alignItems: 'center', padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--line)', transition: 'opacity .2s, background .2s' } },
          h('span', { style: { width: '10px', height: '10px', borderRadius: '50%', background: 'var(--ink3)' } }), h('div', null, h('b', { style: { fontSize: '15px' } }, s.t), det));
        if (s.seam) r.style.borderColor = 'var(--c)';
        return { r, det, s };
      });
      const paint = () => rows.forEach(({ r, det, s }, i) => {
        const skip = !s.use.includes(mode), hit = i === at; if (s.dm) det.textContent = s.dm[mode];
        r.style.opacity = skip ? '.3' : '1'; r.style.background = hit ? 'var(--c-soft)' : 'transparent'; r.firstChild.style.background = hit ? 'var(--c)' : skip ? 'var(--line)' : 'var(--ink3)';
      });
      const send = () => {
        clearInterval(timer); at = -1; const seq = stages.map((s, i) => i).filter(i => stages[i].use.includes(mode));
        let k = 0; timer = setInterval(() => { at = seq[k++]; paint(); if (k >= seq.length) { clearInterval(timer); setTimeout(() => { at = -1; paint(); }, 700); } }, 420);
      };
      onLeave(() => clearInterval(timer));
      const setMode = m => {
        mode = m; at = -1; paint();
        note.innerHTML = m === 'tele' ? '<b>Teleop:</b> the human is the planner. They watch the robot and correct it themselves, so nothing above /cmd_vel is needed. It also works when the autonomy stack is broken.'
          : m === 'nav' ? '<b>Navigate skill:</b> the goal becomes a navigate skill. Nav2 turns it into /cmd_vel, the locomotion policy walks, and the leg controller makes the torques.'
            : '<b>Pick skill:</b> a pick skill goes to the manipulation side instead. The arm policy outputs arm targets and the arm controller makes the torques. Nothing goes through /cmd_vel.';
      };
      body.append(h('div.row', null, Seg([{ v: 'nav', label: 'Goal: go to the table' }, { v: 'pick', label: 'Goal: pick up the cube' }, { v: 'tele', label: 'Teleop: joystick' }], 'nav', setMode), Btn('Send a command', send)), list, note);
      rows.forEach(x => list.append(x.r)); setMode('nav');
    });

    return [
      P('A command reaches the robot in one of two ways. In teleoperation a person drives with a joystick or keyboard. In autonomous operation the person gives a goal in plain language, like “bring me the cube from the table,” and the autonomy stack works out the rest. The two paths look nothing alike at the top. Teleop and Nav2 meet at one ROS topic, /cmd_vel, and below that the locomotion side is the same either way.'),
      H2('How often each layer runs'),
      P('Every layer has its own rate. The bars below count how many times each one runs during a single second. The scale is logarithmic, because the slowest layer runs once per goal and the fastest runs about 30,000 times a second.'),
      Widget('One second of the stack', body => { const l = Ladder(); body.append(h('div.row', null, Btn('Run one second', () => l.run())), l); }),
      P('Going down the list, each layer runs faster and its output gets less abstract. The task planner hands over a list of skills once per goal. The controllers turn position targets into torques 1,000 to 2,000 times a second, and the drive inside each motor closes its current loop 20,000 to 30,000 times a second. Each layer takes what the one above it produced and turns it into something the next one can use.'),
      H2('Two policies side by side'),
      P('There are two learned policies, and they run side by side. The locomotion policy takes /cmd_vel and outputs leg position targets. The manipulation policy handles the arms. A pick or place skill goes to the manipulation side, which finds the object, moves the arm to a pre-grasp pose and runs the learned grasp policy, and arm position targets come out of that. Both policies output joint position targets, and neither talks to the motors directly.'),
      P('Below them sits the control layer, which turns each policy’s position targets into torques: plain PD for the legs, and impedance control with gravity compensation for the arm. Those torques go to ros2_control, then EtherCAT, then the drives.'),
      H2('Teleop versus autonomous'),
      P('With teleop the human is the planner. They watch the robot and correct it with their own eyes and hands, so almost the whole top of the stack gets skipped. A ROS package (teleop_twist_keyboard or teleop_twist_joy) publishes /cmd_vel and the locomotion policy reads it. You use this for testing, demos and collecting data, and it still works when the autonomy stack is broken.'),
      P('The autonomous path starts with the goal as text, and voice goes through Whisper first. A task planner turns the text into a list of skills once, and a behavior tree runs that list at 20 Hz. A navigation skill goes to Nav2, whose local controller publishes /cmd_vel. A pick or place skill goes to the manipulation side. Try each of the three paths below.'),
      trace,
      Trap('The three numbers in /cmd_vel (vx, vy, vyaw) are in the robot’s body frame, so vx means forward from wherever the robot is facing, not north. A goal from the autonomy stack starts out in the world frame and has to be rotated first. The Nav2 lesson shows how. Only the locomotion policy reads /cmd_vel. The arm isn’t driven by velocity commands.'),
      H2('What’s in the rest of the course'),
      UL('<b>Perception</b> covers SLAM, loop closure and how objects end up in the map.', '<b>Planning and navigation</b> covers the task planner, the behavior tree and Nav2.', '<b>Learning to walk</b> covers how the locomotion policy is trained with reinforcement learning.', '<b>Arms and hands</b> covers manipulation methods, starting at inverse kinematics and ending at VLAs.', '<b>Controls</b> covers the layer that turns both policies’ targets into torques.', '<b>Hardware interface</b> covers ros2_control, EtherCAT and the drive.'),
      Check({ q: 'Which layer runs the most times per second?', opts: ['The task planner', 'The behavior tree', 'The policies', 'Field-oriented control inside each drive'], ans: 3, why: 'FOC runs at 20 to 30 kHz. The planner runs once per goal.' }),
      Check({ q: 'What does the locomotion policy know about who produced /cmd_vel?', opts: ['Whether it came from a joystick or from Nav2', 'Nothing. It only receives [vx, vy, vyaw]', 'It uses a different network for each source'], ans: 1, why: 'Teleop and autonomous differ above /cmd_vel and are identical below it.' }),
      Check({ q: 'What sits between the policies and ros2_control?', opts: ['The control layer, which turns position targets into torques', 'The behavior tree', 'The EtherCAT driver'], ans: 0, why: 'Both policies output joint position targets, and the drives need torque.' }),
    ];
  },
  cards: [
    { q: "How do the layers change as you go down the stack?", a: "Each one runs faster and its output is less abstract. The planner runs once per goal, the controllers run at 1-2 kHz, and the drive’s control loop runs at 20-30 kHz." },
    { q: "Where do the teleop and autonomous paths meet?", a: "At /cmd_vel, which carries [vx, vy, vyaw] in the robot’s body frame. Only the locomotion policy reads it. Everything below is identical." },
    { q: "How do the two policies fit together?", a: "They run side by side. The locomotion policy is driven by /cmd_vel and outputs leg targets. The manipulation policy is driven by pick and place skills and outputs arm targets. Both go through the control layer." },
    { q: "What sits between the policies and ros2_control?", a: "The control layer: PD for the legs, impedance control for the arm. It turns position targets into torques." },
    { q: "Typical rates for the behavior tree, the controllers and FOC?", a: "20 Hz, 1-2 kHz, and 20-30 kHz." },
  ],
});
