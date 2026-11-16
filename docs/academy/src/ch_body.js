'use strict';
/* ═════════ Chapters — Planning and navigation, Hardware interface ═════════ */

/* 1 ─ Task planner */
lesson({
  id: 'b-planner', ch: 'plan', mins: 8,
  title: "Task planner",
  lede: "The planner turns a goal in plain language into an ordered list of skills the robot actually has. It runs once per goal.",
  build() {
    const SK = {
      navigate_to: { pre: [], post: ['at:X'], arg: true },
      pick: { pre: ['at:table', 'gripper:empty'], post: ['holding:cube', 'gripper:full'], arg: true },
      place: { pre: ['holding:cube'], post: ['gripper:empty', 'placed:cube'] },
    };
    const wid = Widget('Validate a plan against the skill library', body => {
      let plan = ['navigate_to(table)', 'pick(cube)', 'navigate_to(user)', 'place()']; const list = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } }), out = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } }), lib = h('div.readout');
      const all = ['navigate_to(table)', 'navigate_to(user)', 'pick(cube)', 'place()', 'fly_to(moon)', 'open_door(kitchen)'];
      const validate = () => {
        let world = new Set(['gripper:empty']), at = 'start'; const res = [];
        plan.forEach(step => {
          const nm = step.split('(')[0], arg = (step.match(/\((.*)\)/) || [])[1];
          if (!SK[nm]) { res.push({ step, ok: false, msg: 'not in the skill library, so the LLM made up a capability the robot doesn’t have' }); return; }
          const s = SK[nm], need = s.pre.filter(p => !(p === 'at:table' ? at === 'table' : world.has(p)));
          if (need.length) { res.push({ step, ok: false, msg: 'precondition not satisfied: ' + need.join(', ') }); return; }
          if (nm === 'navigate_to') at = arg; if (nm === 'pick') { world.delete('gripper:empty'); world.add('holding:cube'); world.add('gripper:full'); } if (nm === 'place') { world.delete('holding:cube'); world.delete('gripper:full'); world.add('gripper:empty'); world.add('placed:cube'); }
          res.push({ step, ok: true, msg: 'ok · world now: ' + [...world].concat(['at:' + at]).join(', ') });
        });
        return res;
      };
      const draw = () => {
        list.innerHTML = ''; plan.forEach((s, i) => list.append(h('div.row', { style: { gap: '8px' } }, h('span.chip', null, (i + 1)), h('code', { style: { flex: 1 } }, s), Btn('remove', () => { plan.splice(i, 1); draw(); }, 'ghost sm'))));
        const r = validate(); out.innerHTML = ''; r.forEach(x => out.append(h('div', { style: { fontSize: '14px', padding: '6px 10px', borderRadius: '8px', background: x.ok ? 'var(--ok-soft)' : 'var(--bad-soft)' } }, h('b', null, (x.ok ? '✓ ' : '✗ ') + x.step), h('div', { style: { color: 'var(--ink2)' } }, x.msg))));
        lib.innerHTML = ''; lib.append(Chip('plan', r.every(x => x.ok) && plan.length ? 'valid' : 'rejected: retry / replan', r.every(x => x.ok) && plan.length ? 'ok' : 'bad'));
      };
      const adders = h('div.row', null, ...all.map(a => Btn('+ ' + a, () => { plan.push(a); draw(); }, 'ghost sm')));
      body.append(h('p.cap', null, 'Goal: “bring me the cube from the table.” The list below is the LLM’s output. Remove steps, add steps, or add a skill that does not exist, and see what the validator reports.'), h('div.readout', null, Chip('skill library', 'navigate_to · pick · place'), Chip('pick needs', 'at table, gripper empty'), Chip('place needs', 'holding cube')), list, adders, out, lib, h('div.row', null, Btn('Reset to the LLM’s plan', () => { plan = ['navigate_to(table)', 'pick(cube)', 'navigate_to(user)', 'place()']; draw(); }, 'ghost'))); draw();
    });

    return [
      H2('From input to a string'),
      P('Voice goes through Whisper, which converts speech to text locally in a few hundred milliseconds. Typed text skips that step. Either way you end up with a string for the goal, published on /user_intent, and that wakes the planner up.'),
      H2('What the planner needs'),
      UL('<b>A skill library.</b> A YAML file you write once that defines every capability. Each skill lists its preconditions (what has to be true to start) and postconditions (what will be true when it finishes). The planner can only use skills that are in this file.', '<b>The current world state</b>, put together from live topics: robot pose (from the state estimator), object poses (from perception), gripper state and battery.', '<b>The language-model call</b>: the goal, world state and skill library go in, and a JSON list of skill calls comes out.'),
      P('Then you validate the JSON. Every skill has to exist and the preconditions have to chain from one step to the next. If the output is malformed you retry, and once it passes you publish it. For “bring me the cube” it looks like navigate_to(table) → pick(cube) → navigate_to(user) → place(). After that the planner goes back to sleep until a new goal shows up or something fails badly enough to need a replan.'),
      wid,
      Note('The skill library keeps the LLM to capabilities the robot really has, and the validator checks that each step’s preconditions are met by the postconditions of the steps before it.'),
      Check({ q: 'How often does the task planner run?', opts: ['Continuously at 20 Hz', 'Once per goal, then it sleeps unless a failure needs a replan', 'Once per second'], ans: 1, why: 'It isn’t a loop.' }),
      Check({ q: 'Why does the validator check preconditions and not only that the skill names exist?', opts: ['A plan can be made of real skills in an impossible order, like picking before navigating to the table', 'Because the LLM can’t spell', 'To speed up planning'], ans: 0, why: 'Chaining postconditions to preconditions is a cheap symbolic check.' }),
    ];
  },
  cards: [
    { q: "How often does the task planner run, and what does it produce?", a: "Once per goal (then it sleeps until a new goal or a failure that needs a replan). It produces an ordered list of skills as JSON." },
    { q: "What are the planner’s three inputs?", a: "The goal, the world state (robot pose, object poses, gripper, battery), and the skill library (YAML with preconditions and postconditions)." },
    { q: "What does validating the plan check?", a: "That every skill exists in the library and that preconditions chain from one step’s postconditions to the next. Retry if it’s malformed." },
    { q: "Where does voice input go?", a: "Through Whisper (local speech to text, a few hundred ms), then a string on /user_intent wakes the planner." },
  ],
});

/* 2 ─ Behavior tree */
lesson({
  id: 'b-bt', ch: 'plan', mins: 9,
  title: "Behavior tree",
  lede: "The behavior tree carries out the plan. It re-checks safety on every tick, so it can abort a bad motion within one tick.",
  build() {
    const wid = Widget('Tick the tree', body => {
      let battery = true, upright = true, fault = false, navFail = false, tick = 0, step = 0, prog = 0, retries = 0, status = 'RUNNING', timer = null, running = true, log = [];
      const skills = ['navigate_to(table)', 'pick(cube)', 'navigate_to(user)', 'place()'];
      const view = h('div'), logEl = h('div.eq', { style: { minHeight: '92px', whiteSpace: 'pre-wrap' } }), out = h('div.readout');
      const node = (t, s, indent, extra) => h('div', { style: { marginLeft: indent * 22 + 'px', display: 'flex', gap: '8px', alignItems: 'center', padding: '5px 10px', borderRadius: '8px', border: '1px solid var(--line)', background: s === 'FAILURE' ? 'var(--bad-soft)' : s === 'RUNNING' ? 'var(--c-soft)' : s === 'SUCCESS' ? 'var(--ok-soft)' : 'var(--surface)', fontSize: '14px' } }, h('b', null, t), s ? h('span.chip', { style: { marginLeft: 'auto' } }, s) : null, extra || null);
      const say = m => { log.unshift('tick ' + tick + ': ' + m); log = log.slice(0, 5); };
      const step1 = () => {
        tick++;
        const guards = [['battery ok', battery], ['upright', upright], ['no faults', !fault]], bad = guards.find(g => !g[1]);
        if (bad) { status = 'ABORTED'; say('guard “' + bad[0] + '” failed → ReactiveSequence aborts, action cancelled this tick'); render(); return; }
        if (step >= skills.length) { status = 'SUCCESS'; say('plan complete'); running = false; render(); return; }
        status = 'RUNNING';
        if (navFail && skills[step].startsWith('navigate')) { retries++; if (retries > 3) { status = 'REPLAN'; say('retries used up, so the failure goes up to the planner for a full replan'); running = false; } else say('nav failed → recovery: clear costmap and retry (' + retries + '/3)'); render(); return; }
        prog += 12 + (step % 2) * 6; say('ticking ' + skills[step] + ' (RUNNING, ' + Math.min(100, prog) + '%)');
        if (prog >= 100) { say(skills[step] + ' SUCCESS'); step++; prog = 0; retries = 0; }
        render();
      };
      const render = () => {
        view.innerHTML = '';
        view.append(node('Root · ReactiveSequence', status === 'ABORTED' ? 'FAILURE' : status === 'SUCCESS' ? 'SUCCESS' : 'RUNNING', 0, null));
        [['battery ok', battery], ['upright', upright], ['no faults', !fault]].forEach(g => view.append(node('Condition: ' + g[0], g[1] ? 'SUCCESS' : 'FAILURE', 1)));
        skills.forEach((s, i) => view.append(node('Action: ' + s, i < step ? 'SUCCESS' : i === step && status !== 'ABORTED' && status !== 'SUCCESS' ? (status === 'REPLAN' ? 'FAILURE' : 'RUNNING') : '', 1)));
        logEl.textContent = log.join('\n') || 'press Start';
        out.innerHTML = ''; out.append(Chip('tick', tick, 'c'), Chip('rate', '20 Hz = every 50 ms'), Chip('status', status, status === 'ABORTED' || status === 'REPLAN' ? 'bad' : status === 'SUCCESS' ? 'ok' : ''));
      };
      const start = () => { clearInterval(timer); timer = setInterval(() => { if (status === 'ABORTED' || !running) return; step1(); }, 300); };
      onLeave(() => clearInterval(timer));
      const tgl = (id, lab, get, set) => h('label', { style: { display: 'inline-flex', gap: '6px', alignItems: 'center', fontSize: '14px' } }, h('input', { type: 'checkbox', id, checked: get() ? '' : null, on: { change(e) { set(e.target.checked); render(); } }, style: { accentColor: 'var(--c)' } }), lab);
      body.append(h('div.row', null, tgl('bt-bat', 'Battery OK', () => battery, v => { battery = v; }), tgl('bt-up', 'Upright', () => upright, v => { upright = v; }), tgl('bt-fault', 'Inject fault', () => fault, v => { fault = v; }), tgl('bt-nav', 'Navigation keeps failing', () => navFail, v => { navFail = v; })),
        h('div.row', null, Btn('Start / restart', () => { tick = 0; step = 0; prog = 0; retries = 0; status = 'RUNNING'; running = true; log = []; start(); render(); }), Btn('Pause', () => clearInterval(timer), 'ghost')), view, logEl, out, h('p.cap', null, 'Ticks are slowed down (the real rate is every 50 ms). Untick “Battery OK” while it runs and the guard at the top fails on the next tick, cancelling the running action. Turn on “Navigation keeps failing” to see recovery escalate to a replan.')); render();
    });

    return [
      H2('How it works'),
      UL('<b>Ticking.</b> The whole tree gets evaluated from the root down 20 times a second, and every node returns SUCCESS, FAILURE or RUNNING.', '<b>ReactiveSequence.</b> Safety guards (battery, upright, faults) sit at the top and get re-checked on every tick. If something goes wrong partway through an action, the tree can abort within one tick instead of finishing a motion that no longer makes sense.', '<b>The blackboard.</b> Shared memory that ROS subscribers keep updated: robot pose at 500 Hz, object poses at 30 Hz, gripper state, battery. Condition nodes read it in microseconds and never have to make a slow call.', '<b>Action nodes.</b> These are the long-running skills. Each one is a client of a ROS action server that works asynchronously, sends feedback and can be cancelled. When the tree ticks “navigate to table” it sends a goal and gets RUNNING back while the robot moves.'),
      P('Navigation skills go to Nav2, which ends in /cmd_vel and the locomotion policy. Pick and place skills go to the manipulation side and its arm policy. The two policies run side by side, each fed by its own kind of skill.'),
      P('Each skill is also wrapped in recovery. If it fails, a recovery routine runs (clear the costmap and retry for navigation, re-detect the object and retry for grasping), and only when the retries run out does the failure go up to the planner for a full replan.'),
      wid,
      Trap('Don’t confuse the task planner with the behavior tree. The planner reasons once and produces the plan. The tree runs all the time, executes that plan, watches safety and handles failures.'),
      Order({ q: 'A navigation skill keeps failing. Put the escalation in order.', items: ['The skill fails', 'A recovery routine runs: clear the costmap and retry', 'The retries run out', 'The failure goes up to the task planner', 'The planner replans and publishes a new skill sequence'], why: 'The slow, expensive layer only gets used after the fast, cheap one runs out of options.' }),
      Check({ q: 'The battery drops below the safe threshold in the middle of a pick. How soon does the tree react?', opts: ['After the pick finishes', 'Within one tick, since the guards are re-checked every tick', 'When the planner next runs'], ans: 1, why: 'That’s what a ReactiveSequence does.' }),
      Check({ q: 'What’s the blackboard?', opts: ['A log of failures', 'Shared memory kept fresh by ROS subscribers, so conditions can read pose, objects, gripper and battery immediately', 'The skill library'], ans: 1, why: 'Subscribers update it at their own rates.' }),
    ];
  },
  cards: [
    { q: "What does ReactiveSequence do?", a: "It re-checks the safety guards at the top of the tree on every tick, so a bad action gets aborted within one tick." },
    { q: "What’s the behavior tree’s tick rate and what do nodes return?", a: "20 Hz. Each node returns SUCCESS, FAILURE or RUNNING." },
    { q: "What’s the blackboard?", a: "Shared memory kept fresh by ROS subscribers (pose 500 Hz, objects 30 Hz, gripper, battery), so conditions read state in microseconds." },
    { q: "How do action nodes work?", a: "Each is a client of a ROS action server. It’s asynchronous, sends feedback and can be cancelled. It returns RUNNING while the skill executes." },
    { q: "What’s the failure escalation order?", a: "The skill fails, recovery and retry, retries run out, the failure goes up to the planner, and it does a full replan." },
  ],
});

/* 3 ─ Nav2 & /cmd_vel */
lesson({
  id: 'b-cmdvel', ch: 'plan', mins: 7,
  title: "Nav2 and /cmd_vel",
  lede: "When a skill means “go somewhere,” Nav2 converts it into the same three numbers a joystick would produce.",
  build() {
    const rot = Widget('World frame → body frame', body => {
      let hd = 60, wx = 0, wy = 1, wrong = false; const holder = h('div'), out = h('div.readout');
      const draw = () => {
        const th = hd * Math.PI / 180, vx = Math.cos(th) * wx + Math.sin(th) * wy, vy = -Math.sin(th) * wx + Math.cos(th) * wy;
        const svg = svgRoot(660, 250, { 'aria-label': 'Robot heading and commanded velocity in world and body frames' }), cx = 200, cy = 130, s = 80;
        svg.append(Line(cx - 120, cy, cx + 120, cy, 'var(--line)', 1), Line(cx, cy + 105, cx, cy - 105, 'var(--line)', 1), T(cx + 124, cy + 4, 'x (east)', 'axis'), T(cx + 6, cy - 108, 'y (north)', 'axis'));
        const g = S('g', { transform: `translate(${cx} ${cy}) rotate(${-hd})` });
        g.append(Rect(-16, -26, 32, 52, 'var(--c-soft)', { stroke: 'var(--c)', 'stroke-width': 2, rx: 8 }), S('polygon', { points: '0,-40 -9,-26 9,-26', fill: 'var(--c)' }));
        svg.append(g, Line(cx, cy, cx + wx * s, cy - wy * s, 'var(--hl)', 4), Circle(cx + wx * s, cy - wy * s, 5, 'var(--hl)'), T(cx + wx * s + 8, cy - wy * s, 'command in world', 'lbl', 'start', { style: 'fill:var(--hl)' }));
        const bx = wrong ? wx : vx, by = wrong ? wy : vy; svg.append(T(380, 46, 'what the policy should receive (body frame)', 'lbl-b'));
        [['vx  (forward)', bx], ['vy  (left)', by], ['vyaw', 0]].forEach(([n, v], i) => { const y = 78 + i * 46; svg.append(T(380, y, n, 'axis'), Line(470, y + 8, 630, y + 8, 'var(--line)', 1), Line(550, y, 550, y + 16, 'var(--ink3)', 1), Rect(v >= 0 ? 550 : 550 + v * 80, y + 2, Math.max(2, Math.abs(v) * 80), 12, wrong ? 'var(--bad)' : 'var(--c)', { rx: 3 }), T(636, y + 13, fx(v), 'lbl', 'end')); });
        holder.innerHTML = ''; holder.append(svg);
        out.innerHTML = ''; out.append(Chip('heading', hd + '° (from east)', ''), Chip('command', wrong ? 'sent UNROTATED' : 'rotated into body frame', wrong ? 'bad' : 'ok'), wrong ? Chip('result', 'robot walks the wrong way', 'bad') : Chip('result', 'goes where you meant', 'ok'));
      };
      body.append(holder, Slider({ label: 'robot heading', min: 0, max: 360, step: 5, value: hd, fmt: v => v + '°', on: v => { hd = v; draw(); } }), Slider({ label: 'goal, east/west', min: -1, max: 1, step: .1, value: wx, fmt: v => fx(v, 1), on: v => { wx = v; draw(); } }), Slider({ label: 'goal, north/south', min: -1, max: 1, step: .1, value: wy, fmt: v => fx(v, 1), on: v => { wy = v; draw(); } }),
        h('div.row', null, Seg([{ v: false, label: 'Rotate into body frame' }, { v: true, label: 'Forget to rotate' }], false, v => { wrong = v; draw(); })), out, h('p.cap', null, 'The world-frame command stays fixed while the robot turns. The body-frame numbers change so that the same physical direction is forward or left depending on the robot’s heading. At heading 90°, “go north” is the robot’s forward.')); draw();
    });

    return [
      H2('Nav2'),
      P('When the behavior tree’s active skill is “navigate to the table,” it calls Nav2. Inside Nav2 a global planner runs about once a second and finds a path through a costmap built from the SLAM map, and a local controller runs about 20 times a second and turns that path into a velocity command.'),
      P('The local controller publishes /cmd_vel, which is the same [vx, vy, vyaw] a joystick gives you in the teleop path. The locomotion policy subscribes to it and has no way of knowing whether a thumb or Nav2’s path follower produced the numbers. Above /cmd_vel teleop and autonomous look completely different, and below it they’re identical. Nothing in the high-level stack goes past that topic, and only the locomotion policy reads it. The arm isn’t driven by velocity commands.'),
      H2('Body frame'),
      P('vx is forward relative to where the robot faces, vy is sideways and vyaw is turn rate. A command like “go north” starts in the world frame, so it has to be rotated into the body frame using the robot’s current heading.'),
      rot,
      Trap('If you skip the rotation the robot walks off in the wrong direction. When it faces east, “go north” is the robot’s left, so vy = 1 and vx = 0.'),
      P('Nav2’s velocity limits are also set to match what the locomotion policy saw in training, since the policy only knows the commands from its training distribution.'),
      Check({ q: 'The robot faces east and Nav2 wants it to move north at 1 m/s (world frame). What are vx and vy in the body frame?', opts: ['vx = 1, vy = 0', 'vx = 0, vy = 1', 'vx = 0, vy = −1'], ans: 1, why: 'North is to the robot’s left when it faces east.' }),
      Check({ q: 'What does Nav2’s local controller output, and how often?', opts: ['/cmd_vel [vx, vy, vyaw] at about 20 Hz', 'Joint torques at 1 kHz', 'A path at about 1 Hz'], ans: 0, why: 'The global planner makes the path at about 1 Hz.' }),
    ];
  },
  cards: [
    { q: "What are Nav2’s two levels and their rates?", a: "The <b>global planner</b>, about 1 Hz, finds a path through a costmap from the SLAM map. The <b>local controller</b>, about 20 Hz, turns the path into /cmd_vel." },
    { q: "What is /cmd_vel and in which frame?", a: "Three numbers, <b>[vx, vy, vyaw]</b>, in the robot’s <b>body frame</b>: forward, sideways and turn rate." },
    { q: "What’s the frame mistake to avoid?", a: "A world-frame command has to be rotated into the body frame using the robot’s heading. If it isn’t, the robot walks the wrong way." },
    { q: "Why do Nav2’s velocity limits have to match the policy?", a: "The policy only knows velocity commands from its training distribution." },
  ],
});

/* 4 ─ ros2_control & EtherCAT */
lesson({
  id: 'b-ros2c', ch: 'hw', mins: 8,
  title: "ros2_control",
  lede: "The control layer produces torque commands, and ros2_control is the framework that runs those controllers and connects them to the drives.",
  build() {
    const loopW = Widget('The 1 kHz loop', body => {
      let phase = -1, timer = null, cycles = 0; const stages = [['1 · READ', 'state up from hardware: actual position, velocity, torque', 'state interfaces'], ['2 · UPDATE', 'active controllers compute new commands (PD gains per joint)', 'controllers'], ['3 · WRITE', 'commands down to hardware: target torque', 'command interfaces']];
      const row = h('div.grid2', { style: { gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,190px),1fr))' } }), out = h('div.readout');
      const draw = () => { row.innerHTML = ''; stages.forEach((s, i) => row.append(h('div', { style: { border: '1.5px solid ' + (i === phase ? 'var(--c)' : 'var(--line)'), background: i === phase ? 'var(--c-soft)' : 'var(--surface)', borderRadius: '12px', padding: '12px 14px', transition: 'background .15s' } }, h('b', { style: { fontFamily: 'var(--mono)', fontSize: '13px', color: 'var(--c)' } }, s[0]), h('div', { style: { fontSize: '14.5px', margin: '4px 0' } }, s[1]), h('div', { style: { fontSize: '12.5px', color: 'var(--ink3)' } }, s[2])))); out.innerHTML = ''; out.append(Chip('cycles', cycles, 'c'), Chip('budget per cycle', '1 ms')); };
      const run = () => { clearInterval(timer); timer = setInterval(() => { phase = (phase + 1) % 3; if (phase === 0) cycles++; draw(); }, 420); };
      onLeave(() => clearInterval(timer));
      body.append(row, out, h('div.row', null, Btn('Run the loop (slowed down)', run), Btn('Stop', () => clearInterval(timer), 'ghost')), h('p.cap', null, 'At real speed these three stages run in this order a thousand times a second.')); draw();
    });

    return [
      H2('Where the torques come from'),
      P('The two policies’ position targets have already been converted into torques by the control layer, which the previous chapter covers. Those controllers are ros2_control controllers, so ros2_control is what runs them and passes their output down to the hardware. The result for every joint is a torque in newton-meters.'),
      H2('ros2_control'),
      P('ros2_control is ROS’s standard framework for talking to hardware. It keeps controllers (the code that computes commands) separate from the hardware (the physical robot), and the two talk through standard interfaces:'),
      UL('<b>Command interfaces</b> are what you write down to the hardware, like target torque.', '<b>State interfaces</b> are what you read back up: actual position, velocity and torque.'),
      P('The controller manager runs a real-time loop at 1 kHz, and every millisecond it does three things in a fixed order. The controllers in that loop are the control layer’s. The leg PD controller and the arm impedance controller (CRISP, for example) are both ros2_control controllers.'),
      loopW,
      P('There are two things to configure, and this is where YAML shows up:'),
      UL('A <code>&lt;ros2_control&gt;</code> block in the robot’s URDF that lists, for each joint, the command interfaces it exposes (effort or torque) and the state interfaces it reports (position, velocity, effort). It maps a name like left_hip_pitch to a hardware entity the framework can command, and it’s mostly copy and rename for each joint.', 'A separate controllers YAML that lists which controllers are active, the 1 kHz update rate and the per-joint gains, which are high on knees and low on wrists and tuned by hand.'),
      Trap('ros2_control can’t speak EtherCAT or CAN on its own. It only knows “read state, write command” through a plugin. The EtherCAT driver is that plugin. It isn’t a separate node downstream of ros2_control. ros2_control loads it as its hardware interface and calls its read() and write() directly inside the 1 kHz loop.'),
      Order({ q: 'Put the ros2_control cycle in order.', items: ['Read the current state up from the hardware', 'Update the active controllers so they compute new commands', 'Write those commands down to the hardware'], why: 'Read, update, write, a thousand times a second.' }),
      Check({ q: 'What does a motor drive take as input?', opts: ['A joint angle', 'Electric current, which produces torque', 'A velocity'], ans: 1, why: 'The control layer converts position targets into torque, and the drive converts torque into current.' }),
      Check({ q: 'Where does the EtherCAT driver sit relative to ros2_control?', opts: ['A separate node that ros2_control publishes to', 'A hardware-interface plugin that ros2_control loads and calls read() and write() on', 'Inside the behavior tree'], ans: 1, why: 'That keeps the real-time loop a single tight cycle.' }),
      Check({ q: 'How does the control layer relate to ros2_control?', opts: ['The leg and arm controllers are ros2_control controllers, run by its controller manager', 'ros2_control replaces the control layer', 'They run on separate computers'], ans: 0, why: 'ros2_control hosts the controllers and connects their torque output to the hardware.' }),
    ];
  },
  cards: [
    { q: "What’s the controller manager’s loop?", a: "<b>Read, update, write</b> at <b>1 kHz</b>. Read state from the hardware, let the controllers compute commands, write the commands to the hardware." },
    { q: "What’s the difference between command and state interfaces?", a: "<b>Command</b> interfaces are written down to the hardware (target torque). <b>State</b> interfaces are read up from it (position, velocity, effort)." },
    { q: "What two things do you configure for ros2_control?", a: "The URDF <b>&lt;ros2_control&gt;</b> block (per-joint command and state interfaces), and a <b>controllers YAML</b> (active controllers, 1 kHz rate, per-joint gains)." },
    { q: "How does ros2_control talk to EtherCAT?", a: "It doesn’t. The EtherCAT driver is a <b>hardware-interface plugin</b>, and ros2_control calls its read() and write() inside its loop." },
    { q: "How does the control layer relate to ros2_control?", a: "The leg PD controller and the arm impedance controller are ros2_control controllers. ros2_control runs them and passes their torques to the hardware." },
  ],
});

/* 5 ─ EtherCAT & PDOs */
lesson({
  id: 'b-ethercat', ch: 'hw', mins: 10,
  title: "EtherCAT and PDOs",
  lede: "EtherCAT is a real-time industrial network where one frame passes through every motor drive on a daisy chain. It’s fast because each drive reads and writes its part in hardware as the frame goes by.",
  build() {
    const chain = Widget('The frame’s trip down the chain', body => {
      let n = 25; const holder = h('div'), out = h('div.readout'); let pos = -1, timer = null;
      const draw = () => {
        const W = 660, show = Math.min(n, 12), gap = (W - 80) / (show + 1), svg = svgRoot(W, 130, { 'aria-label': 'EtherCAT daisy chain' });
        svg.append(Rect(10, 45, 56, 40, 'var(--ink)', { rx: 8 }), T(38, 70, 'master', 'lbl', 'middle', { style: 'fill:var(--bg)' }), Line(66, 65, W - 10, 65, 'var(--ink3)', 3));
        for (let i = 0; i < show; i++) { const x = 80 + gap * (i + 1); svg.append(Rect(x - 16, 45, 32, 40, pos === i ? 'var(--c)' : 'var(--c-soft)', { rx: 8, stroke: 'var(--c)', 'stroke-width': 2 }), T(x, 70, i + 1 === show && n > show ? '…N' : String(i + 1), 'lbl', 'middle', { style: pos === i ? 'fill:var(--c-ink)' : '' })); }
        if (pos >= 0) svg.append(Circle(80 + gap * (pos + 1), 65, 6, 'var(--hl)', { stroke: 'var(--ink)', 'stroke-width': 1.5 }));
        svg.append(T(W / 2, 22, 'one frame out through every drive, and back', 'lbl-b', 'middle'), T(W / 2, 112, 'each drive reads its slice and writes its reply as the frame passes: ~300 ns, in hardware, no CPU', 'axis', 'middle'));
        holder.innerHTML = ''; holder.append(svg);
        const rt = n * 3.4; out.innerHTML = ''; out.append(Chip('drives', n, 'c'), Chip('round trip (scaled from 25 drives ≈ 85 µs)', '≈ ' + Math.round(rt) + ' µs', rt < 1000 ? 'ok' : 'bad'), Chip('budget at 1 kHz', '1000 µs'), Chip('headroom', Math.round((1 - rt / 1000) * 100) + '%', 'hl'));
      };
      const run = () => { clearInterval(timer); pos = -1; const show = Math.min(n, 12); timer = setInterval(() => { pos++; if (pos >= show) { clearInterval(timer); pos = -1; } draw(); }, 230); };
      onLeave(() => clearInterval(timer));
      body.append(holder, Slider({ label: 'drives on chain', min: 6, max: 100, step: 1, value: n, fmt: v => v, on: v => { n = v; draw(); } }), out, h('div.row', null, Btn('Send a frame', run)), h('p.cap', null, 'The scaling is a rough linear model from the 25-drive figure (about 85 µs) in the notes. It is not a datasheet number.')); draw();
    });



    return [
      P('The ethercat_driver_ros2 plugin takes the torque command that ros2_control wants to write, packs it into an EtherCAT frame, sends it down the cable and reads the response. Physically it’s one cable that goes from the computer through drive one, then drive two, and on to drive N and back. One frame makes the whole trip.'),
      chain,
      H2('PDOs'),
      P('Each cycle’s data is organized into Process Data Objects, going in two directions:'),
      UL('<b>RxPDO</b> is received by the drive. It carries what you’re commanding: a control word (enable, fault reset) and the target torque.', '<b>TxPDO</b> is transmitted by the drive. It carries what the drive reports back: a status word plus actual position, velocity and torque.'),
      P('CiA 402 is the common standard for servo drives, and it’s why drives from different vendors work the same way. It fixes the addresses: target torque at <code>0x6071</code>, actual position at <code>0x6064</code>, control word at <code>0x6040</code> and status word at <code>0x6041</code>. It also defines a standard state machine that each drive steps through, from powered but disabled up to operational.'),
      Match({ q: 'Match the CiA 402 address to what’s stored there.', rows: [{ l: '0x6071', r: 'Target torque' }, { l: '0x6064', r: 'Actual position' }, { l: '0x6040', r: 'Control word' }, { l: '0x6041', r: 'Status word' }], why: 'Any compliant drive uses the same addresses.' }),
      H2('The mapping YAML'),
      P('This part is tedious and you only do it once. For each drive on the chain you write a slave configuration YAML. It starts with the vendor ID and product ID (from the datasheet or the manufacturer’s ESI file, which is an XML description), then the RxPDO mapping (control word and target torque) and the TxPDO mapping (status word and actual position, velocity and torque). Each entry needs the object’s index, sub-index and bit size, all of which are in the datasheet’s object dictionary. Then come the distributed clock settings, usually a 1 ms cycle.'),
      P('So when a new actuator shows up, you open its datasheet or ESI file, find the object dictionary, look up the index and size for each quantity you need, and copy those into the YAML.'),
      Trap('One wrong index or bit size and that drive gets garbage and either sits there dead or faults. It’s the most common problem the first time you bring up EtherCAT.'),
      P('<b>Distributed clocks</b> keep the drives in step. Every drive has its own clock crystal and they drift apart. The master measures the propagation delay to each slave and keeps adjusting them so that all the drives run their control at the same instant, to within a microsecond. Without that, some joints would lag others by hundreds of microseconds, and the controllers’ assumption that every joint moves together would quietly fail.'),
      Check({ q: 'How can one frame per millisecond serve 25 drives?', opts: ['The frame is very large', 'Each drive reads and writes its slice in hardware as the frame passes, in about 300 ns with no processor involved', 'The drives take turns'], ans: 1, why: 'The round trip for 25 drives is under 100 µs.' }),
      Check({ q: 'Where do the index, sub-index and bit size for the mapping YAML come from?', opts: ['The drive’s datasheet or ESI object dictionary', 'ros2_control', 'You guess and test'], ans: 0, why: 'Only the datasheet says where each quantity lives in the drive’s memory.' }),
    ];
  },
  cards: [
    { q: "Why is one EtherCAT frame per ms enough for 25 drives?", a: "One cable runs through every drive. Each drive reads its slice and writes its reply in hardware, in about 300 ns, as the frame passes. The round trip for 25 drives is under 100 µs." },
    { q: "What’s the difference between RxPDO and TxPDO?", a: "<b>RxPDO</b> is received by the drive (control word and target torque). <b>TxPDO</b> is transmitted by the drive (status word and actual position, velocity, torque)." },
    { q: "What are the key CiA 402 addresses?", a: "<b>0x6071</b> target torque, <b>0x6064</b> actual position, <b>0x6040</b> control word, <b>0x6041</b> status word." },
    { q: "What goes in the slave mapping YAML and where does it come from?", a: "Per drive: vendor ID, product ID, RxPDO and TxPDO entries (index, sub-index, bit size) and DC settings. It comes from the datasheet or ESI object dictionary. One wrong value and the drive is dead or faults." },
    { q: "What do distributed clocks do?", a: "The master measures propagation delay to each slave and adjusts their clocks so all drives act at the same instant, within 1 µs. Otherwise the crystals drift and joints lag each other." },
  ],
});

/* 6 ─ FOC & CAN */
lesson({
  id: 'b-foc', ch: 'hw', mins: 9,
  title: "The drive: FOC and CAN",
  lede: "Each drive receives a target torque and turns it into current thousands of times a second. The encoder reading goes straight back up the chain.",
  build() {
    const foc = Widget('Field-oriented control, one cycle', body => {
      const stages = [['Measure phase currents', 'three-phase currents from the motor windings'], ['Transform into the rotor’s frame', 'Clarke, then Park: a frame that spins with the rotor, so the torque-producing current looks like a constant'], ['PI loops on the torque current', 'target current = commanded torque ÷ motor torque constant (from the datasheet)'], ['Transform back, generate PWM (SVPWM)', 'voltages that switch the transistors feeding the windings'], ['Motor produces torque', 'sign of the command sets the direction: positive one way, negative the other']];
      let cur = 0, timer = null; const list = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } });
      const draw = () => { list.innerHTML = ''; stages.forEach((s, i) => list.append(h('div', { style: { padding: '9px 14px', borderRadius: '10px', border: '1.5px solid ' + (i === cur ? 'var(--c)' : 'var(--line)'), background: i === cur ? 'var(--c-soft)' : 'var(--surface)', display: 'grid', gridTemplateColumns: '24px 1fr', gap: '10px' } }, h('b', { style: { fontFamily: 'var(--mono)', color: 'var(--c)' } }, i + 1), h('div', null, h('b', null, s[0]), h('div', { style: { fontSize: '13.5px', color: 'var(--ink2)' } }, s[1]))))); };
      onLeave(() => clearInterval(timer));
      body.append(list, h('div.row', null, Btn('Step through slowly', () => { clearInterval(timer); timer = setInterval(() => { cur = (cur + 1) % stages.length; draw(); }, 700); }), Btn('Stop', () => clearInterval(timer), 'ghost'), Chip('rate', '20–30 kHz', 'c')), h('p.cap', null, 'The math is left out here. This loop runs much faster than anything above it, and its job is to produce exactly the commanded torque.')); draw();
    });

    const topo = Widget('Native EtherCAT or CAN bridge?', body => {
      let m = 'native'; const box = (t, d, strong) => h('div', { style: { border: '1.5px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'var(--surface)', borderRadius: '10px', padding: '9px 12px' } }, h('b', { style: { fontSize: '14.5px' } }, t), d ? h('div', { style: { fontSize: '13px', color: 'var(--ink3)' } }, d) : null);
      const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } }), out = h('div.readout');
      const arrow = t => h('div.mono', { style: { textAlign: 'center', color: 'var(--ink3)' } }, t);
      const draw = () => {
        st.innerHTML = ''; st.append(box('ros2_control + EtherCAT master', 'writes RxPDO torque, reads TxPDO state', false), arrow('▼ EtherCAT'));
        if (m === 'native') st.append(box('Drives 1…N on the chain', 'They speak EtherCAT and CiA 402 directly and run FOC internally, so there’s no firmware for you to write, only the mapping YAML.', true));
        else st.append(box('Bridge board: MCU (your firmware)', 'It’s an EtherCAT slave as far as ros2_control can tell and a CAN master as far as the motors can tell. Every cycle it reads the EtherCAT command, sends it over CAN, reads the CAN replies and writes them into the TxPDO, like a real-time protocol translator.', true), arrow('▼ CAN (simpler, slower)'), box('CAN motors (Unitree, T-Motor, ODrive-style)', 'cheaper actuators that speak CAN', false));
        out.innerHTML = ''; out.append(m === 'native' ? Chip('you write', 'mapping YAML only', 'ok') : Chip('you write', 'embedded C firmware + YAML', 'hl'), Chip('board', m === 'native' ? 'none' : 'EtherCAT slave controller + CAN transceiver + MCU'));
      };
      body.append(Seg([{ v: 'native', label: 'Native EtherCAT drives' }, { v: 'bridge', label: 'CAN motors via MCU bridge' }], 'native', v => { m = v; draw(); }), st, out, h('p.cap', null, 'Which one you have depends on the actuators you buy.')); draw();
    });

    return [
      H2('Field-oriented control'),
      P('Inside each drive a small microcontroller runs field-oriented control (FOC) at 20 to 30 kHz, way faster than anything above it. Its job is to produce exactly the commanded torque by controlling the current in the motor’s three phases. The math is left out here.'),
      foc,
      H2('Closing the loop'),
      P('The motor’s encoder reads the actual joint angle every cycle, and that reading goes all the way back up:'),
      Eq('encoder → <span class="k">TxPDO</span> → EtherCAT frame → driver → ros2_control <span class="k">state interfaces</span> → state estimator → the policies’ <span class="k">next observation</span>'),
      P('It’s the same data the state estimator fuses to get the base pose and joint state the policies see. The controllers also read the joint state from the state interfaces every cycle, so this is where the low-level side feeds back into the control layer and the policies.'),
      H2('CAN and the MCU bridge'),
      P('Everything so far assumed native EtherCAT drives, which you plug onto the chain and which speak EtherCAT and CiA 402 directly. There’s no firmware to write, just the mapping YAML. But a lot of cheaper actuators (Unitree’s, T-Motor’s, ODrive-style ones) speak CAN instead. CAN is a simpler, slower automotive bus and you can’t daisy-chain it onto the EtherCAT cable, so a microcontroller sits in between.'),
      P('On one side the MCU acts as an EtherCAT slave. To ros2_control it looks like any other drive, receiving RxPDO torque commands and reporting TxPDO state. On the other side it’s a CAN master. It re-sends those torque commands as CAN messages to the motors, reads their replies and folds them into its TxPDO. Every cycle it runs firmware (embedded C) that reads the EtherCAT command, sends it over CAN, reads the CAN replies and writes them back to EtherCAT. A bridge board is usually an EtherCAT slave controller chip, a CAN transceiver and an MCU.'),
      topo,
      P('Which one you end up with depends on which actuators you bought.'),
      Check({ q: 'Where does the encoder’s angle reading go?', opts: ['Nowhere, it’s only used inside the drive', 'Up the chain: TxPDO, EtherCAT, driver, ros2_control state interface, state estimator, the policies’ next observation', 'Straight to the behavior tree'], ans: 1, why: 'The loop closes. The policies asked for angles, and the measured result comes back for the next command.' }),
      Check({ q: 'Which topology needs you to write firmware?', opts: ['Native EtherCAT drives', 'CAN motors behind an EtherCAT-to-CAN bridge MCU', 'Both'], ans: 1, why: 'Native drives only need the mapping YAML.' }),
    ];
  },
  cards: [
    { q: "What does FOC do and how fast does it run?", a: "Field-oriented control in each drive at <b>20-30 kHz</b>. It measures phase currents, transforms them to the rotor’s frame, runs PI loops on the torque current, and transforms back to PWM for the transistors. Target current = torque ÷ torque constant." },
    { q: "How does the joint angle get back up the stack?", a: "Encoder, TxPDO, EtherCAT, driver, ros2_control state interface, then the controllers’ joint state and, through the state estimator, the policies’ next observation." },
    { q: "Native EtherCAT versus a CAN bridge: what do you write?", a: "<b>Native</b> needs the mapping YAML only. <b>CAN motors</b> need an MCU (EtherCAT slave on one side, CAN master on the other) running your firmware, plus the YAML." },
    { q: "Why would you need the CAN bridge?", a: "Cheaper actuators (Unitree, T-Motor, ODrive) speak CAN, which can’t be daisy-chained on the EtherCAT cable." },
  ],
});

/* 7 ─ Capstone */
lesson({
  id: 'b-capstone', ch: 'hw', mins: 8,
  title: "The full chain",
  lede: "A review page with nothing new on it. Put the stack in order from memory, then read the whole thing back as one paragraph.",
  build() {
    return [
      Order({ q: 'Autonomous path. Put the stack in order, from the person’s sentence down to the motor.', items: ['Whisper turns speech into a text string, which wakes the task planner', 'The task planner reasons once and publishes an ordered skill list', 'The behavior tree ticks at 20 Hz, guarding safety and dispatching each skill', 'For a navigation skill, Nav2 makes a path and then /cmd_vel', 'The locomotion policy turns /cmd_vel and its observation into leg position targets, while the manipulation policy turns a pick or place skill into arm position targets', 'The control layer turns each set of targets into torques: PD for the legs, impedance control for the arm', 'ros2_control’s 1 kHz loop hands the torques to the EtherCAT driver, one frame per millisecond', 'Each drive’s FOC (20-30 kHz) turns the torque into phase currents and the motor spins'], why: 'The encoder’s reading then goes back up the same path to become the next observation and the controllers’ joint state.' }),
      Match({ q: 'Match each layer to its rate.', rows: [{ l: 'Task planner', r: 'once per goal' }, { l: 'Behavior tree', r: '20 Hz' }, { l: 'Nav2 global planner', r: '~1 Hz' }, { l: 'Locomotion and manipulation policies', r: '50-100 Hz' }, { l: 'Leg and arm controllers', r: '1-2 kHz' }, { l: 'ros2_control loop', r: '1 kHz' }, { l: 'FOC inside the drive', r: '20-30 kHz' }], why: 'Each layer runs slower and is more abstract than the one below it.' }),
      Check({ q: 'What do the two policies output?', opts: ['Joint torques', 'Joint position targets, with the control layer producing torques below them', 'Velocities in the world frame'], ans: 1, why: 'Neither policy knows what’s below the position-target interface.' }),
      Check({ q: 'What’s identical between teleop and autonomous?', opts: ['The task planner and behavior tree', 'Everything below /cmd_vel on the locomotion side', 'Only the motors'], ans: 1, why: 'The locomotion policy only sees [vx, vy, vyaw].' }),
      Check({ q: 'Why do the arm and the legs have different controllers?', opts: ['The arm is in free space and needs gravity and Coriolis compensation, while the leg policy already learned the dynamics in simulation', 'The arm uses a different motor brand', 'The legs run at a lower rate'], ans: 0, why: 'That’s the whole reason the control layer has two controllers side by side.' }),
      H2('The whole thing as one paragraph'),
      P('A person either uses a joystick, in which case they’re the planner and the command goes straight to /cmd_vel, or gives a goal. Whisper turns speech into text and the text wakes the task planner, which reasons once. It combines the skill library and the live world state through a language model to get an ordered list of skills. A behavior tree runs that list at 20 Hz, checking safety on every tick, reading the world from the blackboard, calling an action server for each skill and recovering when something fails. A navigation skill goes to Nav2, which makes a path and then /cmd_vel, the same three numbers the joystick would have sent. A pick or place skill goes to the manipulation side instead. The locomotion policy takes /cmd_vel and outputs leg position targets, and the manipulation policy outputs arm position targets, with the two running side by side. The control layer turns those targets into torques, using a plain PD controller for the legs and an impedance controller with gravity and Coriolis compensation for the arm, both at 1 to 2 kHz. ros2_control’s 1 kHz read-update-write loop passes the torques to its EtherCAT hardware plugin, which packs each one into a CiA 402 target torque object at 0x6071 and sends a single frame down a daisy chain that reaches every drive in under a hundred microseconds. Inside each drive, field-oriented control at around thirty kilohertz turns the torque into phase currents and the motor turns. The encoder reads the actual angle and sends it back up into the policies’ next observation and the controllers’ joint state.'),
      h('div.row', null, h('a.btn', { href: '#/review' }, 'Go to the review deck'), h('a.btn.ghost', { href: '#/' }, 'Back to the start')),
    ];
  },
  cards: [
    { q: "Put the whole stack in order, top to bottom.", a: "Input, Whisper, task planner (once), behavior tree (20 Hz), then Nav2 and <b>/cmd_vel</b> to the locomotion policy, or a pick skill to the manipulation policy. Then the control layer (leg PD, arm impedance), ros2_control 1 kHz, EtherCAT frame, drive FOC 20-30 kHz, motor. The encoder reading goes back up." },
    { q: "What’s identical between teleop and autonomous?", a: "Everything below /cmd_vel on the locomotion side." },
    { q: "Why are there two controllers in the control layer?", a: "The arm is in free space and needs gravity and Coriolis compensation. The leg policy learned the dynamics in simulation, so a plain PD is enough." },
  ],
});
