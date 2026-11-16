'use strict';
/* ═════════ Chapter — Perception ═════════ */

const loopPt = (u, X0, Y0, RW, RH) => { // point on a rectangular loop, u in [0,1]
  const s = (u % 1) * 2 * (RW + RH); let x, y;
  if (s < RW) { x = s; y = 0; } else if (s < RW + RH) { x = RW; y = s - RW; } else if (s < 2 * RW + RH) { x = RW - (s - RW - RH); y = RH; } else { x = 0; y = RH - (s - 2 * RW - RH); }
  return [X0 + x, Y0 + y];
};

/* 1 ─ what SLAM is */
lesson({
  id: 'p-slam', ch: 'see', mins: 7,
  title: "What SLAM is",
  lede: "SLAM works out where a camera is and builds a map of the places it has been, both from the same stream of images.",
  build() {
    const tri = Widget('Triangulating one point from two views', body => {
      let base = 170, L = [330, 70]; const SIG = 3 * Math.PI / 180, svg = svgRoot(660, 260, { 'aria-label': 'Two cameras looking at one landmark' }), out = h('div.readout');
      const draw = () => {
        svg.innerHTML = ''; const Y = 226, A = [330 - base / 2, Y], B = [330 + base / 2, Y];
        const wedge = c => { const a = Math.atan2(L[1] - c[1], L[0] - c[0]), len = Math.hypot(L[0] - c[0], L[1] - c[1]) + 60; return `M${c[0]} ${c[1]} L${c[0] + len * Math.cos(a - SIG)} ${c[1] + len * Math.sin(a - SIG)} L${c[0] + len * Math.cos(a + SIG)} ${c[1] + len * Math.sin(a + SIG)}Z`; };
        svg.append(S('path', { d: wedge(A), fill: 'var(--c)', opacity: .35 }), S('path', { d: wedge(B), fill: 'var(--c)', opacity: .35 }), Line(A[0], A[1], L[0], L[1], 'var(--c)', 1.5), Line(B[0], B[1], L[0], L[1], 'var(--c)', 1.5),
          Rect(A[0] - 11, A[1] - 6, 22, 14, 'var(--ink)', { rx: 3 }), Rect(B[0] - 11, B[1] - 6, 22, 14, 'var(--ink)', { rx: 3 }), T(A[0], A[1] + 26, 'camera position 1', 'axis', 'middle'), T(B[0], B[1] + 26, 'camera position 2', 'axis', 'middle'),
          Circle(L[0], L[1], 6, 'var(--hl)', { stroke: 'var(--ink)', 'stroke-width': 1.5 }), T(L[0] + 12, L[1] - 8, 'feature (drag me)', 'lbl'));
        const Z = Y - L[1]; out.innerHTML = ''; out.append(Chip('baseline', Math.round(base) + ' px'), Chip('depth uncertainty', '≈ ' + fx(Z * Z * SIG / base, 1) + ' px', Z * Z * SIG / base > 30 ? 'bad' : 'ok'), Chip('sideways uncertainty', '≈ ' + fx(Z * SIG, 1) + ' px'));
      };
      drag(svg, p => { L = [clamp(p.x, 70, 590), clamp(p.y, 30, 170)]; draw(); });
      body.append(svg, Slider({ label: 'baseline', min: 20, max: 420, step: 5, value: base, fmt: v => v + ' px', on: v => { base = v; draw(); } }), out,
        h('p.cap', null, 'Each camera measures the direction to the feature with a small angular error (3° here). The darker overlap is where the feature could actually be. Depth uncertainty is roughly distance² × angle error ÷ baseline.')); draw();
    });
    return [
      H2('The chicken-and-egg problem'),
      P('SLAM stands for simultaneous localization and mapping. To know where you are you need a map, and to build a map you need to know where you are. SLAM does both together, and each estimate helps improve the other.'),
      P('In practice a camera moves through a room. From the images alone (an IMU can help), SLAM produces two things at once. One is the camera’s path through 3D space, and the other is a 3D map of the room made of points that mark where physical features are. Neither is known ahead of time, and both come out of a single optimization.'),
      H2('Why it can be solved'),
      P('If you see the same corner of a table from two camera positions, the way that corner shifts between the two images limits two things at once: how the camera moved, and where the corner sits in 3D. Stack up thousands of those constraints across hundreds of features and many camera positions and you can solve for everything together.'),
      tri,
      P('When the two viewpoints are close together the two rays are nearly parallel, so the point where they cross is poorly pinned down. Depth gets better as the baseline grows, which is one reason the camera has to move.'),
      H2('What SLAM doesn’t tell you'),
      P('The map is a sparse cloud of anonymous 3D points. SLAM has no idea that a group of them is a cube. Putting objects in the room is a separate job on top of it, covered in the lesson on objects, frames and TF.'),
      Check({ q: 'What’s the chicken-and-egg problem in SLAM?', opts: ['You need a map to localize, and you need to know your position to build the map', 'You need a camera to build a map, and a map to calibrate the camera', 'You need an IMU to run the camera'], ans: 0, why: 'SLAM estimates the trajectory and the map together, so each one constrains the other.' }),
      Check({ q: 'Why does a very small baseline give poor depth?', opts: ['The rays from the two positions are almost parallel, so where they cross is uncertain', 'The camera can’t see the feature', 'The descriptors stop matching'], ans: 0, why: 'Depth error scales roughly with distance squared times the angular error, divided by the baseline.' }),
    ];
  },
  cards: [
    { q: "What does SLAM stand for and what problem does it solve?", a: "Simultaneous localization and mapping. Localizing needs a map and mapping needs a position, so both get estimated together from the same images." },
    { q: "What two things does visual SLAM produce?", a: "The camera’s trajectory through 3D space, and a 3D map made of points at physical feature locations." },
    { q: "Why can two views constrain both the camera motion and the feature position?", a: "The same corner shifts between the two images by an amount that depends on both where the camera moved and where the corner is in 3D. Many of these constraints get solved together." },
    { q: "Does SLAM know what objects are in the room?", a: "No. The map is anonymous points. Objects come from a separate detector or pose estimator." },
  ],
});

/* 2 ─ ORB-SLAM3 */
lesson({
  id: 'p-orb', ch: 'see', mins: 8,
  title: "ORB-SLAM3",
  lede: "ORB-SLAM3 is the standard feature-based visual SLAM system. It tracks a sparse set of points and splits the work across three threads that run at different rates.",
  build() {
    const th = Widget('The three threads over four seconds', body => {
      const W = 660, Hh = 200, WIN = 4, x0 = 170, wpx = 470, svg = svgRoot(W, Hh, { 'aria-label': 'Timeline of the tracking, local mapping and loop closing threads' }), g = S('g');
      const rows = [['Tracking', '30 Hz, every frame', 52], ['Local mapping', 'once per keyframe', 108], ['Loop closing', 'when a place is revisited', 164]];
      svg.append(...rows.flatMap(([a, b, y]) => [T(10, y - 2, a, 'lbl-b'), T(10, y + 14, b, 'axis'), Line(x0, y + 5, x0 + wpx, y + 5, 'var(--line)', 1)]), g, Line(x0 + wpx, 20, x0 + wpx, 180, 'var(--ink3)', 1.5), T(x0 + wpx, 14, 'now', 'axis', 'middle'));
      let t = 0;
      loop(dt => {
        t += dt; g.innerHTML = ''; const px = s => x0 + (s - (t - WIN)) / WIN * wpx;
        for (let k = Math.ceil((t - WIN) * 30); k <= Math.floor(t * 30); k++) {
          const s = k / 30, kf = k % 12 === 0; g.append(Line(px(s), kf ? 30 : 40, px(s), 62, 'var(--c)', kf ? 2.5 : 1.4));
          if (kf) g.append(Rect(px(s), 100, .26 / WIN * wpx, 18, 'var(--c)', { rx: 2, opacity: .85 }));
        }
        for (let k = Math.ceil((t - WIN - .8) / 5); k <= Math.floor(t / 5 + .01); k++) { const s = 2.2 + k * 5; g.append(Rect(px(s), 156, .8 / WIN * wpx, 18, 'var(--hl)', { rx: 2 })); }
      });
      body.append(svg, h('p.cap', null, 'The taller ticks in the tracking row are keyframes, and each one kicks off a block of local mapping work. Loop closing only fires now and then. Timing is at real speed and the scene scrolls left.'));
    });
    return [
      H2('Feature-based'),
      P('ORB-SLAM3 doesn’t use whole images. It tracks a sparse set of distinctive points.'),
      H3('ORB features'),
      P('In each image the detector picks out a few hundred keypoints: corners, blobs and other high-contrast spots that are easy to recognize again from a different angle. For each keypoint it computes a descriptor, a compact binary fingerprint of the patch of image around it. The same physical corner seen from a slightly different position or rotation gives a similar descriptor, so you can match “this corner in frame 50” with “the same corner in frame 51.” ORB gets used because it’s very fast to compute and match, which is what lets SLAM run in real time.'),
      H2('Three threads'),
      UL('<b>Tracking</b> runs on every frame, about 30 Hz. It extracts ORB features, matches them against the local map and works out where the camera must be right now. It only looks at a small set of nearby map points, which keeps it fast. It also decides whether the frame should become a keyframe, meaning a saved reference frame that gets added to the map.', '<b>Local mapping</b> runs once per keyframe. It triangulates new 3D map points from the keyframe’s features, using the shift between it and nearby keyframes to get depth. Then it runs local bundle adjustment, which jointly refines the recent keyframe poses and the map points they see by minimizing reprojection error (the gap between where a 3D point should show up in an image and where it was actually observed). It runs behind tracking so it never blocks it.', '<b>Loop closing</b> is rare and event-driven. It watches for the robot coming back to somewhere it has been, and fixes the accumulated drift when it does. The next lesson covers it.'),
      th,
      H2('The Atlas'),
      P('The 3 in ORB-SLAM3 is about staying robust when tracking gets lost, for example when the camera is covered, moves suddenly or looks at a blank wall. Rather than failing, the system starts a new map. If it later recognizes a place that connects the new map to an old one, it merges them. The collection of maps is called the Atlas.'),
      Real('If tracking keeps getting lost, the scene probably has too few features (blank walls) or the camera is moving fast enough to blur. Both come straight from how the tracking thread works.'),
      Check({ q: 'Which thread runs at camera framerate?', opts: ['Tracking', 'Local mapping', 'Loop closing'], ans: 0, why: 'Tracking runs on every frame, about 30 Hz. Local mapping runs per keyframe, and loop closing only fires occasionally.' }),
      Check({ q: 'What does local bundle adjustment minimize?', opts: ['Reprojection error, the gap between where a 3D point should appear in an image and where it was observed', 'The number of keyframes', 'Descriptor length'], ans: 0, why: 'It jointly refines recent keyframe poses and the map points they see.' }),
    ];
  },
  cards: [
    { q: "What’s an ORB feature and what’s a descriptor?", a: "A keypoint (a corner or blob) that’s easy to recognize again, plus a descriptor, which is a compact binary fingerprint of the patch around it. It’s cheap to compute and match." },
    { q: "What are ORB-SLAM3’s three threads?", a: "<b>Tracking</b> (every frame, ~30 Hz), <b>local mapping</b> (per keyframe, triangulation and local bundle adjustment), and <b>loop closing</b> (event-driven, corrects drift)." },
    { q: "What’s a keyframe?", a: "A saved reference frame that tracking adds to the map when a frame is informative enough." },
    { q: "What’s bundle adjustment?", a: "Jointly refining keyframe poses and 3D map points by minimizing reprojection error." },
    { q: "What does the Atlas add in ORB-SLAM3?", a: "When tracking is lost it starts a new map instead of failing, and merges maps later if it recognizes a place that connects them." },
  ],
});

/* 3 ─ drift and loop closure */
lesson({
  id: 'p-loop', ch: 'see', mins: 8,
  title: "Drift and loop closure",
  lede: "Small tracking errors build up over time. Loop closure removes them by noticing that the robot is back somewhere it has been.",
  build() {
    const dr = Widget('Walking a loop', body => {
      let D = 60, closed = false; const W = 660, Hh = 290, X0 = 170, Y0 = 78, RW = 320, RH = 170, N = 120, DIR = [.8, -.6], svg = svgRoot(W, Hh, { 'aria-label': 'A walked loop with and without loop closure' }), out = h('div.readout');
      const draw = () => {
        svg.innerHTML = ''; const tr = [], es = []; let worst = 0;
        for (let i = 0; i <= N; i++) { const p = loopPt(i / N, X0, Y0, RW, RH), d = Math.pow(i / N, 1.3) - (closed ? i / N : 0), e = [p[0] + DIR[0] * D * d, p[1] + DIR[1] * D * d]; tr.push(p); es.push(e); worst = Math.max(worst, Math.hypot(e[0] - p[0], e[1] - p[1])); }
        svg.append(Path(path(tr), 'var(--line)', 2, 'none', { 'stroke-dasharray': '4 5' }), Path(path(es), 'var(--c)', 3), Circle(tr[0][0], tr[0][1], 6, 'var(--ink)'), T(tr[0][0] - 10, tr[0][1] - 10, 'start', 'axis', 'end'));
        const e = es[N]; svg.append(Circle(e[0], e[1], 6, 'var(--c)'));
        if (!closed && D > 2) svg.append(Line(tr[0][0], tr[0][1], e[0], e[1], 'var(--hl)', 2, { 'stroke-dasharray': '3 3' }), T(e[0] > 340 ? e[0] - 12 : e[0] + 12, Math.max(18, e[1] - 12), 'same spot, but the estimate disagrees', 'lbl', e[0] > 340 ? 'end' : 'start', { style: 'fill:var(--hl)' }));
        svg.append(T(W - 12, Hh - 10, 'dashed = true path, solid = estimate', 'axis', 'end'));
        out.innerHTML = ''; out.append(Chip('start-to-end gap', closed ? '0 px' : Math.round(D) + ' px', closed ? 'ok' : 'bad'), Chip('largest error along the path', Math.round(worst) + ' px', closed ? 'ok' : 'bad'));
      };
      body.append(svg, h('div.row', null, Seg([{ v: false, label: 'Without loop closure' }, { v: true, label: 'With loop closure' }], false, v => { closed = v; draw(); })), Slider({ label: 'drift over the lap', min: 0, max: 90, step: 1, value: D, fmt: v => v + ' px', on: v => { D = v; draw(); } }), out,
        h('p.cap', null, 'The estimate finishes away from where it started even though the robot walked back to the same spot. With loop closure the accumulated error gets spread across the whole loop. Here it’s spread linearly as a rough stand-in. Real pose graph optimization solves for every keyframe pose at once.')); draw();
    });
    return [
      H2('Drift'),
      P('Tracking makes a small error on every frame. From one frame to the next it’s tiny, but the errors add up. After five minutes of walking around a building the estimated path can be off by meters, and the map bends and warps because thousands of small errors have piled up. Pure frame-to-frame tracking, called visual odometry, drifts without limit.'),
      H2('Loop closure'),
      P('If the robot returns to a place it has already been and recognizes it, that gives you a strong constraint. If the map says the two visits are two meters apart, then about two meters of drift have built up and can be taken out.'),
      dr,
      H3('Recognizing a place you’ve been'),
      P('ORB-SLAM3 has a vocabulary of visual “words,” which are clusters of common descriptor patterns learned offline from lots of images. Any image can be summarized as a histogram of which words it contains, the way a document can be summarized by its word counts. To detect a loop, the system checks whether the current frame’s histogram closely matches the histogram of any old keyframe. That’s a fast database lookup rather than a pixel comparison, and it’s called bag of words. A strong match becomes a loop candidate.'),
      H3('Verifying and correcting'),
      P('A match can be a coincidence, like two hallways that look alike, so the system checks geometrically that the actual features line up. If they do, it computes the drift, meaning the difference between where the map thinks the robot is and where the loop says it is. Then it runs pose graph optimization. The trajectory is treated as a graph of keyframes connected by constraints, the loop constraint gets added, and the accumulated error is spread out across the whole loop. After a big loop closure it may also run a full global bundle adjustment.'),
      Real('If the map drifts and never corrects itself, loop closure probably isn’t triggering. That points to bag-of-words recognition, for instance a scene the vocabulary matches badly, or a route that never revisits anything.'),
      Check({ q: 'How does ORB-SLAM3 decide the robot has been somewhere before?', opts: ['It compares bag-of-words histograms of the current frame and old keyframes, then verifies geometrically', 'It compares every pixel of the current image against every old image', 'It uses GPS'], ans: 0, why: 'Histogram matching is a fast lookup, and the geometric check rejects look-alike places.' }),
      Check({ q: 'What does pose graph optimization do after a loop is confirmed?', opts: ['Adds the loop constraint and spreads the accumulated error across the loop', 'Deletes the old keyframes', 'Restarts tracking'], ans: 0, why: 'The map snaps back into consistency, and a warped hallway straightens out.' }),
    ];
  },
  cards: [
    { q: "What is drift and why does it happen?", a: "Each frame’s tracking has a tiny error and the errors accumulate. Frame-to-frame tracking (visual odometry) drifts without limit." },
    { q: "What does loop closure do?", a: "It recognizes that the robot has returned to a known place, measures the accumulated drift, and spreads that error through the whole trajectory." },
    { q: "How are revisited places recognized?", a: "<b>Bag of words</b>. Images get summarized as histograms of learned visual words and matched against old keyframes by a fast lookup, then verified geometrically." },
    { q: "What’s pose graph optimization?", a: "Treat the keyframes as a graph with constraints between them, add the loop-closure constraint, and spread the accumulated error across the loop." },
    { q: "The map drifts and never corrects. What’s the likely cause?", a: "Loop closure isn’t triggering, so bag-of-words recognition isn’t finding matches." },
  ],
});

/* 4 ─ objects, frames, TF */
lesson({
  id: 'p-frames', ch: 'see', mins: 9,
  title: "Objects, frames and TF",
  lede: "SLAM doesn’t detect objects. A detector does, and combining the two puts them in the map. The ROS TF tree then decides who publishes which part.",
  build() {
    const comp = Widget('Composing SLAM and the detector', body => {
      let rx = 190, ry = 70, hd = 40; const O = [470, 190], W = 660, Hh = 270, svg = svgRoot(W, Hh, { 'aria-label': 'Top view of a robot with a camera and a cube fixed in the map' }), out = h('div.readout'), Y = y => Hh - 20 - y;
      const draw = () => {
        svg.innerHTML = ''; const th = hd * Math.PI / 180, c = Math.cos(th), s = Math.sin(th), dx = O[0] - rx, dy = O[1] - ry, fwd = dx * c + dy * s, left = -dx * s + dy * c, inView = fwd > 0 && Math.abs(Math.atan2(left, fwd)) < .52;
        const fov = a => [rx + 460 * Math.cos(th + a), ry + 460 * Math.sin(th + a)];
        const f1 = fov(-.52), f2 = fov(.52);
        svg.append(Rect(4, 4, W - 8, Hh - 8, 'none', { stroke: 'var(--line)', rx: 10 }), S('path', { d: `M${rx} ${Y(ry)} L${f1[0]} ${Y(f1[1])} L${f2[0]} ${Y(f2[1])}Z`, fill: 'var(--c)', opacity: inView ? .28 : .12 }),
          S('g', { transform: `translate(${rx} ${Y(ry)}) rotate(${-hd})` }, S('polygon', { points: '16,0 -12,-11 -12,11', fill: 'var(--c)' })), T(rx, Y(ry) + 30, 'robot camera', 'axis', 'middle'),
          Rect(O[0] - 11, Y(O[1]) - 11, 22, 22, inView ? 'var(--hl)' : 'var(--ink3)', { rx: 3 }), T(O[0], Y(O[1]) - 18, 'cube (fixed in the map)', 'axis', 'middle'), T(14, 24, 'map frame, top view', 'axis'));
        const bx = rx + fwd * c - left * s, by = ry + fwd * s + left * c;
        out.innerHTML = ''; out.append(Chip('map → camera (SLAM)', '(' + fx(rx / 100) + ', ' + fx(ry / 100) + ') m, ' + hd + '°', 'c'), Chip('camera → object (detector)', inView ? fx(fwd / 100) + ' m ahead, ' + fx(left / 100) + ' m left' : 'not in view'), Chip('map → object (composed)', '(' + fx(bx / 100) + ', ' + fx(by / 100) + ') m', 'ok'));
      };
      body.append(svg, Slider({ label: 'robot x', min: 60, max: 560, step: 5, value: rx, fmt: v => fx(v / 100) + ' m', on: v => { rx = v; draw(); } }), Slider({ label: 'robot y', min: 30, max: 210, step: 5, value: ry, fmt: v => fx(v / 100) + ' m', on: v => { ry = v; draw(); } }), Slider({ label: 'heading', min: -180, max: 180, step: 5, value: hd, fmt: v => v + '°', on: v => { hd = v; draw(); } }), out,
        h('p.cap', null, 'The detector’s answer changes whenever the robot moves. Once it’s combined with the camera pose from SLAM, the object’s position in the map stays put.')); draw();
    });

    const tf = Widget('Two writers, two edges', body => {
      let tc = 6; const holder = h('div'), out = h('div.readout');
      const draw = () => {
        const p = Plot(660, 230, { x: [0, 10], y: [-.2, 5.6], xt: [0, 2, 4, 6, 8, 10], yt: [0, 2, 4], xl: 'time (s)', yl: 'position along x (m)', m: { l: 52 } });
        const drift = t => .06 * t, corr = t => t >= tc ? -drift(tc) : 0, odom = t => .5 * t + drift(t), inMap = t => odom(t) + corr(t), N = 200, ts = Array.from({ length: N + 1 }, (_, i) => i / N * 10);
        p.svg.append(Path(path(ts.map(t => [p.sx(t), p.sy(.5 * t)])), 'var(--ink3)', 1.5, 'none', { 'stroke-dasharray': '4 4' }), Path(path(ts.map(t => [p.sx(t), p.sy(odom(t))])), 'var(--ink2)', 2.5), Path(path(ts.map(t => [p.sx(t), p.sy(inMap(t))])), 'var(--c)', 3));
        p.svg.append(Line(p.sx(tc), p.m.t, p.sx(tc), 230 - p.m.b, 'var(--hl)', 1.5, { 'stroke-dasharray': '3 3' }), T(p.sx(tc) + 6, p.m.t + 12, 'loop closure', 'lbl', 'start', { style: 'fill:var(--hl)' }),
          T(p.m.l + 8, p.m.t + 14, 'grey: odom → base_link (smooth, drifts)', 'lbl', 'start', { style: 'fill:var(--ink2)' }), T(p.m.l + 8, p.m.t + 30, 'colored: base_link in the map (odom + map → odom)', 'lbl', 'start', { style: 'fill:var(--c)' }), T(p.m.l + 8, p.m.t + 46, 'dashed: true position', 'lbl', 'start', { style: 'fill:var(--ink3)' }));
        holder.innerHTML = ''; holder.append(p.svg);
        out.innerHTML = ''; out.append(Chip('odom → base_link jump', '0.00 m', 'ok'), Chip('map → odom jump', fx(-drift(tc)) + ' m', 'c'));
      };
      body.append(holder, Slider({ label: 'loop closure at', min: 2, max: 9, step: .1, value: tc, fmt: v => fx(v, 1) + ' s', on: v => { tc = v; draw(); } }), out,
        h('p.cap', null, 'Local motion in the odom frame keeps running smoothly. SLAM’s correction shows up as a single step in map → odom, and the robot’s position in the map lands back near the true path.')); draw();
    });

    return [
      H2('From SLAM to objects'),
      P('SLAM gives you the robot’s position and a sparse point cloud. Objects come from a separate step. An object detector or pose estimator like FoundationPose runs on the camera image and reports that there’s a cube and what its 6-DOF pose is relative to the camera. SLAM reports the camera’s pose in the map. Compose the two transforms and you get the cube’s pose in the map frame.'),
      Eq('map → object  =  (map → camera)  ×  (camera → object)\n                     <span class="k">SLAM</span>            <span class="k">detector</span>'),
      comp,
      P('Since the result is anchored to the map instead of the camera, the object stays where it is as the robot moves. That’s how the behavior tree’s blackboard ends up with something like “cube at (0.5, 0.2, 0.8) in the map.” Do it for many objects and you get a semantic map, or scene graph, which is the room as labeled objects with poses. It’s what a task planner needs to reason about “navigate to the table, the cube is on it.”'),
      H2('The TF tree'),
      Eq('map → odom          <span class="k">SLAM</span> owns this edge\nodom → base_link    <span class="k">state estimation</span> owns this edge\nbase_link → camera  static, from the mounting'),
      P('SLAM doesn’t update at one fast, uniform rate. Tracking runs at camera rate, local mapping runs per keyframe and loop closing only fires occasionally. What SLAM publishes is one TF edge, map → odom, updated when it has a correction.'),
      P('State estimation (the fast IEKF and odometry) publishes odom → base_link at a high rate. That edge is smooth and accurate locally but drifts over time. The robot’s pose in the map is the composition of the two edges: fast smooth local motion plus a slow global correction.'),
      tf,
      P('The split is on purpose. Because odom → base_link is smooth and never jumps, a balance controller never sees the robot teleport. The slower map → odom gives global consistency and fixes drift, which is what navigation needs. When a loop closure fires, map → odom jumps and the local motion in the odom frame isn’t disturbed. This is the standard ROS convention (REP 105). There are two writers, each owning a different edge, so they never conflict.'),
      Check({ q: 'Which TF edge does SLAM publish?', opts: ['map → odom', 'odom → base_link', 'base_link → camera'], ans: 0, why: 'SLAM owns map → odom (the drift correction). State estimation owns odom → base_link. base_link → camera is static, from the mounting.' }),
      Check({ q: 'Why is odom → base_link kept separate from map → odom?', opts: ['Control needs motion that never jumps, while navigation needs global consistency that can jump when drift is corrected', 'ROS doesn’t allow more than one TF edge', 'To save bandwidth'], ans: 0, why: 'A loop closure only changes map → odom, so the balance controller running off odom never sees a discontinuity.' }),
      Check({ q: 'How does the cube’s position in the map get computed?', opts: ['Compose the camera’s pose in the map (SLAM) with the cube’s pose in the camera frame (detector)', 'The detector reports it directly in the map', 'SLAM labels the cube in its point cloud'], ans: 0, why: 'The detector only knows about the camera image. SLAM supplies the camera-to-map transform.' }),
    ];
  },
  cards: [
    { q: "How does an object end up in the map frame?", a: "Compose <b>map → camera</b> (from SLAM) with <b>camera → object</b> (from the detector). The result stays fixed as the robot moves." },
    { q: "What’s a semantic map or scene graph?", a: "The room as labeled objects with poses instead of anonymous points. It’s what the task planner reasons over." },
    { q: "Who owns which TF edge?", a: "<b>map → odom</b> belongs to SLAM (slow drift correction, jumps at loop closure). <b>odom → base_link</b> belongs to state estimation (fast, smooth, drifts). <b>base_link → camera</b> is static from the mounting." },
    { q: "Why split map → odom from odom → base_link?", a: "Control needs smooth motion that never jumps, and navigation needs global consistency. A loop closure only changes map → odom. This is REP 105." },
  ],
});

/* 5 ─ what a team builds */
lesson({
  id: 'p-team', ch: 'see', mins: 8,
  title: "What a perception team builds",
  lede: "Perception leans on libraries more than most layers. A team mostly uses the algorithms as they are and spends its time on calibration, integration and tuning.",
  build() {
    const swap = Widget('Stub and real perception', body => {
      let m = 'stub'; const box = (t, d, strong) => h('div', { style: { border: '1px solid ' + (strong ? 'var(--c)' : 'var(--line)'), background: strong ? 'var(--c-soft)' : 'transparent', borderRadius: '8px', padding: '10px 14px' } }, h('b', { style: { fontSize: '15px' } }, t), d ? h('div', { style: { fontSize: '13.5px', color: 'var(--ink2)' } }, d) : null);
      const st = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } });
      const draw = () => {
        st.innerHTML = ''; st.append(box('Perception node', m === 'stub' ? 'Stub: publishes /object_poses with a hardcoded object position and a fake map → odom (identity, no drift).' : 'Real: camera or bag file → ORB-SLAM3 + FoundationPose → real /object_poses and a real map → odom.', true),
          h('div.mono', { style: { textAlign: 'center', color: 'var(--c)', padding: '4px', borderBlock: '1px dashed var(--c)' } }, 'same topic · same message type · same TF edge'), box('Behavior tree and task planner', 'Read /object_poses and the map frame. They can’t tell which node is publishing.', false));
      };
      body.append(Seg([{ v: 'stub', label: 'Stub' }, { v: 'real', label: 'Real perception' }], 'stub', v => { m = v; draw(); }), st); draw();
    });
    const tbl = (head, rows) => h('div.scroll-x', null, h('table.tt', { style: { whiteSpace: 'normal' } }, h('thead', null, h('tr', null, head.map(x => h('th', { style: { textAlign: 'left' } }, x)))), h('tbody', null, rows.map(r => h('tr', null, r.map((c, i) => h('td', { style: { textAlign: 'left', whiteSpace: 'normal', verticalAlign: 'top', fontWeight: i ? 500 : 600 }, html: c })))))));
    return [
      H2('Use as-is'),
      tbl(['Component', 'Notes'], [['ORB-SLAM3', 'A mature research codebase. You compile it, configure it for your camera and use its output. There are ROS2 wrappers that publish its pose and map to topics. You don’t implement SLAM yourself.'], ['FoundationPose', 'Pretrained. You run it.'], ['Camera driver', 'realsense-ros or something similar. It publishes images to topics.'], ['TF / tf2', 'A library.'], ['Object detectors', 'YOLO, SAM, Grounding-DINO. Pretrained models you run.']]),
      H2('Build yourself'),
      tbl(['Work', 'Notes'], [['Camera calibration', 'Intrinsics (lens parameters) and extrinsics (where the camera is mounted relative to the robot base). It’s careful one-time work, and if it’s wrong then everything downstream is wrong.'], ['ROS integration', 'Wiring ORB-SLAM3 and FoundationPose output into your TF tree and a custom /object_poses topic, in your frames and at the right rates. Most of the team’s effort goes here.'], ['Semantic layer', 'Composing the SLAM pose and detections into objects in the map, and building the scene graph the planner uses.'], ['Tuning', 'Feature count, keyframe thresholds and the other ORB-SLAM3 parameters, for your camera and your environment.']]),
      P('The team needs to understand the internals well enough to debug them, but not well enough to reimplement them.'),
      tbl(['Symptom', 'Likely cause'], [['Tracking is lost constantly', 'A scene with few features (blank walls) or motion blur from moving too fast. That’s the tracking thread.'], ['The map drifts and never corrects', 'Loop closure isn’t triggering. That’s bag-of-words recognition.']]),
      H2('Developing without the robot'),
      P('A perception team needs a camera (a RealSense or similar) and a laptop. They can run the camera driver and ORB-SLAM3, walk the camera around a room, and watch it build the map and localize in RViz, with no robot and no other teams involved. A recorded bag file of a walk through the lab can stand in for the live camera for as long as needed. RViz shows the point cloud growing, the trajectory and the detected objects.'),
      P('In the scaffold, the perception stub publishes /object_poses with a hardcoded position and a fake map → odom transform. That lets the task planner and behavior tree get developed against “there’s a cube at a known spot.” The real node gets developed in parallel, and when it’s ready it replaces the stub. It publishes the same topic, message type and TF edge, so nothing downstream changes.'),
      swap,
      Note('Perception has a simple input (a camera feed) and simple outputs (one TF edge and one object-poses topic), which makes it one of the easiest parts of the stack to develop in parallel.'),
      Check({ q: 'Which of these does a perception team implement itself?', opts: ['Camera calibration, ROS integration, the semantic layer and tuning', 'The SLAM optimization and loop closure math', 'The object detection networks'], ans: 0, why: 'The algorithmic core comes from libraries. The team calibrates, integrates, composes objects into the map and tunes.' }),
      Check({ q: 'What lets a real perception node replace the stub without changing anything downstream?', opts: ['It publishes the same topic, message type and TF edge', 'It runs on the same computer', 'It uses the same camera'], ans: 0, why: 'Downstream nodes only depend on the interface.' }),
    ];
  },
  cards: [
    { q: "What does a perception team use as-is?", a: "ORB-SLAM3, FoundationPose, the camera driver, TF/tf2, and pretrained detectors like YOLO, SAM and Grounding-DINO." },
    { q: "What does a perception team build?", a: "<b>Camera calibration</b> (intrinsics and extrinsics), <b>ROS integration</b> (the TF tree and /object_poses), the <b>semantic layer</b> (objects in the map, the scene graph) and <b>tuning</b>." },
    { q: "How well should the team understand ORB-SLAM3?", a: "Well enough to debug it. Constant lost tracking points to a feature-poor scene or motion blur, and drift that never corrects points to loop closure not triggering." },
    { q: "How can perception be developed without the robot?", a: "With a camera and a laptop, or a recorded bag file, visualized in RViz. The real node then replaces the stub, which publishes the same topic, message type and TF edge." },
  ],
});
