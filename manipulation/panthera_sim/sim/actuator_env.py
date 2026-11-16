"""Onshape actuator subassembly scene with reproducible, small pose variation."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from sim.panthera_env import PantheraSim

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'actuator'
TABLE_Z = .05
GEAR_RACK_HEIGHT = .015
NOMINAL = {
    # Keep the far-side radial approach inside the wrist's joint limits.
    'small_carrier': (.36, .00),
    'large_carrier': (.30, -.10),
    'gear_1': (.31, .09), 'gear_2': (.31, .18),
    'gear_3': (.36, .12), 'gear_4': (.36, -.14),
}
PIN_NOMINAL = {'pin_1': (.24, -.14), 'pin_2': (.24, -.06),
               'pin_3': (.24, .06), 'pin_4': (.24, .14)}
PIN_GEARS = {'pin_1': 'gear_4', 'pin_2': 'gear_3',
             'pin_3': 'gear_1', 'pin_4': 'gear_2'}
COLORS = {'small_carrier': '.32 .43 .55 1', 'large_carrier': '.72 .79 .88 1',
          'gear_1': '.95 .53 .16 1', 'gear_2': '.94 .67 .24 1',
          'gear_3': '.88 .40 .14 1', 'gear_4': '.99 .76 .32 1'}
COLORS.update({name: '.35 .67 .90 1' for name in PIN_NOMINAL})


def vec(values):
    return ' '.join(f'{v:.9g}' for v in values)


def build_scene(with_pins=False):
    manifest = json.loads((ASSETS/'manifest.json').read_text())
    root = ET.parse(HERE/'panthera/panthera.xml').getroot()
    root.set('model', 'actuator_carrier_and_planets')
    # All paths remain relative to this generated scene, so it is portable.
    root.find('compiler').set('meshdir', '.')
    for m in root.findall('./asset/mesh'):
        m.set('file', '../panthera/'+m.get('file'))
    root.find('option').set('timestep', '.001')
    root.find('option').set('iterations', '80')
    ET.SubElement(root, 'size', memory='128M')
    ET.SubElement(root, 'statistic', center='.36 0 .12', extent='.55')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='1000', azimuth='135', elevation='-35')
    ET.SubElement(visual, 'headlight', ambient='.4 .4 .4', diffuse='.65 .65 .65')
    asset, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(world, 'light', pos='.2 -.3 1.2', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .01', rgba='.13 .17 .22 1', contype='8', conaffinity='5')
    ET.SubElement(world, 'geom', name='table_top', type='box', pos='.39 0 .025', size='.23 .24 .025', rgba='.30 .35 .40 1', friction='.8 .02 .001', contype='8', conaffinity='5')
    ET.SubElement(world, 'camera', name='overview', pos='.80 -.68 .68', xyaxes='.84 .54 0 -.30 .47 .83', fovy='43')
    ET.SubElement(world, 'camera', name='assembly_close', pos='.52 -.16 .20', xyaxes='.8 .6 0 -.36 .48 .8', fovy='35')
    ET.SubElement(world, 'camera', name='overhead', pos='.40 0 .62', xyaxes='1 0 0 0 1 0', fovy='44')
    stations = NOMINAL | PIN_NOMINAL if with_pins else NOMINAL
    for name, xy in stations.items():
        part = manifest['parts'][name]
        rack_height = GEAR_RACK_HEIGHT if name.startswith('gear_') else 0.
        if rack_height:
            ET.SubElement(world, 'geom', name=name+'_rack', type='cylinder', pos=vec((*xy,TABLE_Z+rack_height/2)),
                          size=vec((.006,rack_height/2)), rgba='.15 .23 .29 1', contype='8', conaffinity='5')
        body = ET.SubElement(world, 'body', name=name, pos=vec((*xy, TABLE_Z+rack_height+.0002)))
        if name != 'small_carrier':
            ET.SubElement(body, 'freejoint', name=name)
        # Approximate steel density, exact visual-mesh volume; explicit inertial
        # avoids counting shared convex boundaries twice as material.
        mass = part['volume_m3']*7800
        extents = np.diff(np.array(part['bounds']), axis=0)[0]
        inertia = mass/12 * np.array([extents[1]**2+extents[2]**2, extents[0]**2+extents[2]**2, extents[0]**2+extents[1]**2])
        ET.SubElement(body, 'inertial', pos=vec(np.mean(part['bounds'], axis=0)), mass=str(mass), diaginertia=vec(inertia))
        meshname = name+'_visual'
        ET.SubElement(asset, 'mesh', name=meshname, file=f'meshes/{name}/visual.stl')
        ET.SubElement(body, 'geom', name=meshname, type='mesh', mesh=meshname, contype='0', conaffinity='0', group='2', rgba=COLORS[name], density='0')
        # A convex hull is valid for supporting an upright part on the table.
        # Use it ONLY against table/racks (bit 8); assembly and gripper contacts
        # continue to use the concavity-preserving section meshes (bit 2).
        ET.SubElement(body, 'geom', name=name+'_table_contact', type='mesh', mesh=meshname,
                      contype='4', conaffinity='8', group='3', density='0', friction='.65 .005 .0001')
        for i in range(part['collision_pieces']):
            meshname = f'{name}_col_{i:03d}'
            ET.SubElement(asset, 'mesh', name=meshname, file=f'meshes/{name}/collision_{i:03d}.stl')
            ET.SubElement(body, 'geom', name=meshname, type='mesh', mesh=meshname, group='3', rgba=COLORS[name],
                          contype='2', conaffinity='3', condim='4', friction='.65 .005 .0001', solref='.003 1', solimp='.95 .99 .0001', density='0')
        ET.SubElement(body, 'site', name=name+'_origin', pos='0 0 0', size='.001', group='4')
    if with_pins:
        # The exported pins are straight clearance-fit cylinders with no heads
        # or clips. Physical backing stops in the assembly fixture set their
        # insertion depth; without a stop they can fall through the bores.
        fixture_body = world.find("body[@name='small_carrier']")
        for name in PIN_NOMINAL:
            delta = np.array(manifest['parts'][name]['assembled_origin'])-manifest['parts']['small_carrier']['assembled_origin']
            ET.SubElement(fixture_body, 'geom', name=name+'_depth_stop', type='cylinder',
                          pos=vec([*delta[:2], delta[2]/2]), size=vec([.003, delta[2]/2]),
                          rgba='.15 .23 .29 1', contype='16', conaffinity='3', density='0',
                          solref='.003 1', solimp='.95 .99 .0001')
    # A fixture holds the small carrier; all parts to be manipulated are free.
    ET.SubElement(world, 'geom', name='fixture', type='cylinder', pos=vec((*NOMINAL['small_carrier'], .049)), size='.013 .001', rgba='.12 .22 .28 1', contype='0', conaffinity='0')
    ET.indent(root)
    scene = ASSETS/('scene_pins.xml' if with_pins else 'scene.xml')
    ET.ElementTree(root).write(scene, encoding='unicode')
    return scene


class ActuatorSim(PantheraSim):
    def __init__(self, seed=0, xy_jitter=.002, yaw_jitter_deg=2., with_pins=False, *, scene=None):
        self.manifest = json.loads((ASSETS/'manifest.json').read_text())
        self.seed, self.xy_jitter, self.yaw_jitter_deg = seed, xy_jitter, yaw_jitter_deg
        self.with_pins = with_pins
        self.stations = NOMINAL | PIN_NOMINAL if with_pins else NOMINAL
        super().__init__(scene or ASSETS/('scene_pins.xml' if with_pins else 'scene.xml'))
        self._cube_geoms = {gid:name for name in self.object_names
                            for gid in range(self.model.ngeom)
                            if self.model.geom_bodyid[gid] == self.model.body(name).id}

    def reset(self, *, randomize=True, rng=None):
        # Override cube scattering, including during the parent constructor.
        super().reset(randomize=False)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        for name, xy in self.stations.items():
            dxy = rng.uniform(-self.xy_jitter, self.xy_jitter, 2) if randomize else np.zeros(2)
            yaw = np.deg2rad(rng.uniform(-self.yaw_jitter_deg, self.yaw_jitter_deg)) if randomize else 0.
            rack_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name+'_rack')
            rack_height = GEAR_RACK_HEIGHT if rack_id >= 0 else 0.
            pos = np.r_[np.array(xy)+dxy, TABLE_Z+rack_height+.0002]
            if rack_height:
                self.model.geom_pos[rack_id,:2] = pos[:2]
            quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            bid = self.model.body(name).id
            if name == 'small_carrier':
                self.model.body_pos[bid] = pos
                self.model.body_quat[bid] = quat
                self.model.geom_pos[self.model.geom('fixture').id, :2] = pos[:2]
            else:
                self.set_part_pose(name, pos, quat)
        mujoco.mj_forward(self.model, self.data)

    def set_part_pose(self, name, pos, quat=(1, 0, 0, 0)):
        j = self.model.joint(name).id
        a, v = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.data.qpos[a:a+7] = np.r_[pos, quat]
        self.data.qvel[v:v+6] = 0

    def assembled_pose(self, name):
        small = self.model.body('small_carrier').id
        rotation = self.data.xmat[small].reshape(3, 3)
        parts = self.manifest['parts']
        delta = np.array(parts[name]['assembled_origin'])-parts['small_carrier']['assembled_origin']
        return self.data.xpos[small]+rotation@delta, self.data.xquat[small].copy()

    def part_contacts(self, name, *, include_robot=True):
        bid = self.model.body(name).id
        out = []
        for c in self.data.contact:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if bid not in (b1, b2):
                continue
            other = b2 if b1 == bid else b1
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other)
            if not include_robot and other_name not in self.stations and other != 0:
                continue
            out.append(dict(other=other_name, distance_m=float(c.dist),
                            geom1=self.model.geom(c.geom1).name, geom2=self.model.geom(c.geom2).name))
        return sorted(out, key=lambda x: x['distance_m'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with-pins', action='store_true')
    args = parser.parse_args()
    if args.build:
        print(build_scene(with_pins=args.with_pins))
    else:
        import mujoco.viewer
        import time
        sim = ActuatorSim(seed=args.seed, with_pins=args.with_pins)
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.opt.geomgroup[3] = 0
            while viewer.is_running():
                start = time.monotonic()
                sim.step(10)
                viewer.sync()
                time.sleep(max(0, .01-(time.monotonic()-start)))
