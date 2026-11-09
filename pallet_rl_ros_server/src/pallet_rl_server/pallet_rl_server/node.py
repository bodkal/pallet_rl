"""ROS 2 node: the `PlanPlacement` service around `Planner`.

    ros2 run pallet_rl_server server --ros-args -p config:=/path/server.yaml

The one ROS parameter is `config`, the settings file; everything else is in
it.  Requests are served one at a time (single-threaded executor).
"""
from __future__ import annotations

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from pallet_rl_interfaces.msg import Box as BoxMsg
from pallet_rl_interfaces.srv import PlanPlacement

from . import settings, snapshot
from .planner import Box, Planner


def _box(m):
    return Box(m.length_cm, m.width_cm, m.height_cm, m.type,
               m.x_cm, m.y_cm, m.z_cm)


def _msg(b):
    return BoxMsg(x_cm=float(b.x_cm), y_cm=float(b.y_cm), z_cm=float(b.z_cm),
                  length_cm=float(b.length_cm), width_cm=float(b.width_cm),
                  height_cm=float(b.height_cm), type=int(b.type))


class PlannerNode(Node):
    def __init__(self):
        super().__init__("pallet_rl_server")
        default = os.path.join(get_package_share_directory("pallet_rl_server"),
                               "config", "server.yaml")
        path = self.declare_parameter("config", default).value
        cfg = settings.load(path)
        self.cfg = cfg
        self.planner = Planner(cfg, log=self.get_logger().warning)
        self.snap_path = cfg["server"]["snapshot_file"]
        self.seq = 0
        name = cfg["server"]["service"]
        self.create_service(PlanPlacement, name, self.on_request)
        r = self.planner.rules
        self.get_logger().info(
            f"ready on {name}: {r.Lx}x{r.Ly}x{r.Lz} cells of "
            f"{self.planner.cell_cm:g} cm, rule {r.type_rule}, "
            f"checkpoint {cfg['server']['checkpoint']}"
            + (f", last step -> {self.snap_path}" if self.snap_path else ""))

    def on_request(self, req, res):
        plan = self.planner.plan([_box(m) for m in req.placed],
                                 [_box(m) for m in req.station])
        res.status = plan.status
        res.message = plan.message
        res.station_index = plan.station_index
        res.rotated = plan.rotated
        res.pick_probability = plan.pick_probability
        res.place_probability = plan.place_probability
        res.n_candidates = plan.n_candidates
        if plan.placement is not None:
            res.placement = _msg(plan.placement)
        p = plan.placement
        where = (f"box {plan.station_index} -> ({p.x_cm:g}, {p.y_cm:g}, "
                 f"{p.z_cm:g}) cm{' rotated' if plan.rotated else ''}"
                 if p else plan.message)
        self.get_logger().info(
            f"[{req.pallet_id}] {len(req.placed)} placed, "
            f"{len(req.station)} at station: status {plan.status}, {where} "
            f"({plan.ms:.0f} ms)")
        for w in plan.warnings:
            self.get_logger().warning(f"[{req.pallet_id}] {w}")
        if self.snap_path:
            self.seq += 1
            try:
                snapshot.write(self.snap_path, snapshot.build(
                    plan, req.pallet_id, self.seq, self.cfg))
            except (OSError, TypeError, ValueError) as e:
                self.get_logger().warning(
                    f"could not write the snapshot {self.snap_path}: {e}")
        return res


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
