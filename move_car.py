#!/usr/bin/env python3
"""
Gazebo (gz sim Harmonic) içindeki hedef aracı hareket ettirir.
set_pose servisini gz-transport Python API'si ile 20 Hz çağırır (plugin gerekmez).

Kullanım:
  python3 move_car.py                          # hedef_arac'ı çemberde gezdir
  python3 move_car.py --mode line --speed 3    # ileri-geri
  python3 move_car.py --radius 20 --speed 4
"""
import argparse
import math
import threading
import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node


def get_start_pose(node, world, name, timeout=5.0):
    """Aracın mevcut konumunu /world/<world>/pose/info'dan oku."""
    found = {}
    evt = threading.Event()

    def cb(msg: Pose_V):
        for p in msg.pose:
            if p.name == name:
                found['p'] = (p.position.x, p.position.y, p.position.z)
                evt.set()
                return

    node.subscribe(Pose_V, f'/world/{world}/pose/info', cb)
    evt.wait(timeout)
    return found.get('p')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='iris_runway')
    ap.add_argument('--name', default='hedef_arac')
    ap.add_argument('--mode', choices=['circle', 'line'], default='circle')
    ap.add_argument('--speed', type=float, default=2.0, help='m/s')
    ap.add_argument('--radius', type=float, default=15.0, help='circle: yarıçap / line: yarım uzunluk')
    a = ap.parse_args()

    node = Node()
    start = get_start_pose(node, a.world, a.name)
    if start is None:
        print(f"UYARI: '{a.name}' konumu okunamadı, (0,0,0) varsayılıyor")
        start = (0.0, 0.0, 0.0)
    sx, sy, sz = start
    print(f"'{a.name}' başlangıç: x={sx:.1f} y={sy:.1f} z={sz:.2f} | mod={a.mode}. Ctrl+C ile dur.")

    service = f'/world/{a.world}/set_pose'
    t0 = time.time()
    fails = 0
    try:
        while True:
            t = time.time() - t0
            if a.mode == 'circle':
                # Araç başlangıç noktasından geçen bir çember çizer
                ang = (a.speed / a.radius) * t
                x = sx - a.radius + a.radius * math.cos(ang)
                y = sy + a.radius * math.sin(ang)
                yaw = ang + math.pi / 2
            else:
                period = 4 * a.radius / a.speed
                ph = (t % period) / period
                s = 4 * ph if ph < 0.25 else (2 - 4 * ph if ph < 0.75 else 4 * ph - 4)  # 0..1..-1..0
                x = sx + a.radius * s
                y = sy
                yaw = 0.0 if (ph < 0.25 or ph >= 0.75) else math.pi

            req = Pose()
            req.name = a.name
            req.position.x, req.position.y, req.position.z = x, y, sz
            req.orientation.z = math.sin(yaw / 2)
            req.orientation.w = math.cos(yaw / 2)
            ok, rep = node.request(service, req, Pose, Boolean, 1000)
            if not (ok and rep.data):
                fails += 1
                if fails % 20 == 1:
                    print("set_pose başarısız (model adı / dünya adı doğru mu?)")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
