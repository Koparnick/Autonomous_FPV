"""
Otonom FPV "kameraman" düğümü
------------------------------
ROS2 Humble + YOLOv8 + Gazebo + ArduCopter SITL (pymavlink ile)

Akış:
  1. ArduCopter'a bağlan  -> GUIDED mod -> ARM -> otonom kalkış (varsayılan 10 m)
  2. Gimbal'ı aşağı eğ (atan(irtifa / mesafe) => 10 m / 10 m için -45°)
  3. SEARCH : hedef yokken rastgele dolaş + yavaşça dönerek etrafı tara
  4. TRACK  : araba/kamyon/otobüs bulunca
       - yaw hızı   -> hedefi yatayda ekran ortasında tut
       - ileri/geri -> hedefi dikeyde ekran ortasında tut
                       (gimbal sabit açıda olduğu için bu = ~10 m yatay mesafe)
       - yan hız    -> sabit hız => hedefin etrafında yörünge (orbit)
  5. Ctrl+C -> RTL (eve dön ve in)

Kurulum:  pip install pymavlink
Çalıştırma:
  python3 tracker_node.py
  python3 tracker_node.py --ros-args -p connection:=tcp:127.0.0.1:5762 -p orbit_speed:=2.5
"""
import math
import random
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from pymavlink import mavutil
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

# COCO sınıfları: 2=car, 5=bus, 7=truck
TARGET_CLASSES = {2: 'car', 5: 'bus', 7: 'truck'}

# SET_POSITION_TARGET_LOCAL_NED: sadece hız (vx,vy,vz) + yaw_rate kullan
# pos(1|2|4) + accel(64|128|256) + yaw(1024) yok sayılır
TYPE_MASK_VEL_YAWRATE = 0b0000_0101_1100_0111


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class DroneTracker(Node):
    def __init__(self):
        super().__init__('drone_tracker')

        # ---------------- Parametreler ----------------
        p = self.declare_parameter
        self.conn_str = p('connection', 'udpin:0.0.0.0:14550').value
        self.alt = float(p('altitude', 10.0).value)           # m
        self.distance = float(p('distance', 10.0).value)      # m (yatay)
        self.orbit_speed = float(p('orbit_speed', 2.0).value)  # m/s
        self.orbit_dir = int(p('orbit_dir', 1).value)          # 1 = saat yönü tersi(sağa kay), -1 = sola
        self.search_radius = float(p('search_radius', 30.0).value)  # m, evden max uzaklık
        self.search_speed = float(p('search_speed', 1.5).value)     # m/s
        self.conf_th = float(p('conf', 0.25).value)            # yeni kilit için min güven
        self.conf_low = float(p('conf_low', 0.10).value)       # kilitliyken takibi sürdürmek için min güven
        self.imgsz = int(p('imgsz', 960).value)                # YOLO giriş boyutu (640 uzaktaki aracı kaçırıyor)
        self.confirm_frames = int(p('confirm_frames', 3).value)  # kilit için art arda kare sayısı
        self.blind_zone = float(p('blind_zone', 0.05).value)  # ekranın üst kısmı (pervaneler)
        self.hfov = float(p('hfov', 2.0).value)                # rad, gimbal_small_3d kamerası
        self.target_len = float(p('target_len', 5.0).value)    # m, tipik araç boyu
        self.size_gate = (0.35, 2.2)                           # kutu boyu / beklenen boy oranı sınırları
        self.debug = bool(p('debug', True).value)              # tüm tespitleri gri çiz

        # Gimbal açısı: hedef ekran ortasındayken yatay mesafe = self.distance olsun
        self.gimbal_pitch = -math.degrees(math.atan2(self.alt, self.distance))

        # ---------------- Durum ----------------
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.target = None          # {'ex','ey','t','name'}
        self.phase = 'BAGLANIYOR'
        self.rel_alt = 0.0
        self.local_pos = (0.0, 0.0, 0.0)
        self.armed = False
        self.custom_mode = -1
        self.master = None
        self.params = {}
        self.ex_f = 0.0             # filtrelenmiş hatalar
        self.ey_f = 0.0
        self.trk = None             # görüntüdeki aday/kilit: {'box','hits','t','locked','name'}

        # ---------------- Görüntü ----------------
        self.bridge = CvBridge()
        self.get_logger().info("YOLOv8 yükleniyor...")
        self.model = YOLO('yolov8n.pt')
        # İlk çıkarım saniyeler sürüyor; uçuş başlamadan ısıt
        self.model(np.zeros((480, 640, 3), np.uint8), verbose=False, imgsz=self.imgsz)
        self.get_logger().info("YOLOv8 hazır.")

        self.subscription = self.create_subscription(
            Image,
            '/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image',
            self.image_callback,
            1)  # kuyruk 1: çıkarım yavaşsa eski kareler birikip gecikme yaratmasın

        # Uçuş görevi ayrı thread'de (YOLO çıkarımı kontrol döngüsünü bloklamasın)
        self.mission_thread = threading.Thread(target=self.mission, daemon=True)
        self.mission_thread.start()

    # ======================================================================
    #                               GÖRÜNTÜ
    # ======================================================================
    def expected_px(self, v, w, h, alt):
        """Yer düzlemi geometrisi: görüntünün v satırında duran target_len metrelik bir
        aracın piksel boyu. Ufuk ve üstü için 0 döner."""
        f = (w / 2) / math.tan(self.hfov / 2)
        pitch = math.radians(-self.gimbal_pitch)
        down = math.sin(pitch) + math.cos(pitch) * (v - h / 2) / f
        if down < 0.05:
            return 0.0
        return f * self.target_len * down / alt

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        cx_img, cy_img = w // 2, h // 2
        blind_y = int(h * self.blind_zone)
        with self.lock:
            phase, alt = self.phase, self.rel_alt

        t_inf = time.time()
        results = self.model(frame, verbose=False, conf=self.conf_low, imgsz=self.imgsz)
        now = time.time()
        inf_ms = (now - t_inf) * 1000

        cands = []  # hedef sınıfı tespitleri: (conf, box, name)
        others = []  # filtreleri geçen diğer sınıflar (kilidi sürdürmek için)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                is_target = cls_id in TARGET_CLASSES
                if self.debug and not is_target and conf > self.conf_th:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"{self.model.names[cls_id]} {conf:.2f}", (x1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

                reject = None
                # Pervane filtresi: kutunun MERKEZİ kör bölgedeyse at
                if (y1 + y2) / 2 < blind_y:
                    reject = "kor bolge"
                # Zemin filtresi: pist çizgisi gibi yanlış tespitler bu mesafedeki bir
                # araçtan çok daha büyük (ya da drone gölgesi gibi çok daha küçük) olur
                elif alt > 3.0:
                    exp = self.expected_px((y1 + y2) / 2, w, h, alt)
                    ratio = max(x2 - x1, y2 - y1) / exp if exp > 0 else 99.0
                    if not self.size_gate[0] < ratio < self.size_gate[1]:
                        reject = f"boyut x{ratio:.1f}"
                if reject:
                    if self.debug and is_target:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 1)
                        cv2.putText(frame, f"RED {reject}", (x1, y1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)
                    continue
                if is_target:
                    cands.append((conf, (x1, y1, x2, y2), TARGET_CLASSES[cls_id]))
                else:
                    others.append((conf, (x1, y1, x2, y2), None))

        # Kilit mantığı: yeni kilit için confirm_frames kare üst üste yüksek güvenli hedef
        # sınıfı gerekir. Kilitlendikten sonra aynı yerdeki düşük güvenli tespitler ve
        # (YOLO aracı tepeden başka sınıf sandığında) aynı boydaki diğer sınıflar da
        # kabul edilir; ama hedef sınıfı 2 sn görülmezse kilit bırakılır.
        best = None
        trk = self.trk
        if trk is not None:
            bx1, by1, bx2, by2 = trk['box']
            size = max(bx2 - bx1, by2 - by1)
            gate = 0.6 * size + 0.06 * w
            pool = cands
            min_conf = self.conf_th
            if trk['locked']:
                min_conf = self.conf_low
                if now - trk['t_cls'] < 2.0:
                    pool = cands + others
            near = None
            for c in pool:
                x1, y1, x2, y2 = c[1]
                d = math.hypot((x1 + x2 - bx1 - bx2) / 2, (y1 + y2 - by1 - by2) / 2)
                if c[2] is None:
                    if not 0.6 < max(x2 - x1, y2 - y1) / size < 1.6:
                        continue
                    d += 0.5 * gate  # aynı yerde hedef sınıfı varsa onu tercih et
                if c[0] >= min_conf and d < gate and (near is None or d < near[0]):
                    near = (d, c)
            if near is not None:
                best = near[1]
                trk.update(box=best[1], t=now, hits=trk['hits'] + 1)
                if best[2] is not None:
                    trk.update(name=best[2], t_cls=now)
                trk['locked'] = trk['locked'] or trk['hits'] >= self.confirm_frames
            elif now - trk['t'] > (1.0 if trk['locked'] else 0.3):
                trk = self.trk = None
        if trk is None:
            strong = [c for c in cands if c[0] >= self.conf_th]
            if strong:
                best = max(strong, key=lambda c: c[0])
                trk = self.trk = {'box': best[1], 'name': best[2], 't': now, 't_cls': now,
                                  'hits': 1, 'locked': self.confirm_frames <= 1}

        # HUD
        cv2.line(frame, (0, blind_y), (w, blind_y), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(frame, (cx_img, cy_img), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        if best is not None and not trk['locked']:
            conf, (x1, y1, x2, y2), name = best
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, f"ADAY {trk['hits']}/{self.confirm_frames} {name} {conf:.2f}",
                        (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        elif best is not None:
            conf, (x1, y1, x2, y2), name = best[0], best[1], trk['name']
            tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
            ex = (tx - cx_img) / cx_img
            ey = (ty - cy_img) / cy_img
            with self.lock:
                self.target = {'ex': ex, 'ey': ey, 't': now, 'name': name}
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (cx_img, cy_img), (tx, ty), (0, 255, 255), 2)
            cv2.putText(frame, f"KILIT: {name} {conf:.2f} | ex {ex:+.2f} ey {ey:+.2f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"YOLO {inf_ms:.0f} ms", (w - 130, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(frame, f"MOD: {phase} | ALT: {alt:.1f} m", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("FPV Terminal Guidance HUD", frame)
        cv2.waitKey(1)

    # ======================================================================
    #                               MAVLINK
    # ======================================================================
    def connect(self):
        self.get_logger().info(f"ArduCopter'a bağlanılıyor: {self.conn_str}")
        self.master = mavutil.mavlink_connection(self.conn_str)
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"Heartbeat alındı (sys {self.master.target_system}, comp {self.master.target_component})")
        threading.Thread(target=self.mav_reader, daemon=True).start()

        # Gerekli mesajları 10 Hz iste
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                       mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED):
            self.command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 100000)

        self.ensure_gimbal_params()

    def get_param(self, name, timeout=3.0):
        self.params.pop(name, None)
        self.master.mav.param_request_read_send(
            self.master.target_system, self.master.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if name in self.params:
                return self.params[name]
            time.sleep(0.05)
        return None

    def ensure_gimbal_params(self):
        """ArduPilot gimbal'ı Gazebo'daki /gimbal/cmd_* topic'lerine servo çıkışlarıyla sürer.
        MNT1_TYPE=0 ise mount komutları yok sayılır ve kamera düz bakar."""
        wanted = {
            'SERVO9_FUNCTION': 8,    # Mount1 Roll  -> /gimbal/cmd_roll
            'SERVO10_FUNCTION': 7,   # Mount1 Tilt  -> /gimbal/cmd_pitch
            'SERVO11_FUNCTION': 6,   # Mount1 Pan   -> /gimbal/cmd_yaw
            # Gazebo modeli (iris_with_gimbal) servo aralığını -135..+45 dereceye eşler;
            # buradaki sınırlar farklı olursa komut edilen açı ile gerçek açı tutmaz.
            'MNT1_PITCH_MIN': -135,
            'MNT1_PITCH_MAX': 45,
            'MNT1_DEFLT_MODE': 2,    # MAVLink targeting
        }
        for name, val in wanted.items():
            cur = self.get_param(name)
            if cur is not None and abs(cur - val) > 1e-3:
                self.master.param_set_send(name, val)
                self.get_logger().info(f"{name}: {cur} -> {val}")

        mnt_type = self.get_param('MNT1_TYPE')
        if mnt_type is not None and int(mnt_type) != 1:
            self.master.param_set_send('MNT1_TYPE', 1)
            self.get_logger().error(
                "MNT1_TYPE=0 idi, 1 yapıldı. Gimbal'ın çalışması için ArduPilot SITL'i "
                "YENİDEN BAŞLATIN (sim_vehicle.py'yi kapatıp açın).")

    def mav_reader(self):
        m = self.master
        while not self.stop_evt.is_set():
            msg = m.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            t = msg.get_type()
            if t == 'PARAM_VALUE':
                self.params[msg.param_id] = msg.param_value
                continue
            if t == 'HEARTBEAT':
                if msg.get_srcSystem() == m.target_system and msg.get_srcComponent() == 1:
                    with self.lock:
                        self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        self.custom_mode = msg.custom_mode
            elif t == 'GLOBAL_POSITION_INT':
                with self.lock:
                    self.rel_alt = msg.relative_alt / 1000.0
            elif t == 'LOCAL_POSITION_NED':
                with self.lock:
                    self.local_pos = (msg.x, msg.y, msg.z)
            elif t == 'STATUSTEXT':
                self.get_logger().info(f"[AP] {msg.text}")

    def command(self, cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            cmd, 0, p1, p2, p3, p4, p5, p6, p7)

    def send_velocity(self, vx, vy, vz, yaw_rate, body=True):
        """body=True: vx ileri, vy sağ | body=False: vx kuzey, vy doğu. vz + = aşağı."""
        frame = (mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED if body
                 else mavutil.mavlink.MAV_FRAME_LOCAL_NED)
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            frame, TYPE_MASK_VEL_YAWRATE,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)

    def point_gimbal(self):
        mount_mode = mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
        self.command(mavutil.mavlink.MAV_CMD_DO_MOUNT_CONFIGURE, mount_mode, 1, 1, 1)
        self.command(mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                     self.gimbal_pitch, 0, 0, 0, 0, 0, mount_mode)

    def set_phase(self, phase):
        with self.lock:
            if self.phase != phase:
                self.get_logger().info(f"Mod: {self.phase} -> {phase}")
            self.phase = phase

    # ======================================================================
    #                               GÖREV
    # ======================================================================
    def takeoff(self):
        guided = self.master.mode_mapping()['GUIDED']

        # İlk irtifa mesajını bekle (yoksa rel_alt=0 sanıp yanlış karar veririz)
        t0 = time.time()
        while self.master.messages.get('GLOBAL_POSITION_INT') is None and time.time() - t0 < 5.0:
            time.sleep(0.1)

        self.set_phase('GUIDED')
        while not self.stop_evt.is_set():
            with self.lock:
                if self.custom_mode == guided:
                    break
            self.master.set_mode(guided)
            time.sleep(1.0)

        self.set_phase('ARM')
        while not self.stop_evt.is_set():
            with self.lock:
                if self.armed:
                    break
            self.get_logger().info("ARM deneniyor (EKF/pre-arm kontrolleri bekleniyor)...")
            self.master.arducopter_arm()
            time.sleep(2.0)

        self.set_phase('KALKIS')
        with self.lock:
            alt = self.rel_alt
        if alt > 1.5:
            # Zaten havada: NAV_TAKEOFF yok sayılır, beklemeye gerek yok.
            # Eksik irtifayı control_loop'taki irtifa tutucu tamamlar.
            self.get_logger().info(f"Drone zaten havada ({alt:.1f} m), kalkış atlanıyor")
            return

        last_cmd = 0.0
        t_start = time.time()
        while not self.stop_evt.is_set():
            with self.lock:
                alt, armed = self.rel_alt, self.armed
            if alt >= self.alt * 0.92:
                break
            # Kalkış belli bir irtifada takılırsa (ör. 8 m'de durduysa) yine de devam et
            if alt > 2.0 and time.time() - t_start > 20.0:
                self.get_logger().warn(f"Kalkış {alt:.1f} m'de takıldı, görev başlıyor")
                break
            if not armed:  # takeoff gecikirse ArduCopter disarm eder -> tekrar
                self.get_logger().warn("Disarm oldu, tekrar ARM ediliyor")
                self.master.arducopter_arm()
                time.sleep(2.0)
                continue
            if alt < 1.0 and time.time() - last_cmd > 5.0:
                self.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, self.alt)
                last_cmd = time.time()
            time.sleep(0.3)
        self.get_logger().info(f"Kalkış tamam: {self.rel_alt:.1f} m")

    def mission(self):
        try:
            self.connect()
            self.point_gimbal()
            self.takeoff()
            self.point_gimbal()
            self.control_loop()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Görev hatası: {e}")

    def control_loop(self):
        dt = 0.1  # 10 Hz
        last_gimbal = time.time()
        search_vel = (0.0, 0.0)
        next_search_change = 0.0

        while not self.stop_evt.is_set():
            now = time.time()
            with self.lock:
                tgt = dict(self.target) if self.target else None
                alt = self.rel_alt
                n, e, _ = self.local_pos

            # Gimbal komutunu ara ara tazele
            if now - last_gimbal > 3.0:
                self.point_gimbal()
                last_gimbal = now

            # İrtifa tutma (NED: vz + = aşağı)
            vz = clamp(-0.8 * (self.alt - alt), -1.0, 1.0)
            age = (now - tgt['t']) if tgt else 1e9

            if age < 0.7:
                # ------------------- TRACK / ORBIT -------------------
                self.set_phase(f"TAKIP ({tgt['name']})")
                a = 0.4  # alçak geçiren filtre
                self.ex_f = (1 - a) * self.ex_f + a * tgt['ex']
                self.ey_f = (1 - a) * self.ey_f + a * tgt['ey']

                yaw_rate = clamp(1.2 * self.ex_f, -0.9, 0.9)   # rad/s, + = sağa dön
                # Hedef ekranda aşağıdaysa (ey>0) çok yakınız -> geri git
                vx = clamp(-3.0 * self.ey_f, -2.5, 2.5)
                # Hedef ortalandıkça yanlamasına kay -> çember
                centered = max(0.0, 1.0 - 2.0 * abs(self.ex_f))
                vy = self.orbit_dir * self.orbit_speed * centered
                self.send_velocity(vx, vy, vz, yaw_rate, body=True)

            elif age < 3.0:
                # ------------- Kısa süreli kayıp: havada bekle -------------
                self.set_phase('HEDEF KAYIP - BEKLE')
                self.ex_f *= 0.8
                self.ey_f *= 0.8
                self.send_velocity(0, 0, vz, 0.0, body=True)

            else:
                # ------------------- SEARCH (rastgele) -------------------
                self.set_phase('ARAMA')
                dist_home = math.hypot(n, e)
                if dist_home > self.search_radius:
                    # Sınır dışı: eve doğru dön
                    search_vel = (-n / dist_home * self.search_speed,
                                  -e / dist_home * self.search_speed)
                    next_search_change = now + 3.0
                elif now > next_search_change:
                    hdg = random.uniform(0, 2 * math.pi)
                    spd = random.uniform(0.5, self.search_speed)
                    search_vel = (spd * math.cos(hdg), spd * math.sin(hdg))
                    next_search_change = now + random.uniform(3.0, 6.0)
                # Dolaşırken yavaşça dönerek etrafı tara
                self.send_velocity(search_vel[0], search_vel[1], vz, 0.35, body=False)

            time.sleep(dt)

    def shutdown(self):
        self.stop_evt.set()
        if self.master is not None:
            try:
                self.get_logger().info("RTL moduna geçiliyor...")
                self.master.set_mode(self.master.mode_mapping()['RTL'])
            except Exception:  # noqa: BLE001
                pass


def main(args=None):
    rclpy.init(args=args)
    node = DroneTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
