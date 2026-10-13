import json
import math
import tempfile
import unittest
import queue
import threading
import time
import socket
from pathlib import Path

import cv2
import numpy as np

from real_device.angles import interpolate_circular, wrap_deg
from real_device.cli import evaluate_predictions
from real_device.export import export_angle_validation, export_training_episode
from real_device.metrics import angular_metrics, image_metrics
from real_device.protocol import SequenceTracker, SensorSample, parse_packet
from real_device.session import TrialController, TrialState
from real_device.sync import SyncExchange, clock_quality, fit_clock
from real_device.vision import HSVRange, detect_markers
from real_device.camera import CameraFrame
from real_device.pipeline import _compress_frame, capture_real, probe_sensor, synthetic_demo
from real_device.sync import ClockMap


class FakeReceiver:
    def __init__(self,restart=False):
        self.samples=queue.Queue(); self.responses=queue.Queue(); self.tracker=SequenceTracker(); self.boot_id="fake-a"; self.host_queue_drops=0; self.error=None
        self.stop=threading.Event(); self.restart=restart; self.base_pc=time.monotonic_ns(); self.base_dev=1_000_000; self.thread=None
    def start(self):
        def produce():
            seq=0
            while not self.stop.is_set():
                elapsed=(time.monotonic_ns()-self.base_pc)//1000; boot="fake-b" if self.restart and seq>125 else "fake-a"
                item=SensorSample(1,boot,seq,self.base_dev+elapsed,20000,81967,0.0,True,0,0,"test",time.monotonic_ns())
                self.tracker.accept(item); self.samples.put(item); seq+=1; time.sleep(.002)
        self.thread=threading.Thread(target=produce,daemon=True); self.thread.start()
    def synchronize(self,count=16,timeout_s=1):
        slope=1.0; intercept=self.base_pc-self.base_dev*1000; items=[]
        for i in range(6):
            dev=self.base_dev+i*100_000; pc=int(dev*1000+intercept); items.append(SyncExchange(pc-500_000,dev,dev,pc+500_000))
        return ClockMap(slope,intercept,0.0,1_000_000,6,6),items
    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(.2)


class FakeCamera:
    timestamp_kind="fake_exposure"
    actual_fps=100.0; actual_width=64; actual_height=64
    def __init__(self,blocked=False): self.closed=False; self.frame_id=0; self.blocked=blocked
    def read(self):
        if self.blocked: time.sleep(2)
        else: time.sleep(.01)
        if self.closed: return None
        stamp=time.monotonic_ns(); image=np.full((64,64,3),self.frame_id%255,np.uint8); item=CameraFrame(self.frame_id,image,stamp-1_000_000,stamp,stamp,self.timestamp_kind); self.frame_id+=1; return item
    def close(self): self.closed=True


class SyncTests(unittest.TestCase):
    def test_offset_drift_and_high_rtt_rejection(self):
        slope, intercept = 1.00012, 4_200_000_000.0
        exchanges = []
        for i in range(20):
            device_mid = 1_000_000_000 + i * 500_000_000
            delay = 800_000 if i < 15 else 80_000_000
            pc_mid = slope * device_mid + intercept
            t0 = int(pc_mid - delay / 2); t3 = int(pc_mid + delay / 2)
            device_us = int(device_mid / 1000)
            exchanges.append(SyncExchange(t0, device_us, device_us, t3))
        clock = fit_clock(exchanges, keep_fraction=.5)
        self.assertAlmostEqual(clock.drift_ppm, 120, delta=2)
        self.assertAlmostEqual(clock.intercept_ns, intercept, delta=30_000)
        self.assertLess(clock.max_rtt_ns, 2_000_000)
        self.assertTrue(clock_quality(clock)[0])

    def test_quality_rejects_high_rtt(self):
        exchanges = [SyncExchange(i*1_000_000_000, i*1_000_000, i*1_000_000, i*1_000_000_000+50_000_000) for i in range(6)]
        clock = fit_clock(exchanges, keep_fraction=1)
        self.assertIn("sync_rtt_too_high", clock_quality(clock)[1])

    def test_invalid_exchange_and_nonfinite_clock_rejected(self):
        with self.assertRaises(ValueError): SyncExchange(10,10,9,20)
        from real_device.sync import ClockMap
        self.assertIn("sync_fit_nonfinite",clock_quality(ClockMap(float("nan"),0,0,0,4,4))[1])


class AngleTests(unittest.TestCase):
    def test_wrap_interpolation_and_gap_no_extrapolation(self):
        values, valid = interpolate_circular([0, 10_000_000, 100_000_000], [179, -179, 0], [-1, 5_000_000, 50_000_000, 100_000_000, 110_000_000], max_gap_ms=20)
        self.assertFalse(valid[0]); self.assertTrue(valid[1]); self.assertAlmostEqual(abs(values[1]), 180, places=6)
        self.assertFalse(valid[2]); self.assertTrue(valid[3]); self.assertFalse(valid[4])

    def test_nan_source_filtered(self):
        values, valid = interpolate_circular([0, 5, 10], [1, np.nan, 3], [5], max_gap_ms=.001)
        self.assertTrue(valid[0]); self.assertAlmostEqual(values[0], 2)


def sample(boot="a", seq=0):
    return SensorSample(1, boot, seq, seq*1000, 10, 100, 20.0, True)


class ProtocolTests(unittest.TestCase):
    def test_duplicate_reorder_gap_and_boot(self):
        tracker = SequenceTracker()
        self.assertTrue(tracker.accept(sample(seq=1))[0])
        self.assertFalse(tracker.accept(sample(seq=1))[0])
        self.assertTrue(tracker.accept(sample(seq=4))[0]); self.assertEqual(tracker.gaps, 2)
        self.assertFalse(tracker.accept(sample(seq=3))[0])
        self.assertTrue(tracker.accept(sample("b", 0))[0]); self.assertEqual(tracker.restarts, 1)
        self.assertEqual((tracker.duplicates, tracker.reordered), (1, 1))

    def test_parser_rejects_bad_boolean_nonfinite_and_negative(self):
        base = {"v":1,"type":"sample","boot_id":"x","seq":1,"t_device_us":2,"high_ticks":3,"period_ticks":4,"raw_angle_deg":5,"valid":True}
        self.assertIsNotNone(parse_packet(json.dumps(base)))
        for key, value in (("valid", "false"), ("raw_angle_deg", float("nan")), ("seq", -1)):
            bad = dict(base); bad[key] = value
            self.assertIsNone(parse_packet(json.dumps(bad)))


class SessionTests(unittest.TestCase):
    def ready(self, at=1000):
        trial = TrialController(); trial.sensor(at-10, 0); trial.camera(at-10)
        trial.set_ready(at-5, sensor_stable=True, camera_stable=True, clock_valid=True, geometry_confirmed=True, zero_calibrated=True)
        return trial

    def test_initial_angle_gate(self):
        trial = self.ready(); trial.release(1000, 10.01)
        self.assertEqual(trial.state, TrialState.INVALID); self.assertEqual(trial.reason, "initial_angle_out_of_range")
        trial = self.ready(); trial.release(1000, -10.0); self.assertEqual(trial.state, TrialState.RECORDING)

    def test_duration_and_measurement_time_angle_stop(self):
        trial = self.ready(1_000_000_000); trial.release(1_000_000_000, 0)
        trial.sensor(2_000_000_000, 121); self.assertEqual(trial.stop_ns, 2_000_000_000); self.assertEqual(trial.reason, "angle_limit")
        trial = self.ready(1_000_000_000); trial.release(1_000_000_000, 0); trial.sensor(1_050_000_000, 0); trial.camera(1_050_000_000)
        trial.poll(8_000_000_000); self.assertEqual(trial.stop_ns, 7_000_000_000); self.assertEqual(trial.reason, "duration")

    def test_stale_sensor_and_stalled_video(self):
        trial = self.ready(1_000_000_000); trial.release(1_000_000_000, 0); trial.camera(1_150_000_000); trial.poll(1_150_000_001)
        self.assertEqual(trial.reason, "sensor_stale"); self.assertEqual(trial.state, TrialState.INVALID)
        trial = self.ready(1_000_000_000); trial.release(1_000_000_000, 0); trial.sensor(1_550_000_000, 0); trial.poll(1_550_000_001)
        self.assertEqual(trial.reason, "camera_stale")

    def test_prestart_late_sample_cannot_stop(self):
        trial = self.ready(); trial.release(1_000, 0); trial.sensor(999, 170)
        self.assertEqual(trial.state, TrialState.RECORDING)

    def test_deadline_wins_and_late_predeadline_angle_corrects(self):
        trial=self.ready(1_000_000_000); trial.release(1_000_000_000,0); trial.sensor(8_000_000_000,121)
        self.assertEqual((trial.stop_ns,trial.reason),(7_000_000_000,"duration"))
        trial.sensor(6_000_000_000,121); self.assertEqual((trial.stop_ns,trial.reason),(6_000_000_000,"angle_limit"))

    def test_release_rejects_nonfinite_and_stale(self):
        trial=self.ready(); trial.release(1000,float("nan")); self.assertEqual(trial.state,TrialState.INVALID)
        trial=self.ready();
        with self.assertRaisesRegex(RuntimeError,"stale"): trial.release(1_000_000_000,0)


class VisionMetricTests(unittest.TestCase):
    def test_missing_marker_is_invalid_not_zero(self):
        image = np.zeros((100,100,3), np.uint8)
        detection = detect_markers(image, HSVRange((45,100,100),(85,255,255)), HSVRange((0,150,100),(10,255,255)))
        self.assertFalse(detection.valid); self.assertTrue(math.isnan(detection.angle_deg))

    def test_marker_angle_and_wrapped_metrics(self):
        image=np.zeros((200,200,3),np.uint8); cv2.circle(image,(100,150),10,(0,255,0),-1); cv2.circle(image,(100,50),10,(0,0,255),-1)
        detection=detect_markers(image,HSVRange((45,100,100),(85,255,255)),HSVRange((0,150,100),(10,255,255)))
        self.assertTrue(detection.valid); self.assertAlmostEqual(detection.angle_deg,0,places=4)
        result=angular_metrics([179,-179],[ -179,179]); self.assertAlmostEqual(result["mae_deg"],2)
        images=image_metrics(np.zeros((8,8,3),np.uint8),np.zeros((8,8,3),np.uint8)); self.assertEqual(images["mse"],0); self.assertAlmostEqual(images["ssim_global"],1)

    def test_training_export_guard(self):
        with self.assertRaisesRegex(ValueError,"action semantics"):
            export_training_episode("unused")
        with self.assertRaisesRegex(ValueError,"graph labels"):
            export_training_episode("unused",action_semantics="measured pivot velocity")


class PipelineTests(unittest.TestCase):
    def test_compressed_buffer_does_not_retain_raw_image(self):
        image=np.zeros((1080,1920,3),np.uint8); frame=CameraFrame(0,image,1,2,2,"test")
        metadata,jpeg=_compress_frame(frame)
        self.assertNotIn("image",metadata); self.assertIsInstance(jpeg,bytes); self.assertLess(len(jpeg),image.nbytes)

    def test_udp_probe_handshakes_before_receiving_samples(self):
        server=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); server.bind(("127.0.0.1",0)); server.settimeout(1); port=server.getsockname()[1]; saw_sync=threading.Event()
        def firmware():
            seq=0; address=None
            try:
                for _ in range(6):
                    data,address=server.recvfrom(2048); request=json.loads(data); saw_sync.set(); device_us=time.monotonic_ns()//1000
                    response={"v":1,"type":"sync_resp","id":request["id"],"t0_pc_ns":request["t0_pc_ns"],"t1_device_us":device_us,"t2_device_us":device_us,"boot_id":"udp-fake"}
                    server.sendto(json.dumps(response).encode(),address)
                for _ in range(10):
                    device_us=time.monotonic_ns()//1000; sample={"v":1,"type":"sample","boot_id":"udp-fake","seq":seq,"t_device_us":device_us,"high_ticks":20000,"period_ticks":81967,"raw_angle_deg":12.5,"valid":True,"capture_drops":0,"tx_drops":0}; seq+=1
                    server.sendto(json.dumps(sample).encode(),address); time.sleep(.005)
            finally: server.close()
        thread=threading.Thread(target=firmware,daemon=True); thread.start()
        result=probe_sensor(udp_device=("localhost",port),duration_s=.1)
        thread.join(1); self.assertTrue(saw_sync.is_set()); self.assertGreater(result["valid_samples"],0); self.assertAlmostEqual(result["suggested_zero_deg"],12.5)

    def test_capture_rejects_nonfinite_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"finite"):
                capture_real(Path(folder)/"bad",port="COM0",camera_source=0,category="free_fall",zero_deg=float("nan"),direction=1,geometry_confirmed=True)

    def test_initialization_failure_closes_and_records_invalid_metadata(self):
        class Broken(FakeReceiver):
            def synchronize(self,*args,**kwargs): raise RuntimeError("sync failed")
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"bad"
            with self.assertRaisesRegex(RuntimeError,"sync failed"):
                capture_real(target,port=None,camera_source=0,category="free_fall",zero_deg=0,direction=1,geometry_confirmed=True,receiver_override=Broken(),camera_override=FakeCamera(),show_preview=False)
            metadata=json.loads((target/"metadata.json").read_text(encoding="utf-8")); self.assertFalse(metadata["valid_trial"]); self.assertEqual(metadata["stop_reason"],"initialization_failed")

    def test_synthetic_session_video_csv_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=synthetic_demo(Path(folder)/"trial",category="free_fall")
            with (root/"frames.csv").open(encoding="utf-8") as handle: frame_count=sum(1 for _ in handle)-1
            cap=cv2.VideoCapture(str(root/"rgb.mp4")); decoded=0
            while cap.read()[0]: decoded+=1
            cap.release(); self.assertEqual(decoded,frame_count); self.assertGreater(decoded,10)
            output=export_angle_validation(root/"sensor.csv",root/"angles.npz")
            with np.load(output) as data: self.assertGreater(len(data["angle_deg"]),100)
            predictions=root/"future.csv"; predictions.write_text("t_pc_ns,angle_deg\n999999999999999,0\n",encoding="utf-8")
            result=evaluate_predictions(root,predictions,max_gap_ms=20); self.assertEqual(result["count"],0)

    def test_demo_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"trial"; synthetic_demo(target,category="assisted_balance",duration_s=.2)
            with self.assertRaises(FileExistsError): synthetic_demo(target,category="assisted_balance",duration_s=.2)

    def test_injected_capture_trigger_and_video_csv_match(self):
        with tempfile.TemporaryDirectory() as folder:
            fired=[False]
            def key(state,angle):
                if state==TrialState.READY and not fired[0]: fired[0]=True; return ord('r')
                return -1
            root=capture_real(Path(folder)/"trial",port=None,camera_source=0,category="free_fall",zero_deg=0,direction=1,geometry_confirmed=True,duration_s=.15,fps=50,width=64,height=64,readiness_timeout_s=1,receiver_override=FakeReceiver(),camera_override=FakeCamera(),key_provider=key,show_preview=False)
            metadata=json.loads((root/"metadata.json").read_text(encoding="utf-8")); self.assertTrue(metadata["valid_trial"]); self.assertEqual(metadata["stop_reason"],"duration")
            with (root/"frames.csv").open(encoding="utf-8") as handle: rows=sum(1 for _ in handle)-1
            cap=cv2.VideoCapture(str(root/"rgb.mp4")); decoded=0
            while cap.read()[0]: decoded+=1
            cap.release(); self.assertEqual(rows,decoded); self.assertGreater(rows,0)

    def test_injected_restart_and_blocked_camera_are_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fired=[False]
            def key(state,angle):
                if state==TrialState.READY and not fired[0]: fired[0]=True; return ord('r')
                return -1
            root=capture_real(Path(folder)/"restart",port=None,camera_source=0,category="free_fall",zero_deg=0,direction=1,geometry_confirmed=True,duration_s=.5,fps=50,width=64,height=64,readiness_timeout_s=1,receiver_override=FakeReceiver(restart=True),camera_override=FakeCamera(),key_provider=key,show_preview=False)
            meta=json.loads((root/"metadata.json").read_text(encoding="utf-8")); self.assertFalse(meta["valid_trial"]); self.assertIn("restart",meta["stop_reason"])
            root=capture_real(Path(folder)/"blocked",port=None,camera_source=0,category="free_fall",zero_deg=0,direction=1,geometry_confirmed=True,duration_s=.1,fps=30,width=64,height=64,readiness_timeout_s=.1,receiver_override=FakeReceiver(),camera_override=FakeCamera(blocked=True),key_provider=lambda *_:-1,show_preview=False)
            meta=json.loads((root/"metadata.json").read_text(encoding="utf-8")); self.assertFalse(meta["valid_trial"]); self.assertIn(meta["stop_reason"],{"operator_cancelled","camera_close_timeout"})

    def test_no_recorded_frame_cannot_be_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            fired=[False]
            def key(state,angle):
                if state==TrialState.READY and not fired[0]: fired[0]=True; return ord('r')
                return -1
            root=capture_real(Path(folder)/"short",port=None,camera_source=0,category="free_fall",zero_deg=0,direction=1,geometry_confirmed=True,duration_s=.001,fps=50,width=64,height=64,readiness_timeout_s=1,receiver_override=FakeReceiver(),camera_override=FakeCamera(),key_provider=key,show_preview=False)
            meta=json.loads((root/"metadata.json").read_text(encoding="utf-8")); self.assertFalse(meta["valid_trial"]); self.assertEqual(meta["stop_reason"],"insufficient_recorded_data")


if __name__ == "__main__": unittest.main()
