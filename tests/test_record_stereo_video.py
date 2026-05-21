import importlib.util
from contextlib import nullcontext
import signal
import tempfile
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_stereo_video.py"


def load_module():
    spec = importlib.util.spec_from_file_location("record_stereo_video", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordStereoVideoTest(unittest.TestCase):
    def test_module_does_not_import_or_use_opencv(self):
        source = SCRIPT_PATH.read_text()

        self.assertNotIn("import cv2", source)
        self.assertNotIn("cv2.", source)

    def test_prepare_output_paths_allows_existing_video_files_for_overwrite(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "left.mp4").write_bytes(b"existing")
            (output_dir / "right.mp4").write_bytes(b"existing")

            left_path, right_path = module.prepare_output_paths(output_dir)

            self.assertEqual(left_path, output_dir / "left.mp4")
            self.assertEqual(right_path, output_dir / "right.mp4")

    def test_build_preview_pipeline_uses_gstreamer_only_and_scales_to_requested_size(self):
        module = load_module()

        pipeline = module.build_preview_pipeline(
            "/dev/video11",
            3840,
            1080,
            60,
            1280,
            360,
            "waylandsink",
        )

        self.assertIn("v4l2src device=/dev/video11 do-timestamp=true", pipeline)
        self.assertIn("image/jpeg,width=3840,height=1080,framerate=60/1", pipeline)
        self.assertIn("mppjpegdec", pipeline)
        self.assertIn("videoscale", pipeline)
        self.assertIn("video/x-raw,width=1280,height=360", pipeline)
        self.assertIn("waylandsink sync=false", pipeline)

    def test_parse_args_sets_recording_defaults_and_allows_no_preview(self):
        module = load_module()

        default_args = module.parse_args([])
        no_preview_args = module.parse_args(["--no-preview"])

        self.assertTrue(default_args.preview)
        self.assertFalse(no_preview_args.preview)
        self.assertEqual(default_args.width, 1280)
        self.assertEqual(default_args.height, 480)
        self.assertEqual(default_args.fps, 60)
        self.assertEqual(default_args.bitrate, 40000000)
        self.assertIsNone(default_args.preview_width)
        self.assertIsNone(default_args.preview_height)
        self.assertEqual(default_args.video_sink, "waylandsink")

    def test_parse_args_allows_explicit_video_sink(self):
        module = load_module()

        args = module.parse_args(["--video-sink", "ximagesink"])

        self.assertEqual(args.video_sink, "ximagesink")

    def test_resolve_video_sink_uses_wayland_sink_for_wayland_sessions(self):
        module = load_module()

        sink = module.resolve_video_sink(
            "auto",
            env={"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"},
        )

        self.assertEqual(sink, "waylandsink")

    def test_resolve_video_sink_uses_autovideosink_for_non_wayland_sessions(self):
        module = load_module()

        sink = module.resolve_video_sink(
            "auto",
            env={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":10"},
        )

        self.assertEqual(sink, "autovideosink")

    def test_resolve_video_sink_preserves_explicit_sink(self):
        module = load_module()

        sink = module.resolve_video_sink(
            "gtkwaylandsink",
            env={"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"},
        )

        self.assertEqual(sink, "gtkwaylandsink")

    def test_fit_preview_size_preserves_aspect_and_limits_to_screen(self):
        module = load_module()

        self.assertEqual(module.fit_preview_size(3840, 1080, 1280, 720), (1280, 360))
        self.assertEqual(module.fit_preview_size(3840, 1080, 5000, 3000), (3840, 1080))

    def test_resolve_preview_size_uses_detected_screen_by_default(self):
        module = load_module()
        args = module.parse_args([])

        preview_size = module.resolve_preview_size(
            args,
            screen_size_func=lambda: (1280, 720),
        )

        self.assertEqual(preview_size, (1280, 480))

    def test_resolve_preview_size_allows_manual_override(self):
        module = load_module()
        args = module.parse_args(["--preview-width", "1024", "--preview-height", "288"])

        preview_size = module.resolve_preview_size(
            args,
            screen_size_func=lambda: (1280, 720),
        )

        self.assertEqual(preview_size, (1024, 288))

    def test_detect_screen_size_reads_primary_xrandr_resolution(self):
        module = load_module()

        class Result:
            stdout = """Screen 0: minimum 8 x 8, current 1280 x 720, maximum 32767 x 32767
HDMI-1 connected primary 1280x720+0+0 normal
DP-1 disconnected normal
"""

        def command_runner(command, **kwargs):
            self.assertEqual(command, ["xrandr", "--current"])
            return Result()

        self.assertEqual(module.detect_screen_size(command_runner=command_runner), (1280, 720))

    def test_terminal_key_reader_reads_keys_from_command_line(self):
        module = load_module()

        class FakeStream:
            def __init__(self):
                self.keys = ["e", ""]

            def isatty(self):
                return True

            def read(self, size):
                if size != 1:
                    raise AssertionError(size)
                return self.keys.pop(0)

        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                self.target()

        reader = module.TerminalKeyReader(
            stream=FakeStream(),
            thread_factory=ImmediateThread,
        )

        self.assertTrue(reader.start())
        self.assertEqual(reader.read_key(), "e")
        self.assertIsNone(reader.read_key())

    def test_parse_args_sets_hardware_encoder_bitrate(self):
        module = load_module()

        default_args = module.parse_args([])
        custom_args = module.parse_args(["--bitrate", "8000000"])

        self.assertEqual(default_args.bitrate, 40000000)
        self.assertEqual(custom_args.bitrate, 8000000)

    def test_build_recording_pipeline_uses_pure_gstreamer_hardware_path(self):
        module = load_module()

        pipeline = module.build_recording_pipeline(
            device="/dev/video11",
            width=3840,
            height=1080,
            fps=60,
            left_path=Path("/tmp/left.mp4"),
            right_path=Path("/tmp/right.mp4"),
            bitrate=12000000,
        )

        self.assertIn("v4l2src device=/dev/video11 do-timestamp=true", pipeline)
        self.assertIn("mppjpegdec", pipeline)
        self.assertIn("image/jpeg,width=3840,height=1080,framerate=60/1", pipeline)
        self.assertIn("video/x-raw,format=NV12,width=3840,height=1080", pipeline)
        self.assertIn("video/x-raw,format=NV12,width=1920,height=1080", pipeline)
        self.assertNotIn("video/x-raw,format=NV12,width=3840,height=1080,framerate=60/1", pipeline)
        self.assertNotIn("video/x-raw,format=NV12,width=1920,height=1080,framerate=60/1", pipeline)
        self.assertIn("videocrop right=1920", pipeline)
        self.assertIn("videocrop left=1920", pipeline)
        self.assertIn("mpph264enc level=42 bps=12000000", pipeline)
        self.assertIn("filesink location=/tmp/left.mp4", pipeline)
        self.assertIn("filesink location=/tmp/right.mp4", pipeline)
        self.assertNotIn("appsrc", pipeline)
        self.assertNotIn("autovideosink", pipeline)
        self.assertNotIn("videoscale", pipeline)

    def test_build_recording_pipeline_can_keep_preview_visible(self):
        module = load_module()

        pipeline = module.build_recording_pipeline(
            device="/dev/video11",
            width=3840,
            height=1080,
            fps=60,
            left_path=Path("/tmp/left.mp4"),
            right_path=Path("/tmp/right.mp4"),
            bitrate=12000000,
            preview_width=1280,
            preview_height=360,
            video_sink="waylandsink",
        )

        self.assertIn("video/x-raw,format=NV12,width=1920,height=1080", pipeline)
        self.assertIn("filesink location=/tmp/left.mp4", pipeline)
        self.assertIn("filesink location=/tmp/right.mp4", pipeline)
        self.assertIn("videoscale", pipeline)
        self.assertIn("video/x-raw,width=1280,height=360", pipeline)
        self.assertIn("waylandsink sync=false", pipeline)

    def test_build_recording_pipeline_for_args_does_not_include_preview_sink(self):
        module = load_module()
        args = module.parse_args([])

        pipeline = module.build_recording_pipeline_for_args(
            args,
            Path("/tmp/left.mp4"),
            Path("/tmp/right.mp4"),
            video_sink="waylandsink",
        )

        self.assertIn("filesink location=/tmp/left.mp4", pipeline)
        self.assertIn("filesink location=/tmp/right.mp4", pipeline)
        self.assertNotIn("waylandsink", pipeline)
        self.assertNotIn("videoscale", pipeline)

    def test_resolve_embedded_video_sink_uses_gtk_wayland_for_default_wayland_sink(self):
        module = load_module()

        self.assertEqual(module.resolve_embedded_video_sink("waylandsink"), "gtkwaylandsink")
        self.assertEqual(module.resolve_embedded_video_sink("gtkwaylandsink"), "gtkwaylandsink")
        self.assertEqual(module.resolve_embedded_video_sink("ximagesink"), "ximagesink")

    def test_main_uses_embedded_recorder_for_preview_instead_of_preview_process_restart(self):
        module = load_module()
        calls = []

        class FakeRecorder:
            def __init__(self, args, left_path, right_path, preview_enabled, video_sink):
                calls.append(
                    {
                        "args": args,
                        "left_path": left_path,
                        "right_path": right_path,
                        "preview_enabled": preview_enabled,
                        "video_sink": video_sink,
                    }
                )

            def run(self):
                return 0

        original_recorder = module.EmbeddedStereoRecorder
        original_display_available = module.display_available
        original_cbreak_stdin = module.cbreak_stdin
        original_stdin = sys.stdin
        try:
            module.EmbeddedStereoRecorder = FakeRecorder
            module.display_available = lambda: True
            module.cbreak_stdin = lambda: nullcontext()
            sys.stdin = type("FakeStdin", (), {"isatty": lambda self: False})()

            rc = module.main([])
        finally:
            module.EmbeddedStereoRecorder = original_recorder
            module.display_available = original_display_available
            module.cbreak_stdin = original_cbreak_stdin
            sys.stdin = original_stdin

        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["preview_enabled"])
        self.assertEqual(calls[0]["video_sink"], "gtkwaylandsink")

    def test_build_recording_command_uses_gst_launch_e(self):
        module = load_module()

        command = module.build_recording_command("fakesrc ! fakesink")

        self.assertEqual(command, "gst-launch-1.0 -q -e fakesrc ! fakesink")

    def test_start_gstreamer_process_does_not_share_terminal_streams(self):
        module = load_module()
        calls = []

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))

            class FakeProcess:
                pass

            return FakeProcess()

        original_popen = module.subprocess.Popen
        try:
            module.subprocess.Popen = fake_popen
            module.start_gstreamer_process("fakesrc ! fakesink")
        finally:
            module.subprocess.Popen = original_popen

        command, kwargs = calls[0]
        self.assertEqual(command, "gst-launch-1.0 -q -e fakesrc ! fakesink")
        self.assertIs(kwargs["stdin"], module.subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], module.subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], module.subprocess.DEVNULL)

    def test_stop_recording_process_sends_sigint_to_process_group_and_waits(self):
        module = load_module()
        calls = []

        class FakeProcess:
            pid = 1234

            def send_signal(self, sig):
                calls.append(("send_signal", sig))

            def wait(self, timeout):
                calls.append(("wait", timeout))
                return 0

        module.stop_recording_process(
            FakeProcess(),
            killpg_func=lambda pid, sig: calls.append(("killpg", pid, sig)),
        )

        self.assertEqual(calls, [("killpg", 1234, signal.SIGINT), ("wait", 5)])


if __name__ == "__main__":
    unittest.main()
