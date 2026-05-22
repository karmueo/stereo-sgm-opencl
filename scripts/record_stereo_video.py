#!/usr/bin/env python3
"""Record a side-by-side stereo USB camera stream into left/right MP4 files.

使用说明:
    1. 确认相机设备和格式:
       v4l2-ctl --list-devices
       v4l2-ctl -d /dev/video11 --list-formats-ext

    2. 启动录制程序:
       python3 scripts/record_stereo_video.py --output-dir ./output/stereo_record

    3. 默认按键:
       s  开始录制
       e  结束录制
       q  退出程序

    4. 输出文件:
       <output-dir>/left.mp4
       <output-dir>/right.mp4

说明:
    默认采集 /dev/video11 的 3840x1080@60 MJPEG 双目拼接流。
    录制时由 GStreamer 在 NV12/硬件路径中解码、裁剪并同步写入 left.mp4 和 right.mp4。
    MP4 按 GStreamer 实际 buffer 时间戳封装，不在解码/编码端强制固定输出帧率。
    视频通过 Rockchip MPP JPEG/H.264 硬件编解码器保存，默认 H.264 码率为 40000000 bps。
    有图形显示环境时会弹出 GStreamer/GTK 实时预览窗口，默认参数等价于
    --video-sink waylandsink --width 3840 --height 1080 --fps 60 --bitrate 40000000。
    在 GNOME/Wayland 下，默认 waylandsink 会自动使用 gtkwaylandsink 嵌入 GTK 窗口，
    因此终端窗口和预览窗口都支持按键控制。
    开始/停止录制不会停止预览输出。
    保存视频保持左右各 1920x1080 原始分辨率。
    按键在启动脚本的终端窗口和预览窗口中都生效。
    如果没有 DISPLAY/WAYLAND_DISPLAY，会自动退回终端按键控制。
    如果输出目录中已存在 left.mp4 或 right.mp4，开始录制时会直接覆盖。
"""

import argparse
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import termios
import threading
import time
import tty


VIDEO_SINKS = (
    "auto",
    "autovideosink",
    "waylandsink",
    "gtkwaylandsink",
    "ximagesink",
    "xvimagesink",
    "rkximagesink",
    "kmssink",
)


def prepare_output_paths(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_path = output_dir / "left.mp4"
    right_path = output_dir / "right.mp4"

    return left_path, right_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Record a side-by-side stereo camera stream to left.mp4/right.mp4."
    )
    parser.add_argument("--device", default="/dev/video11", help="V4L2 camera device")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="directory for left.mp4 and right.mp4",
    )
    parser.add_argument("--start-key", default="s", help="key used to start recording")
    parser.add_argument("--stop-key", default="e", help="key used to stop recording")
    parser.add_argument("--fps", type=int, default=60, help="capture and output FPS")
    parser.add_argument("--width", type=int, default=3840, help="captured frame width")
    parser.add_argument("--height", type=int, default=1080, help="captured frame height")
    parser.add_argument("--bitrate", type=int, default=40000000, help="H.264 bitrate")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=None,
        help="preview window video width; defaults to fitting the current screen",
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=None,
        help="preview window video height; defaults to fitting the current screen",
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="disable the realtime preview window",
    )
    parser.add_argument(
        "--video-sink",
        choices=VIDEO_SINKS,
        default="waylandsink",
        help="GStreamer preview sink; use auto to choose by session type",
    )
    parser.set_defaults(preview=True)
    return parser.parse_args(argv)


@contextmanager
def cbreak_stdin():
    if not sys.stdin.isatty():
        raise RuntimeError("stdin must be a TTY for keyboard control")

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


class TerminalKeyReader:
    def __init__(self, stream=None, thread_factory=threading.Thread):
        self.stream = sys.stdin if stream is None else stream
        self.thread_factory = thread_factory
        self.thread = None
        self.keys = queue.Queue()

    def start(self):
        if not self.stream.isatty():
            return False

        self.thread = self.thread_factory(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        while True:
            try:
                key = self.stream.read(1)
            except (OSError, ValueError):
                return
            if not key:
                return
            self.keys.put(key)

    def read_key(self):
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None


def display_available(env=None, platform=None):
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux"):
        return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    return True


def resolve_video_sink(video_sink, env=None):
    if video_sink != "auto":
        return video_sink

    env = os.environ if env is None else env
    if env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE") == "wayland":
        return "waylandsink"

    return "autovideosink"


def resolve_embedded_video_sink(video_sink):
    if video_sink in ("auto", "waylandsink", "gtkwaylandsink"):
        return "gtkwaylandsink"
    return video_sink


def fit_preview_size(frame_width, frame_height, screen_width, screen_height):
    scale = min(screen_width / frame_width, screen_height / frame_height, 1.0)
    return max(1, int(frame_width * scale)), max(1, int(frame_height * scale))


def detect_screen_size(command_runner=subprocess.run):
    try:
        result = command_runner(
            ["xrandr", "--current"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    connected_sizes = []
    current_size = None
    for line in result.stdout.splitlines():
        current_match = re.search(r"\bcurrent\s+(\d+)\s+x\s+(\d+)", line)
        if current_match:
            current_size = (int(current_match.group(1)), int(current_match.group(2)))

        connected_match = re.search(r"\bconnected\b(?:\s+primary)?\s+(\d+)x(\d+)\+", line)
        if connected_match:
            size = (int(connected_match.group(1)), int(connected_match.group(2)))
            if " connected primary " in line:
                return size
            connected_sizes.append(size)

    if connected_sizes:
        return connected_sizes[0]
    return current_size


def resolve_preview_size(args, screen_size_func=detect_screen_size):
    if args.preview_width is not None and args.preview_height is not None:
        return args.preview_width, args.preview_height

    if args.preview_width is not None:
        return args.preview_width, max(1, int(args.preview_width * args.height / args.width))

    if args.preview_height is not None:
        return max(1, int(args.preview_height * args.width / args.height)), args.preview_height

    screen_size = screen_size_func()
    if screen_size is None:
        return args.width, args.height

    return fit_preview_size(args.width, args.height, *screen_size)


def h264_level_for(frame_size, fps):
    width, height = frame_size
    if width > 1920 or height > 1080:
        return 52
    if fps > 30:
        return 42
    return 40


def build_preview_pipeline(
    device,
    width,
    height,
    fps,
    preview_width,
    preview_height,
    video_sink="autovideosink",
):
    return (
        f"v4l2src device={shlex.quote(str(device))} do-timestamp=true "
        f"! image/jpeg,width={width},height={height},framerate={fps}/1 "
        "! mppjpegdec "
        "! videoscale "
        f"! video/x-raw,width={preview_width},height={preview_height} "
        "! videoconvert "
        f"! {video_sink} sync=false"
    )


def build_recording_pipeline(
    device,
    width,
    height,
    fps,
    left_path,
    right_path,
    bitrate,
    preview_width=None,
    preview_height=None,
    video_sink="autovideosink",
):
    half_width = width // 2
    level = h264_level_for((half_width, height), fps)
    left_location = shlex.quote(str(left_path))
    right_location = shlex.quote(str(right_path))
    pipeline = (
        f"v4l2src device={shlex.quote(str(device))} do-timestamp=true "
        f"! image/jpeg,width={width},height={height},framerate={fps}/1 "
        "! mppjpegdec "
        f"! video/x-raw,format=NV12,width={width},height={height} "
        "! tee name=t "
        f"t. ! queue ! videocrop right={half_width} "
        f"! video/x-raw,format=NV12,width={half_width},height={height} "
        f"! mpph264enc level={level} bps={bitrate} "
        "! h264parse ! mp4mux faststart=true "
        f"! filesink location={left_location} "
        f"t. ! queue ! videocrop left={half_width} "
        f"! video/x-raw,format=NV12,width={half_width},height={height} "
        f"! mpph264enc level={level} bps={bitrate} "
        "! h264parse ! mp4mux faststart=true "
        f"! filesink location={right_location}"
    )
    if preview_width is not None and preview_height is not None:
        pipeline += (
            " "
            "t. ! queue ! videoscale "
            f"! video/x-raw,width={preview_width},height={preview_height} "
            "! videoconvert "
            f"! {video_sink} sync=false"
        )
    return pipeline


def build_recording_pipeline_for_args(args, left_path, right_path, video_sink):
    return build_recording_pipeline(
        args.device,
        args.width,
        args.height,
        args.fps,
        left_path,
        right_path,
        args.bitrate,
        video_sink=video_sink,
    )


def build_recording_command(pipeline):
    return f"gst-launch-1.0 -q -e {pipeline}"


def start_gstreamer_process(pipeline, use_eos=True):
    eos_arg = "-e " if use_eos else ""
    return subprocess.Popen(
        f"gst-launch-1.0 -q {eos_arg}{pipeline}",
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )


def start_recording_process(pipeline):
    return start_gstreamer_process(pipeline, use_eos=True)


def start_preview_process(pipeline):
    return start_gstreamer_process(pipeline, use_eos=False)


def stop_recording_process(process, timeout=5, killpg_func=os.killpg):
    killpg_func(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        killpg_func(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            killpg_func(process.pid, signal.SIGKILL)
            return process.wait()


class EmbeddedStereoRecorder:
    def __init__(self, args, left_path, right_path, preview_enabled, video_sink):
        self.args = args
        self.left_path = Path(left_path)
        self.right_path = Path(right_path)
        self.preview_enabled = preview_enabled
        self.video_sink = video_sink
        self.preview_width = None
        self.preview_height = None
        self.pipeline = None
        self.tee = None
        self.loop = None
        self.bus = None
        self.window = None
        self.video_widget = None
        self.terminal_key_reader = TerminalKeyReader()
        self.recording_branches = []
        self.recording_stopping = False
        self.quit_after_stop = False
        self.stop_timeout_id = None
        self.Gst = None
        self.GLib = None
        self.Gtk = None
        self.Gdk = None

    def run(self):
        try:
            self._load_gi()
            self.Gst.init(None)
            if self.preview_enabled:
                ok, _ = self.Gtk.init_check(None)
                if not ok:
                    print(
                        "Preview disabled: GTK could not open a display; using terminal controls.",
                        flush=True,
                    )
                    self.preview_enabled = False

            self.loop = self.GLib.MainLoop()
            self._build_pipeline()
            self._setup_bus()
            self._setup_terminal_keys()
            self.pipeline.set_state(self.Gst.State.PLAYING)
            if self.window is not None:
                self.window.show_all()
                self.window.present()
                if self.video_widget is not None:
                    self.video_widget.grab_focus()
                else:
                    self.window.grab_focus()
            self.loop.run()
            return 0
        except KeyboardInterrupt:
            print("\ninterrupted", flush=True)
            return 0
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            if self.recording_branches:
                self._cleanup_recording_branches(force=True)
            if self.pipeline is not None:
                self.pipeline.set_state(self.Gst.State.NULL)
            if self.window is not None:
                self.window.destroy()

    def _load_gi(self):
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GLib", "2.0")
        from gi.repository import GLib, Gst

        self.GLib = GLib
        self.Gst = Gst

        if self.preview_enabled:
            gi.require_version("Gtk", "3.0")
            gi.require_version("Gdk", "3.0")
            from gi.repository import Gdk, Gtk

            self.Gdk = Gdk
            self.Gtk = Gtk

    def _make_element(self, factory_name, element_name):
        element = self.Gst.ElementFactory.make(factory_name, element_name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory_name}")
        return element

    def _link_many(self, *elements):
        for left, right in zip(elements, elements[1:]):
            if not left.link(right):
                raise RuntimeError(
                    f"failed to link {left.get_name()} -> {right.get_name()}"
                )

    def _build_pipeline(self):
        self.preview_width, self.preview_height = resolve_preview_size(self.args)
        self.pipeline = self.Gst.Pipeline.new("stereo-recorder")

        source = self._make_element("v4l2src", "camera")
        source.set_property("device", self.args.device)
        source.set_property("do-timestamp", True)

        input_caps = self._make_element("capsfilter", "input_caps")
        input_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"image/jpeg,width={self.args.width},height={self.args.height},"
                f"framerate={self.args.fps}/1"
            ),
        )

        decoder = self._make_element("mppjpegdec", "jpeg_decoder")
        raw_caps = self._make_element("capsfilter", "raw_caps")
        raw_caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=NV12,width={self.args.width},height={self.args.height}"
            ),
        )

        self.tee = self._make_element("tee", "video_tee")
        for element in (source, input_caps, decoder, raw_caps, self.tee):
            self.pipeline.add(element)
        self._link_many(source, input_caps, decoder, raw_caps, self.tee)
        self._add_preview_or_drain_branch()

    def _request_tee_pad(self):
        template = self.tee.get_pad_template("src_%u")
        return self.tee.request_pad(template, None, None)

    def _add_preview_or_drain_branch(self):
        queue_element = self._make_element("queue", "preview_queue")

        if self.preview_enabled:
            scale = self._make_element("videoscale", "preview_scale")
            caps = self._make_element("capsfilter", "preview_caps")
            caps.set_property(
                "caps",
                self.Gst.Caps.from_string(
                    f"video/x-raw,width={self.preview_width},height={self.preview_height}"
                ),
            )
            convert = self._make_element("videoconvert", "preview_convert")
            sink = self._make_element(self.video_sink, "preview_sink")
            if sink.find_property("sync") is not None:
                sink.set_property("sync", False)

            elements = (queue_element, scale, caps, convert, sink)
            for element in elements:
                self.pipeline.add(element)
            self._link_many(*elements)

            widget = sink.get_property("widget") if sink.find_property("widget") else None
            self._create_preview_window(widget)
            print(
                f"Preview: enabled (GTK/GStreamer, {self.preview_width}x{self.preview_height}, {self.video_sink})",
                flush=True,
            )
        else:
            sink = self._make_element("fakesink", "preview_drain")
            sink.set_property("sync", False)
            for element in (queue_element, sink):
                self.pipeline.add(element)
            self._link_many(queue_element, sink)

        tee_pad = self._request_tee_pad()
        sink_pad = queue_element.get_static_pad("sink")
        if tee_pad.link(sink_pad) != self.Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link preview branch")

    def _create_preview_window(self, video_widget):
        if self.Gtk is None:
            return

        self.window = self.Gtk.Window(title="Stereo Recorder")
        self.window.set_default_size(self.preview_width, self.preview_height)
        self.window.set_can_focus(True)
        self.window.add_events(self.Gdk.EventMask.KEY_PRESS_MASK)
        self.window.connect("key-press-event", self._on_window_key)
        self.window.connect("delete-event", self._on_window_delete)
        self.window.connect("destroy", self._on_window_destroy)

        if video_widget is not None:
            self.video_widget = video_widget
            video_widget.set_can_focus(True)
            video_widget.add_events(self.Gdk.EventMask.KEY_PRESS_MASK)
            video_widget.connect("key-press-event", self._on_window_key)
            self.window.add(video_widget)

    def _setup_bus(self):
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    def _setup_terminal_keys(self):
        if self.terminal_key_reader.start():
            self.GLib.timeout_add(20, self._poll_terminal_keys)

    def _poll_terminal_keys(self):
        while True:
            key = self.terminal_key_reader.read_key()
            if key is None:
                return True
            self._handle_key(key)

    def _on_window_key(self, _widget, event):
        key = event.string or self.Gdk.keyval_name(event.keyval) or ""
        self._handle_key(key)
        return True

    def _on_window_delete(self, _widget, _event):
        self._handle_key("q")
        return True

    def _on_window_destroy(self, _widget):
        if self.loop is not None and not self.recording_branches:
            self.loop.quit()

    def _on_bus_message(self, _bus, message):
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = f": {debug}" if debug else ""
            print(f"gstreamer error from {message.src.get_name()}: {error}{detail}", file=sys.stderr)
            self.loop.quit()
        elif message.type == self.Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            detail = f": {debug}" if debug else ""
            print(f"gstreamer warning from {message.src.get_name()}: {warning}{detail}", file=sys.stderr)

    def _handle_key(self, key):
        if not key:
            return
        key = key.lower()

        if key == self.args.start_key and not self.recording_branches:
            self.start_recording()
        elif key == self.args.stop_key and self.recording_branches:
            self.stop_recording()
        elif key == "q":
            if self.recording_branches:
                self.stop_recording(quit_after=True)
            elif self.loop is not None:
                self.loop.quit()

    def start_recording(self):
        if self.recording_stopping:
            return

        for path in (self.left_path, self.right_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        try:
            left_branch = self._create_recording_branch(
                "left_recording", crop_side="right", location=self.left_path
            )
            right_branch = self._create_recording_branch(
                "right_recording", crop_side="left", location=self.right_path
            )
            self._attach_recording_branch(left_branch)
            self._attach_recording_branch(right_branch)
        except Exception:
            self._cleanup_recording_branches(force=True)
            raise

        print("recording started", flush=True)

    def _create_recording_branch(self, name, crop_side, location):
        half_width = self.args.width // 2
        branch_bin = self.Gst.Bin.new(name)
        queue_element = self._make_element("queue", f"{name}_queue")
        crop = self._make_element("videocrop", f"{name}_crop")
        crop.set_property(crop_side, half_width)
        caps = self._make_element("capsfilter", f"{name}_caps")
        caps.set_property(
            "caps",
            self.Gst.Caps.from_string(
                f"video/x-raw,format=NV12,width={half_width},height={self.args.height}"
            ),
        )
        encoder = self._make_element("mpph264enc", f"{name}_encoder")
        encoder.set_property("level", h264_level_for((half_width, self.args.height), self.args.fps))
        encoder.set_property("bps", self.args.bitrate)
        parser = self._make_element("h264parse", f"{name}_parser")
        muxer = self._make_element("mp4mux", f"{name}_muxer")
        muxer.set_property("faststart", True)
        sink = self._make_element("filesink", f"{name}_sink")
        sink.set_property("location", str(location))

        elements = (queue_element, crop, caps, encoder, parser, muxer, sink)
        for element in elements:
            branch_bin.add(element)
        self._link_many(*elements)

        sink_pad = queue_element.get_static_pad("sink")
        ghost_pad = self.Gst.GhostPad.new("sink", sink_pad)
        branch_bin.add_pad(ghost_pad)

        branch = {
            "name": name,
            "bin": branch_bin,
            "sink_pad": ghost_pad,
            "tee_pad": None,
            "eos_seen": False,
        }

        def on_sink_event(_pad, info):
            event = info.get_event()
            if event.type == self.Gst.EventType.EOS:
                branch["eos_seen"] = True
                self.GLib.idle_add(self._finish_recording_if_ready)
            return self.Gst.PadProbeReturn.OK

        sink.get_static_pad("sink").add_probe(
            self.Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_sink_event,
        )
        return branch

    def _attach_recording_branch(self, branch):
        self.pipeline.add(branch["bin"])
        tee_pad = self._request_tee_pad()
        if tee_pad.link(branch["sink_pad"]) != self.Gst.PadLinkReturn.OK:
            self.pipeline.remove(branch["bin"])
            self.tee.release_request_pad(tee_pad)
            raise RuntimeError(f"failed to link {branch['name']}")
        branch["tee_pad"] = tee_pad
        branch["bin"].sync_state_with_parent()
        self.recording_branches.append(branch)

    def stop_recording(self, quit_after=False):
        if self.recording_stopping:
            self.quit_after_stop = self.quit_after_stop or quit_after
            return

        self.recording_stopping = True
        self.quit_after_stop = quit_after
        print("recording stopping...", flush=True)
        for branch in self.recording_branches:
            branch["sink_pad"].send_event(self.Gst.Event.new_eos())
        self.stop_timeout_id = self.GLib.timeout_add_seconds(
            5,
            self._force_finish_recording,
        )

    def _force_finish_recording(self):
        self.stop_timeout_id = None
        return self._finish_recording_if_ready(force=True)

    def _finish_recording_if_ready(self, force=False):
        if not self.recording_stopping:
            return False
        if not force and not all(branch["eos_seen"] for branch in self.recording_branches):
            return False

        self._cleanup_recording_branches(force=force)
        self.recording_stopping = False
        print("recording stopped", flush=True)
        if self.quit_after_stop and self.loop is not None:
            self.loop.quit()
        self.quit_after_stop = False
        return False

    def _cleanup_recording_branches(self, force=False):
        if self.stop_timeout_id is not None:
            self.GLib.source_remove(self.stop_timeout_id)
            self.stop_timeout_id = None

        for branch in self.recording_branches:
            branch["bin"].set_state(self.Gst.State.NULL)
            if branch["tee_pad"] is not None:
                branch["tee_pad"].unlink(branch["sink_pad"])
                self.tee.release_request_pad(branch["tee_pad"])
            self.pipeline.remove(branch["bin"])
        self.recording_branches = []
        if force:
            self.recording_stopping = False


def main(argv=None):
    args = parse_args(argv)

    try:
        left_path, right_path = prepare_output_paths(args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    preview_enabled = args.preview and display_available()
    video_sink = resolve_video_sink(args.video_sink)
    if preview_enabled:
        video_sink = resolve_embedded_video_sink(video_sink)

    print(f"Camera: {args.device} ({args.width}x{args.height}@{args.fps})")
    print(f"Output: {left_path} and {right_path}")
    print(
        f"Controls: {args.start_key}=start, {args.stop_key}=stop, q=quit",
        flush=True,
    )

    if not preview_enabled and args.preview:
        print(
            "Preview disabled: no DISPLAY/WAYLAND_DISPLAY; using terminal controls.",
            flush=True,
        )
    elif not args.preview:
        print("Preview disabled by --no-preview; using terminal controls.", flush=True)

    try:
        control_context = cbreak_stdin() if sys.stdin.isatty() else nullcontext()
        with control_context:
            recorder = EmbeddedStereoRecorder(
                args,
                left_path,
                right_path,
                preview_enabled=preview_enabled,
                video_sink=video_sink,
            )
            return recorder.run()
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("exited", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
