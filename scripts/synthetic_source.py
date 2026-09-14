#!/usr/bin/env python3
"""
synthetic_source.py - stand in for the DevKit, to test this UI on its own.

    python3 scripts/synthetic_source.py                  # both sources, forever
    python3 scripts/synthetic_source.py --seconds 30
    python3 scripts/synthetic_source.py --channels 0
    python3 scripts/synthetic_source.py --check           # report vf's counters

It sends, to the SAME Insight ingest sockets the DevKit sends to:

    udp <host>:9000   H.264 over RTP, payload type 96, 90 kHz   (channel 0)
    udp <host>:9001   H.264 over RTP, payload type 96, 90 kHz   (channel 1)
    udp <host>:9100   object-detection metadata JSON            (channel 0)
    udp <host>:9101   segmentation metadata JSON                (channel 1)

WHY IT EXISTS

Video continuity, overlay filtering and the video/metadata pairing all have to
be checked, and the DevKit pipeline is not always on the bench.  This produces
traffic in exactly the shapes the vision pipeline produces - same message types, same
field names, same chunk framing above 1200 bytes, and the same timestamp rule -
so the UI can be exercised and Insight's own correlation counters can be read
back as evidence.

THE PAIRING RULE, WHICH IS THE POINT

vision-local/src/stream_worker.cpp stamps each metadata message with the source
frame's millisecond and relies on

    metadata.timestamp * 90 == the frame's RTP timestamp

because rtph264pay runs with timestamp-offset=0.  Insight matches within 90
units (one millisecond).  This sender uses the identical rule, with the RTP
timestamp generated from the same millisecond, so `matched_metadata_first` in
GET /api/ingest/stats climbing is a direct measurement that pairing works.

It is a TEST TOOL.  It is never started by run.sh and nothing in the
application imports it.
"""

import argparse
import json
import os
import random
import socket
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
CACHE_DIR = os.path.join(APP_DIR, "runtime", "cache")

RTP_PAYLOAD_TYPE = 96
RTP_CLOCK = 90000
MTU_PAYLOAD = 1200                  # conservative; vf reassembles FU-A
FPS = 30
WIDTH, HEIGHT = 1280, 720

# The metadata chunk framing, as the Insight metadata ingest expects it,
# which read it out of core/src/nodes/io/MetadataSender.cpp.
CHUNK_MAGIC = (0x4E, 0x01)
CHUNK_HEADER_SIZE = 12
MAX_DATAGRAM_PAYLOAD = 1200
MAX_CHUNK_PAYLOAD = MAX_DATAGRAM_PAYLOAD - CHUNK_HEADER_SIZE

COCO_SAMPLE = ["person", "dog", "chair", "car", "bus", "cat", "bicycle",
               "bottle", "laptop", "cell phone"]


# --------------------------------------------------------------------------
# H.264 source
# --------------------------------------------------------------------------

def make_h264(path, seconds=8):
    """One short Annex-B H.264 clip, produced once and reused.

    baseline / one slice per frame / repeat-headers so SPS and PPS are in band
    before every IDR - which is what lets vf identify the codec and answer an
    /offer at all.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 10000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=%dx%d:rate=%d" % (WIDTH, HEIGHT, FPS),
        "-t", str(seconds),
        "-c:v", "libx264", "-profile:v", "baseline", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-g", str(FPS), "-bf", "0",
        "-x264-params", "slices=1:repeat-headers=1:scenecut=0",
        "-f", "h264", path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(path):
        raise SystemExit("ffmpeg could not build the test clip:\n%s"
                         % result.stderr.strip())
    return path


def split_nals(blob):
    """Annex-B -> [nal_bytes] without start codes."""
    out = []
    index = 0
    length = len(blob)
    start = None
    while index < length - 3:
        if blob[index] == 0 and blob[index + 1] == 0 and blob[index + 2] == 1:
            if start is not None:
                end = index
                if end > start and blob[end - 1] == 0:
                    end -= 1
                out.append(blob[start:end])
            start = index + 3
            index += 3
            continue
        index += 1
    if start is not None:
        out.append(blob[start:])
    return [nal for nal in out if nal]


def access_units(nals):
    """Group NALs into frames: parameter sets ride with the next picture."""
    units = []
    pending = []
    for nal in nals:
        nal_type = nal[0] & 0x1F
        pending.append(nal)
        if nal_type in (1, 5):            # a coded slice ends the access unit
            units.append(pending)
            pending = []
    if pending:
        units.append(pending)
    return units


# --------------------------------------------------------------------------
# RTP
# --------------------------------------------------------------------------

class RtpSender:
    def __init__(self, sock, address, ssrc=None):
        self.sock = sock
        self.address = address
        self.ssrc = ssrc if ssrc is not None else random.getrandbits(32)
        self.sequence = random.getrandbits(16)
        self.packets = 0

    def _header(self, timestamp, marker):
        first = (2 << 6)                                   # version 2
        second = (0x80 if marker else 0) | RTP_PAYLOAD_TYPE
        header = struct.pack("!BBHII", first, second,
                             self.sequence & 0xFFFF, timestamp & 0xFFFFFFFF,
                             self.ssrc)
        self.sequence = (self.sequence + 1) & 0xFFFF
        return header

    def send_access_unit(self, nals, timestamp):
        """RFC 6184: single-NAL packets, FU-A for anything over the MTU."""
        for position, nal in enumerate(nals):
            last_nal = position == len(nals) - 1
            if len(nal) <= MTU_PAYLOAD:
                self._emit(self._header(timestamp, last_nal) + nal)
                continue
            indicator = (nal[0] & 0xE0) | 28               # FU-A
            header_byte = nal[0] & 0x1F
            body = nal[1:]
            offset = 0
            chunk_size = MTU_PAYLOAD - 2
            while offset < len(body):
                chunk = body[offset:offset + chunk_size]
                start = offset == 0
                end = offset + chunk_size >= len(body)
                fu_header = ((0x80 if start else 0) | (0x40 if end else 0)
                             | header_byte)
                marker = last_nal and end
                self._emit(self._header(timestamp, marker)
                           + bytes([indicator, fu_header]) + chunk)
                offset += chunk_size

    def _emit(self, packet):
        self.sock.sendto(packet, self.address)
        self.packets += 1


# --------------------------------------------------------------------------
# Metadata, in the vision pipeline's exact shapes
# --------------------------------------------------------------------------

def build_datagrams(payload, message_id):
    """Frame a JSON payload the way core's MetadataSender does."""
    raw = payload.encode("utf-8")
    if len(raw) <= MAX_DATAGRAM_PAYLOAD:
        return [raw]
    count = (len(raw) + MAX_CHUNK_PAYLOAD - 1) // MAX_CHUNK_PAYLOAD
    if count > 255:
        raise ValueError("payload needs %d chunks; the framing allows 255" % count)
    out = []
    for index in range(count):
        chunk = raw[index * MAX_CHUNK_PAYLOAD:(index + 1) * MAX_CHUNK_PAYLOAD]
        header = bytes(CHUNK_MAGIC) + struct.pack(">Q", message_id) \
            + bytes([index, count])
        out.append(header + chunk)
    return out


def moving_box(frame, seed, width, height):
    """A box that drifts, so a filter's effect is obvious on screen."""
    phase = (frame * (1.7 + seed * 0.6)) % 360
    x = 60 + (seed * 150 + frame * (2 + seed)) % max(1, width - 320)
    y = 60 + int(120 * abs(((phase / 180.0) % 2) - 1)) + seed * 40
    w = 150 + seed * 20
    h = 220 - seed * 15
    return [float(int(x)), float(min(y, height - h - 10)), float(w), float(h)]


def detection_payload(frame, labels):
    objects = []
    for index, label in enumerate(labels):
        bbox = moving_box(frame, index, WIDTH, HEIGHT)
        objects.append({"id": str(index), "label": label,
                        "confidence": round(0.55 + 0.04 * index, 2),
                        "bbox": bbox})
    return {"objects": objects}


def segmentation_payload(frame, labels):
    segments = []
    for index, label in enumerate(labels):
        x, y, w, h = moving_box(frame, index, WIDTH, HEIGHT)
        # A polygon at a realistic vertex count, which is what pushes the
        # segmentation channel over 1200 bytes and into the chunk framing -
        # the same thing the real pipeline does.
        points = []
        steps = 28
        for step in range(steps):
            angle = (step / float(steps)) * 6.28318
            points.append([round(x + w / 2 + (w / 2) * 0.92 * _cos(angle), 1),
                           round(y + h / 2 + (h / 2) * 0.92 * _sin(angle), 1)])
        segments.append({"id": str(index), "label": label,
                         "confidence": round(0.50 + 0.05 * index, 2),
                         "bbox": [x, y, w, h],
                         "mask_format": "polygon", "mask": points})
    return {"segments": segments}


def _cos(angle):
    import math
    return math.cos(angle)


def _sin(angle):
    import math
    return math.sin(angle)


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------

class Channel:
    def __init__(self, index, host, video_port, metadata_port, units, task):
        self.index = index
        self.task = task
        self.units = units
        self.video_address = (host, video_port)
        self.metadata_address = (host, metadata_port)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rtp = RtpSender(self.video_socket, self.video_address)
        self.message_id = 1
        self.metadata_sent = 0
        self.labels_cycle = 0

    def labels_for(self, frame):
        # The visible set changes every ~4 s so a "person only" filter can be
        # seen taking effect, and so a class can genuinely be absent.
        bucket = (frame // (FPS * 4)) % 3
        if bucket == 0:
            return ["person", "dog", "chair", "car"]
        if bucket == 1:
            return ["person", "bus", "cat"]
        return ["dog", "bicycle", "bottle", "laptop", "cell phone"]

    def send_frame(self, frame, frame_ms):
        unit = self.units[frame % len(self.units)]
        # The one rule that makes overlays land on the right picture.
        self.rtp.send_access_unit(unit, frame_ms * 90)

        labels = self.labels_for(frame)
        if self.task == "segmentation":
            data = segmentation_payload(frame, labels)
            message_type = "segmentation"
        else:
            data = detection_payload(frame, labels)
            message_type = "object-detection"
        message = {"type": message_type, "timestamp": frame_ms,
                   "frame_id": str(frame), "data": data}
        blob = json.dumps(message, ensure_ascii=False)
        for datagram in build_datagrams(blob, self.message_id):
            self.metadata_socket.sendto(datagram, self.metadata_address)
        self.message_id += 1
        self.metadata_sent += 1

    def close(self):
        self.video_socket.close()
        self.metadata_socket.close()


def load_config():
    """The UI's own configuration, through the UI's own loader.

    So insight.host / api_port -> api_base is resolved exactly once, in
    backend/server.py, and this tool cannot disagree with the app about where
    Insight is.
    """
    sys.path.insert(0, os.path.join(APP_DIR, "backend"))
    import server
    return server.load_ui_config()


def check_counters(config, host):
    """Read vf's own view back, which is the evidence that matters."""
    import ssl
    from urllib import request as urlrequest
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    url = config["insight"]["api_base"].rstrip("/") + "/api/ingest/stats?all=1"
    with urlrequest.urlopen(url, timeout=8, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--host", default=os.environ.get("INSIGHT_HOST", "127.0.0.1"),
                        help="where Insight's vf is listening (default 127.0.0.1)")
    parser.add_argument("--channels", default="0,1")
    parser.add_argument("--seconds", type=float, default=0,
                        help="0 = run until interrupted")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--check", action="store_true",
                        help="print vf's ingest counters and exit")
    args = parser.parse_args()

    if args.check:
        stats = check_counters(config, args.host)
        for channel in stats.get("channels", []):
            if channel.get("channel") not in (0, 1):
                continue
            media = channel.get("media") or {}
            rtp = channel.get("rtp") or {}
            meta = channel.get("metadata") or {}
            print("channel %s  active=%s" % (channel["channel"], channel.get("active")))
            print("  video : codec=%s packets=%s bitrate=%s sps=%s pps=%s idr=%s"
                  % (media.get("codec"), rtp.get("packets_received"),
                     rtp.get("bitrate_bps"), media.get("seen_sps"),
                     media.get("seen_pps"), media.get("idr_count")))
            print("  meta  : recv=%s chunks=%s bad_json=%s"
                  % (meta.get("messages_received"),
                     meta.get("chunk_datagrams_received"),
                     meta.get("invalid_json")))
            print("  paired: video_first=%s metadata_first=%s expired=%s evicted=%s"
                  % (meta.get("matched_video_first"),
                     meta.get("matched_metadata_first"),
                     meta.get("expired_metadata"), meta.get("evicted_metadata")))
        return 0

    clip = make_h264(os.path.join(CACHE_DIR, "synthetic_%dx%d.h264"
                                  % (WIDTH, HEIGHT)))
    with open(clip, "rb") as handle:
        units = access_units(split_nals(handle.read()))
    if not units:
        raise SystemExit("no access units in %s" % clip)

    wanted = [int(part) for part in args.channels.split(",") if part.strip()]
    channels = []
    for index in wanted:
        entry = config["sources"].get(str(index))
        task = entry["task"] if entry else "detection"
        channels.append(Channel(
            index, args.host,
            config["insight"]["video_udp_base"] + index,
            config["insight"]["metadata_udp_base"] + index,
            units, task))

    print("synthetic source -> %s" % args.host)
    for channel in channels:
        print("  channel %d  %-13s video udp %d   metadata udp %d"
              % (channel.index, channel.task,
                 channel.video_address[1], channel.metadata_address[1]))
    print("  %d access units, %dx%d, %d fps, timestamp*90 = RTP timestamp"
          % (len(units), WIDTH, HEIGHT, args.fps))
    print("  Ctrl+C to stop")

    period = 1.0 / args.fps
    started = time.time()
    # The wall clock is the timeline, exactly as a camera's would be: a
    # pipeline that stamps its two branches from different clocks drifts out of
    # Insight's one-millisecond tolerance permanently.
    epoch_ms = int(started * 1000)
    frame = 0
    try:
        while True:
            now = time.time()
            if args.seconds and now - started >= args.seconds:
                break
            frame_ms = epoch_ms + int(frame * 1000.0 / args.fps)
            for channel in channels:
                channel.send_frame(frame, frame_ms)
            frame += 1
            target = started + frame * period
            delay = target - time.time()
            if delay > 0:
                time.sleep(delay)
            if frame % (args.fps * 5) == 0:
                print("  %d frames  %s"
                      % (frame, "  ".join(
                          "ch%d rtp=%d meta=%d" % (c.index, c.rtp.packets,
                                                   c.metadata_sent)
                          for c in channels)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        for channel in channels:
            channel.close()
    print("sent %d frames per channel" % frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
