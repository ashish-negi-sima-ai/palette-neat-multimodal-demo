#!/usr/bin/env python3
"""
fps_forensics.py - where does the frame rate actually go?

    # on the DevKit (the sender), needs root for the raw socket:
    sudo /usr/bin/python3 scripts/fps_forensics.py --capture 10 --channel 0 \
         --out runtime/test/forensics
    # anywhere, on what the capture produced:
    /usr/bin/python3 scripts/fps_forensics.py --analyze runtime/forensics/ch0

WHY THIS EXISTS

The UI's "19 fps" is `RTCInboundRtpStreamStats.framesPerSecond` - the number of
frames the BROWSER DECODER finished in the last second.  A stream of twenty
identical frames per second reads exactly the same as a stream of twenty
different ones, so that number cannot answer "is the picture updating?".

This measures the stages the browser statistic cannot see, on the wire, at the
sender:

    encoded frame rate        RTP marker bits per second (one per frame)
    encoded frame sizes       bytes between marker bits.  A re-encoded
                              IDENTICAL 1080p frame costs a few hundred bytes
                              (skip macroblocks); a frame of real motion costs
                              tens of kilobytes.  This is what separates
                              "20 fps of new pixels" from "1 new frame, sent
                              twenty times".
    RTP timestamp deltas      the presentation clock the receiver obeys.  If
                              these advance faster than wall clock, the
                              receiver decodes every frame and still shows
                              them slowly - which looks exactly like the
                              reported fault.
    unique decoded frames     the capture is reassembled into an Annex-B
                              stream, decoded with ffmpeg and hashed frame by
                              frame.  Identical consecutive hashes are
                              duplicate PICTURES, whatever the frame rate says.

HOW THE CAPTURE WORKS

An AF_PACKET socket on the sending interface, filtered in Python to UDP
packets whose destination port is the channel's video port.  No tcpdump, no
tshark (neither is installed on either machine), and nothing is injected into
the running pipeline: this is a passive read of packets that are being sent
anyway.  The pipeline is not restarted, reconfigured or signalled.

WHAT IT DOES NOT MEASURE

The browser.  Decoded-vs-presented can only be measured in the browser, and
`window.demoFrameRates` in web/app.js is where that lives - it counts
requestVideoFrameCallback firings, which is one per frame actually presented.
"""

import argparse
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time

ETH_P_ALL = 0x0003
# H.264 RTP payload types this pipeline uses (rtph264pay default, and what
# /api/ingest/stats reports as payload_type).
DEFAULT_PAYLOAD_TYPES = (96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                         107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                         117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def parse_ipv4(frame):
    """(src, dst, ip_id, proto, frag_offset, more_fragments, payload) or None.

    IP fragmentation matters here: a segmentation metadata datagram is several
    kilobytes, so the kernel fragments it, and only the FIRST fragment carries
    the UDP header.  The later fragments are where the JSON's `timestamp`
    field ends up, because the masks come before it in the message.  Ignoring
    them cost channel 1 almost every sample (measured: 1 of ~300).
    """
    if len(frame) < 14:
        return None
    etype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    if etype == 0x8100:
        if len(frame) < 18:
            return None
        etype = struct.unpack("!H", frame[16:18])[0]
        offset = 18
    if etype != 0x0800 or len(frame) < offset + 20:
        return None
    ihl = (frame[offset] & 0x0F) * 4
    total_length = struct.unpack("!H", frame[offset + 2:offset + 4])[0]
    ip_id = struct.unpack("!H", frame[offset + 4:offset + 6])[0]
    flags_frag = struct.unpack("!H", frame[offset + 6:offset + 8])[0]
    more_fragments = bool(flags_frag & 0x2000)
    frag_offset = (flags_frag & 0x1FFF) * 8
    proto = frame[offset + 9]
    src = frame[offset + 12:offset + 16]
    dst = frame[offset + 16:offset + 20]
    end = offset + total_length if total_length else len(frame)
    payload = frame[offset + ihl:min(end, len(frame))]
    return src, dst, ip_id, proto, frag_offset, more_fragments, payload


def parse_udp(frame):
    """(dst_port, payload) for an IPv4/UDP ethernet frame, else None."""
    if len(frame) < 14:
        return None
    etype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    if etype == 0x8100:                      # VLAN tag
        if len(frame) < 18:
            return None
        etype = struct.unpack("!H", frame[16:18])[0]
        offset = 18
    if etype != 0x0800:                      # not IPv4
        return None
    if len(frame) < offset + 20:
        return None
    ihl = (frame[offset] & 0x0F) * 4
    if frame[offset + 9] != 17:               # not UDP
        return None
    udp = offset + ihl
    if len(frame) < udp + 8:
        return None
    dst_port = struct.unpack("!H", frame[udp + 2:udp + 4])[0]
    length = struct.unpack("!H", frame[udp + 4:udp + 6])[0]
    payload = frame[udp + 8:udp + max(8, length)]
    return dst_port, payload


def rtp_header(payload):
    """(pt, seq, timestamp, marker, header_len) or None."""
    if len(payload) < 12 or (payload[0] >> 6) != 2:
        return None
    csrc = payload[0] & 0x0F
    extension = (payload[0] >> 4) & 0x01
    marker = (payload[1] >> 7) & 0x01
    pt = payload[1] & 0x7F
    seq, ts = struct.unpack("!HI", payload[2:8])
    head = 12 + 4 * csrc
    if extension:
        if len(payload) < head + 4:
            return None
        words = struct.unpack("!H", payload[head + 2:head + 4])[0]
        head += 4 + 4 * words
    return pt, seq, ts, marker, head


# Segmentation metadata does not always fit one datagram, and the SDK chunks
# it (neat-core MetadataSender::send_raw_json): a 12-byte header of magic
# 0x4e 0x01, an 8-byte big-endian message id, then the chunk index and the
# chunk count.  Chunk 0 carries the start of the JSON, which is where the
# `timestamp` field is.  Without this, channel 1 loses almost every message
# (measured: 1 of ~300 parsed) because every fragment fails json.loads.
CHUNK_MAGIC = b"\x4e\x01"
CHUNK_HEADER_SIZE = 12
_STAMP_RE = re.compile(br'"timestamp"\s*:\s*(-?\d+)')


_FRAME_ID_RE = re.compile(br'"frame_id"\s*:\s*"?(-?\d+)"?')


def metadata_frame_id(payload):
    """The pipeline's own frame counter, for cross-checking with its log."""
    match = _FRAME_ID_RE.search(payload)
    return int(match.group(1)) if match else None


def reassemble(pending, payload, now):
    """(complete_message_bytes, first_chunk_time) once every chunk is in.

    An unchunked datagram is complete on arrival.  A chunked one is held in
    `pending` until all `count` chunks have been seen; `pending` is bounded so
    a lost chunk cannot grow it without limit.
    """
    if payload[:2] != CHUNK_MAGIC or len(payload) <= CHUNK_HEADER_SIZE:
        return payload, now
    message_id = int.from_bytes(payload[2:10], "big")
    index = payload[10]
    count = payload[11] or 1
    entry = pending.get(message_id)
    if entry is None:
        entry = [now, {}, count]
        pending[message_id] = entry
        if len(pending) > 64:
            oldest = min(pending, key=lambda key: pending[key][0])
            del pending[oldest]
    entry[1][index] = payload[CHUNK_HEADER_SIZE:]
    if len(entry[1]) < entry[2]:
        return None, None
    del pending[message_id]
    return b"".join(entry[1][i] for i in sorted(entry[1])), entry[0]


def metadata_stamp(payload):
    """The `timestamp` (frame_ms) in a complete metadata message."""
    match = _STAMP_RE.search(payload)
    return int(match.group(1)) if match else None


def capture_pair(interface, video_port, meta_port, seconds, out_dir):
    """Both of a channel's UDP streams, captured at the sender on ONE clock.

    This is the measurement that locates latency without needing the capture
    clock at all.  The pipeline sends every frame down two independent paths
    that carry the SAME capture stamp (`stream_worker.cpp`, the loop):

        metadata   inference output -> its own UDP socket, IMMEDIATELY
        video      push_video() -> appsrc -> neatencoder -> rtph264pay
                   -> udpsink

    Captured here, on the sending interface, both are timestamped by the same
    clock, so for one frame

        video_send_time - metadata_send_time

    is exactly what the video path added after inference: appsrc queue +
    encoder + payloader + sink.  The frame is identified by its stamp, which
    both paths carry - the metadata JSON as `timestamp` (ms) and `frame_id`,
    the video as its RTP timestamp (= timestamp * 90, verified against the
    pipeline's own log line `t=...ms rtp=...`).

    This cannot be done at the receiving end: vf pairs metadata to video
    BEFORE forwarding either (`matched_metadata_first` in
    /api/ingest/stats), and re-stamps both onto its own output clock, so a
    WebRTC peer sees them arrive together whatever the sender did.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                             socket.htons(ETH_P_ALL))
    except PermissionError:
        sys.exit("a raw socket needs root: run this with sudo on the sender")
    sock.bind((interface, 0))
    sock.settimeout(1.0)

    video_frames = {}      # frame_ms -> (first_packet_at, last_packet_at, bytes)
    meta_frames = {}       # frame_ms -> (at, frame_id, bytes)
    # message id -> [first_chunk_at, {index: bytes}, chunk_count]
    pending = {}
    started = time.time()
    deadline = started + seconds
    while time.time() < deadline:
        try:
            data = sock.recv(65535)
        except socket.timeout:
            continue
        parsed = parse_udp(data)
        if parsed is None:
            continue
        port, payload = parsed
        now = time.time()

        if port == meta_port:
            # Two framings on this port.  A detection message fits one
            # datagram.  A segmentation message does not, and the SDK chunks
            # it ITSELF (MetadataSender::send_raw_json): magic 0x4e 0x01, an
            # 8-byte big-endian message id, the chunk index and the chunk
            # count, then a slice of the JSON.  The `timestamp` field sits
            # near the END of the message (the key order on the wire is data,
            # frame_id, timestamp, type), so it arrives in the LAST chunk -
            # which is why reading only chunk 0 found 1 message in 300.
            #
            # The time recorded is the FIRST chunk's, because that is when the
            # pipeline handed the message to the socket.
            body, first_at = reassemble(pending, payload, now)
            if body is None:
                continue
            stamp = metadata_stamp(body)
            if stamp is None:
                continue
            key = int(round(float(stamp)))
            if key not in meta_frames:
                meta_frames[key] = [first_at, metadata_frame_id(body),
                                    len(body)]
            continue

        if port == video_port:
            header = rtp_header(payload)
            if header is None:
                continue
            pt, _seq, ts, marker, head = header
            if pt not in DEFAULT_PAYLOAD_TYPES:
                continue
            key = ts // 90
            entry = video_frames.get(key)
            size = len(payload) - head
            if entry is None:
                video_frames[key] = [now, now, size, bool(marker)]
            else:
                entry[1] = now
                entry[2] += size
                entry[3] = entry[3] or bool(marker)

    sock.close()
    elapsed = time.time() - started

    shared = sorted(set(video_frames) & set(meta_frames))
    record = {
        "interface": interface, "video_port": video_port,
        "metadata_port": meta_port, "seconds": elapsed,
        "video_frames": len(video_frames), "metadata_messages": len(meta_frames),
        "matched": len(shared),
        "pairs": [{
            "frame_ms": key,
            "frame_id": meta_frames[key][1],
            "metadata_at": round(meta_frames[key][0] - started, 6),
            "video_first_packet_at": round(video_frames[key][0] - started, 6),
            "video_last_packet_at": round(video_frames[key][1] - started, 6),
            "video_bytes": video_frames[key][2],
            "lag_ms": round((video_frames[key][0] - meta_frames[key][0]) * 1000.0, 3),
        } for key in shared],
    }
    path = os.path.join(out_dir, "pairs.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    print("captured %.1f s at the sender: %d video frames, %d metadata "
          "messages, %d matched" % (elapsed, len(video_frames),
                                    len(meta_frames), len(shared)))
    print("  %s" % path)
    analyze_pairs(record)
    return record


def report_drops(record):
    """Did the video path DROP frames, and at what rate did each stage run?

    The metadata is sent once per frame the loop consumed and inferred, so its
    count is the INPUT rate.  The video frames on the wire are the OUTPUT rate.
    With a non-leaky queue the two are equal and latency accumulates instead;
    with `leaky=downstream` a full queue discards its oldest buffer, and the
    difference between the two counts is that discard, measured rather than
    inferred.

    Edge effect, stated rather than hidden: the video lags the metadata by
    most of a second, so the first video frames in the window belong to
    metadata sent before it opened, and the last metadata has no video yet.
    Those two truncations are of the same size and cancel in the totals as
    long as the lag is stable, which the per-frame lag figures above show.
    """
    seconds = record["seconds"] or 1.0
    video = record["video_frames"]
    meta = record["metadata_messages"]
    dropped = meta - video
    print("  STAGE RATES AND DROPS over %.1f s" % seconds)
    print("    input  (inferred frames, metadata sent) : %4d  = %5.2f /s"
          % (meta, meta / seconds))
    print("    output (encoded frames on the wire)     : %4d  = %5.2f /s"
          % (video, video / seconds))
    print("    difference (video path discards)        : %4d  = %5.2f /s"
          " = %+.1f%%"
          % (dropped, dropped / seconds,
             (100.0 * dropped / meta) if meta else 0.0))
    if dropped > 0:
        print("      positive: frames were inferred but never reached the wire")
    elif dropped < 0:
        print("      negative: more video than metadata in this window, which")
        print("      is an edge effect of the lag, not a gain of frames")
    else:
        print("      zero: nothing was dropped - every inferred frame was sent")
    print("")


def analyze_pairs(record):
    pairs = record["pairs"]
    if not pairs:
        print("  nothing matched: the metadata and video stamps did not agree")
        return
    lags = sorted(p["lag_ms"] for p in pairs)
    stamps = [p["frame_ms"] for p in pairs]
    meta_at = [p["metadata_at"] for p in pairs]
    video_at = [p["video_first_packet_at"] for p in pairs]

    def pct(values, fraction):
        if not values:
            return float("nan")
        return values[min(len(values) - 1,
                          max(0, int(round(fraction * (len(values) - 1)))))]

    print("")
    print("=" * 72)
    print("  SENDER-SIDE LATENCY, both paths on one clock")
    print("=" * 72)
    print("  video send - metadata send   min %8.1f  median %8.1f  "
          "p90 %8.1f  max %8.1f ms"
          % (lags[0], pct(lags, 0.5), pct(lags, 0.9), lags[-1]))
    print("     = appsrc queue + neatencoder + rtph264pay + udpsink,")
    print("       for the same frame, after inference.")
    print("")
    meta_gaps = sorted((meta_at[i] - meta_at[i - 1]) * 1000.0
                       for i in range(1, len(meta_at)))
    video_gaps = sorted((video_at[i] - video_at[i - 1]) * 1000.0
                        for i in range(1, len(video_at)))
    stamp_gaps = sorted(stamps[i] - stamps[i - 1]
                        for i in range(1, len(stamps)))
    print("  metadata send cadence       median %8.1f ms  (the loop period:"
          " pull + inference)" % pct(meta_gaps, 0.5))
    print("  video send cadence          median %8.1f ms"
          % pct(video_gaps, 0.5))
    print("  capture-stamp spacing       median %8.1f ms  mean %6.1f ms  "
          "(the frames the loop consumed)"
          % (pct(stamp_gaps, 0.5),
             sum(stamp_gaps) / len(stamp_gaps) if stamp_gaps else float("nan")))
    print("     Bimodal on purpose: the sensor runs at 30 fps (33 ms) and the")
    print("     loop takes the newest frame each time, so consecutive frames")
    print("     are 33 or 67 ms apart. The MEAN is the rate; the median is not.")
    period = pct(meta_gaps, 0.5) or 50.0
    print("")
    report_drops(record)
    print("  %.0f ms of video-path lag at a %.0f ms loop period is %.1f frames"
          % (pct(lags, 0.5), period, pct(lags, 0.5) / period))
    print("  standing in the video path at any moment.")
    print("")


def capture(interface, port, seconds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                             socket.htons(ETH_P_ALL))
    except PermissionError:
        sys.exit("a raw socket needs root: run this with sudo on the sender")
    sock.bind((interface, 0))
    sock.settimeout(1.0)

    packets = []            # (wall_clock, seq, ts, marker, payload)
    started = time.time()
    deadline = started + seconds
    while time.time() < deadline:
        try:
            data = sock.recv(65535)
        except socket.timeout:
            continue
        parsed = parse_udp(data)
        if parsed is None or parsed[0] != port:
            continue
        header = rtp_header(parsed[1])
        if header is None:
            continue
        pt, seq, ts, marker, head = header
        if pt not in DEFAULT_PAYLOAD_TYPES:
            continue
        packets.append((time.time(), seq, ts, marker, parsed[1][head:]))
    sock.close()
    elapsed = time.time() - started

    if not packets:
        sys.exit("no RTP seen on %s udp/%d in %.1f s - wrong interface or the "
                 "pipeline is not sending" % (interface, port, seconds))

    # ---- frames: a frame ends at the packet whose marker bit is set --------
    frames = []
    current = {"bytes": 0, "packets": 0, "ts": packets[0][2],
               "first_at": packets[0][0]}
    nal_units = []                  # (frame_index, nal_bytes) in order
    pending_fu = None
    losses = 0
    previous_seq = None

    for at, seq, ts, marker, payload in packets:
        if previous_seq is not None:
            gap = (seq - previous_seq) & 0xFFFF
            if gap != 1:
                losses += (gap - 1) if gap < 1000 else 1
        previous_seq = seq

        current["bytes"] += len(payload)
        current["packets"] += 1
        current["ts"] = ts
        current["last_at"] = at

        # de-packetize: single NAL, STAP-A and FU-A are what rtph264pay emits
        if payload:
            kind = payload[0] & 0x1F
            if kind == 28:                              # FU-A
                if len(payload) >= 2:
                    start = payload[1] & 0x80
                    end = payload[1] & 0x40
                    nal_type = payload[1] & 0x1F
                    if start:
                        pending_fu = bytearray([(payload[0] & 0xE0) | nal_type])
                        pending_fu += payload[2:]
                    elif pending_fu is not None:
                        pending_fu += payload[2:]
                    if end and pending_fu is not None:
                        nal_units.append((len(frames), bytes(pending_fu)))
                        pending_fu = None
            elif kind == 24:                            # STAP-A
                cursor = 1
                while cursor + 2 <= len(payload):
                    size = struct.unpack("!H", payload[cursor:cursor + 2])[0]
                    cursor += 2
                    if size <= 0 or cursor + size > len(payload):
                        break
                    nal_units.append((len(frames), payload[cursor:cursor + size]))
                    cursor += size
            elif 1 <= kind <= 23:                       # single NAL
                nal_units.append((len(frames), bytes(payload)))

        if marker:
            frames.append(current)
            current = {"bytes": 0, "packets": 0, "ts": ts, "first_at": at}

    record = {
        "interface": interface, "port": port,
        "seconds": elapsed,
        "packets": len(packets),
        "sequence_losses": losses,
        "frames": [{"bytes": f["bytes"], "packets": f["packets"],
                    "rtp_ts": f["ts"],
                    "at": round(f["first_at"] - started, 6)}
                   for f in frames],
    }
    meta_path = os.path.join(out_dir, "frames.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)

    # Annex-B for ffmpeg/ffprobe.
    h264_path = os.path.join(out_dir, "capture.h264")
    with open(h264_path, "wb") as handle:
        for _index, nal in nal_units:
            handle.write(b"\x00\x00\x00\x01" + nal)

    print("captured %.1f s: %d packets, %d complete frames, %d NAL units"
          % (elapsed, len(packets), len(frames), len(nal_units)))
    print("  %s" % meta_path)
    print("  %s" % h264_path)
    if losses:
        print("  NOTE: %d sequence-number gaps (capture loss or real loss)"
              % losses)
    return out_dir


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def unique_frames(h264_path, limit_seconds=None):
    """Decode and hash every picture: how many are new, how many repeat.

    ffmpeg's framehash muxer writes one line per decoded frame with a checksum
    of the decoded pixels.  Two consecutive identical checksums mean the
    DECODER produced the same picture twice - a duplicate, whatever the frame
    rate claimed.
    """
    if not os.path.isfile(h264_path):
        return None
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "h264", "-i", h264_path,
               "-map", "0:v", "-f", "framehash", "-hash", "md5", "-"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": "ffmpeg failed: %s" % exc}
    hashes = []
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            hashes.append(parts[-1].strip())
    if not hashes:
        return {"error": "ffmpeg decoded no frames: %s"
                         % result.stderr.decode("utf-8", "replace")[:400]}
    repeats = sum(1 for i in range(1, len(hashes)) if hashes[i] == hashes[i - 1])
    distinct_runs = 1 + sum(1 for i in range(1, len(hashes))
                            if hashes[i] != hashes[i - 1])
    return {"decoded": len(hashes),
            "consecutive_repeats": repeats,
            "distinct_pictures": len(set(hashes)),
            "distinct_runs": distinct_runs}


def analyze(directory):
    meta_path = os.path.join(directory, "frames.json")
    with open(meta_path, encoding="utf-8") as handle:
        record = json.load(handle)
    frames = record["frames"]
    seconds = record["seconds"]
    sizes = [f["bytes"] for f in frames]
    print("=" * 68)
    print("  udp/%d on %s, %.2f s of capture"
          % (record["port"], record["interface"], seconds))
    print("=" * 68)
    print("")
    print("3. ENCODED FRAME RATE (RTP marker bits)")
    print("   %d frames / %.2f s = %.1f fps"
          % (len(frames), seconds, len(frames) / seconds))
    print("   %d packets = %.1f packets per frame"
          % (record["packets"], record["packets"] / max(1, len(frames))))
    if record["sequence_losses"]:
        print("   %d sequence gaps" % record["sequence_losses"])
    print("")
    print("2. ENCODED FRAME SIZES - the duplicate-frame test")
    print("   min %d B   p10 %d B   median %d B   p90 %d B   max %d B"
          % (min(sizes), percentile(sizes, 0.10), percentile(sizes, 0.50),
             percentile(sizes, 0.90), max(sizes)))
    tiny = [s for s in sizes if s < 1500]
    print("   frames under 1500 B (a re-encoded identical picture): %d of %d"
          " (%.0f%%)" % (len(tiny), len(sizes), 100.0 * len(tiny) / len(sizes)))
    print("   total %.2f Mbit over %.2f s = %.2f Mbit/s"
          % (sum(sizes) * 8 / 1e6, seconds, sum(sizes) * 8 / 1e6 / seconds))
    print("")
    print("   RTP TIMESTAMP PROGRESSION (the receiver's presentation clock)")
    deltas = [frames[i]["rtp_ts"] - frames[i - 1]["rtp_ts"]
              for i in range(1, len(frames))]
    deltas = [d for d in deltas if 0 < d < 90000 * 5]
    if deltas:
        print("   median %d units = %.1f ms between frames -> %.1f fps implied"
              % (percentile(deltas, 0.5), percentile(deltas, 0.5) / 90.0,
                 90000.0 / max(1, percentile(deltas, 0.5))))
        span = frames[-1]["rtp_ts"] - frames[0]["rtp_ts"]
        print("   span %d units = %.2f s of presentation time in %.2f s of wall"
              " clock  (ratio %.3f)"
              % (span, span / 90000.0, seconds, (span / 90000.0) / seconds))
        print("   a ratio above 1.0 means the receiver is told to show the")
        print("   frames more slowly than they arrive.")
    print("")
    print("2b. UNIQUE DECODED PICTURES (ffmpeg decode + per-frame md5)")
    uniqueness = unique_frames(os.path.join(directory, "capture.h264"))
    if uniqueness is None:
        print("   no capture.h264 in %s" % directory)
    elif "error" in uniqueness:
        print("   %s" % uniqueness["error"])
    else:
        decoded = uniqueness["decoded"]
        runs = uniqueness["distinct_runs"]
        print("   decoded pictures        : %d  (%.1f /s)"
              % (decoded, decoded / seconds))
        print("   consecutive duplicates  : %d"
              % uniqueness["consecutive_repeats"])
        print("   NEW pictures            : %d  (%.1f /s)   <-- the real"
              " visual frame rate" % (runs, runs / seconds))
        print("   distinct pictures total : %d" % uniqueness["distinct_pictures"])
    print("")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--capture", type=float, metavar="SECONDS",
                        help="capture for this long (needs root, on the sender)")
    parser.add_argument("--channel", type=int, default=0,
                        help="0 or 1; video port is 9000 + channel")
    parser.add_argument("--port", type=int, help="override the video port")
    parser.add_argument("--interface", default="end0",
                        help="the sending interface (default end0, the DevKit's)")
    parser.add_argument("--out", default=None, help="directory for the capture")
    parser.add_argument("--analyze", metavar="DIR",
                        help="analyze a capture directory")
    parser.add_argument("--pair", type=float, metavar="SECONDS",
                        help="capture BOTH the video and metadata ports of a "
                             "channel and report the sender-side latency "
                             "between them (needs root, on the sender)")
    parser.add_argument("--metadata-port", type=int,
                        help="override the metadata port (default 9100+channel)")
    args = parser.parse_args()

    if args.analyze:
        return analyze(args.analyze)
    if args.pair:
        video_port = args.port or (9000 + args.channel)
        meta_port = args.metadata_port or (9100 + args.channel)
        out = args.out or os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "runtime", "forensics")
        directory = os.path.join(out, "pair-ch%d" % args.channel)
        capture_pair(args.interface, video_port, meta_port, args.pair,
                     directory)
        return 0
    if args.capture:
        port = args.port or (9000 + args.channel)
        out = args.out or os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "runtime", "forensics")
        directory = os.path.join(out, "ch%d" % args.channel)
        capture(args.interface, port, args.capture, directory)
        return analyze(directory)
    parser.error("give --capture SECONDS, --pair SECONDS (both on the "
                 "sender) or --analyze DIR")


if __name__ == "__main__":
    sys.exit(main())
