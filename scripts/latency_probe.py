#!/usr/bin/env python3
"""
latency_probe.py - WHERE the frame gets old, per stage, without guessing.

    runtime/wrtcvenv/bin/python scripts/latency_probe.py --channel 0 --seconds 30
    runtime/wrtcvenv/bin/python scripts/latency_probe.py --channel 1 --seconds 30 \
        --json runtime/forensics/latency-ch1.json

THE PROBLEM WITH MEASURING THIS

The frame rate is fine (~20 fps of genuinely new pictures at every stage -
scripts/fps_forensics.py and scripts/webrtc_probe.py) and the picture is about
a second old.  A frame's age can only be computed against the moment it was
captured, and the capture clock is not directly readable from outside the
board: the pipeline stamps every frame with `frame_ms`, a SYNTHETIC timeline
(`stream_worker.cpp stamp_for()`), whose epoch shifts by the length of every
camera restart (`reanchor` sets `ms_base = last_frame_ms + 1`).  So
`now - frame_ms` is not an age, and anything built on it would be wrong by
however long the cameras have been stopped since boot.

WHAT THIS MEASURES INSTEAD - AND WHY IT NEEDS NO EPOCH

The pipeline sends each frame down TWO independent paths that carry the SAME
capture stamp (`stream_worker.cpp`, the loop):

    metadata branch   inference output -> its own UDP socket, IMMEDIATELY
    video branch      push_video() -> appsrc -> neatencoder -> rtph264pay
                      -> udpsink

Both arrive at this process, whose clock is the same for both.  Therefore

    video_arrival(frame) - metadata_arrival(frame)

is the time the frame spent in the VIDEO path after inference: the appsrc
queue, the encoder, the payloader and the sink.  No capture clock is needed,
because the two paths share the same capture stamp and the same receiver
clock.  The two are matched on that stamp: the video frame's RTP timestamp
(aiortc exposes it as `frame.pts`, 90 kHz) is `frame_ms * 90`, and the
metadata message carries `timestamp = frame_ms` and `frame_id`.

It also measures, from the same run:

    metadata inter-arrival      the pipeline's real loop period (capture ->
                                inference -> send), so inference cannot hide
    video inter-arrival         the encoder's output cadence
    per-frame jitter            how steady each path is
    vf residency                with --ingest, how long Insight held the frame
                                between its UDP socket and this peer
    frame_id continuity         whether frames are dropped or reordered

WHAT IT CANNOT SEE

Capture -> inference (the camera buffer and the model), because that is
upstream of both branches and upstream of the only shared stamp.  It is
bounded separately: the metadata inter-arrival IS the loop period, and the
loop is synchronous - pull, infer, send - so capture -> metadata-send cannot
exceed one loop period plus one camera frame interval unless the camera itself
queues, and the camera Output is built with
`nn::OutputOptions::Latest()` (drop=true, max_buffers=1, "lowest latency").
"""

import argparse
import asyncio
import json
import os
import ssl
import statistics
import sys
import time
import urllib.request

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def post_offer(base, channel, sdp, sdp_type):
    body = json.dumps({"sdp": sdp, "type": sdp_type}).encode("utf-8")
    request = urllib.request.Request(
        "%s/offer?channel=%d" % (base.rstrip("/"), channel), data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=20, context=CTX) as reply:
        return json.load(reply)


def align_timelines(keys, targets, tolerance=1, window=6000):
    """The constant shift that maps `keys` onto `targets`.

    Both sides are in the same clock and describe the same frames, whose
    spacing wanders (33 or 67 ms at the source), so the spacing sequence is a
    fingerprint and exactly one shift aligns them.  Returns (shift, hits) and
    refuses - (None, hits) - unless it matches at least half the keys, so a
    bad alignment is visible instead of producing a plausible wrong number.
    """
    if not keys or not targets:
        return None, 0
    base = min(targets) - min(keys)
    best = (None, 0)
    for delta in range(base - window, base + window + 1, max(1, tolerance)):
        hits = 0
        for key in keys:
            shifted = key + delta
            if (shifted in targets or shifted - tolerance in targets
                    or shifted + tolerance in targets):
                hits += 1
        if hits > best[1]:
            best = (delta, hits)
    if best[1] < max(8, len(keys) // 2):
        return None, best[1]
    return best


def summarize(name, values, unit="ms"):
    if not values:
        return "%-34s (no samples)" % name
    ordered = sorted(values)

    def pct(fraction):
        return ordered[min(len(ordered) - 1,
                           max(0, int(round(fraction * (len(ordered) - 1)))))]
    return ("%-34s n=%-5d min %8.1f  median %8.1f  p90 %8.1f  max %8.1f %s"
            % (name, len(ordered), ordered[0], pct(0.5), pct(0.9),
               ordered[-1], unit))


async def run(base, channel, seconds, ingest_url, out_json):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    # frame_ms -> arrival monotonic seconds, for each path
    video_at = {}
    video_pts_at = {}          # raw 90 kHz pts -> arrival
    meta_at = {}
    meta_frame_id = {}
    video_order = []
    meta_order = []
    done = asyncio.Event()

    data_channel = pc.createDataChannel("metadata", ordered=True,
                                        maxRetransmits=0)

    # frame_ms -> vf's OUTPUT rtp timestamp for that frame, taken from the
    # metadata vf itself annotated.  vf re-stamps the forwarded stream onto its
    # own clock, so this is the only bridge between the pipeline's frame_ms and
    # what a WebRTC peer sees.
    output_rtp = {}

    @data_channel.on("message")
    def on_message(message):
        now = time.monotonic()
        try:
            payload = json.loads(message if isinstance(message, str)
                                 else message.decode("utf-8", "replace"))
        except (ValueError, AttributeError):
            return
        stamp = payload.get("timestamp")
        insight = payload.get("_insight") or {}
        # `timestamp` is frame_ms; `_insight.rtp_timestamp` is frame_ms * 90 and
        # is what the video frame carries, so either identifies the frame.
        if stamp is None and "rtp_timestamp" in insight:
            stamp = insight["rtp_timestamp"] / 90.0
        if stamp is None:
            return
        key = int(round(float(stamp)))
        if key not in meta_at:
            meta_at[key] = now
            meta_frame_id[key] = payload.get("frame_id")
            meta_order.append(key)
            if "rtp_timestamp" in insight:
                output_rtp[key] = int(insight["rtp_timestamp"])

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def consume():
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("  (no video frame for 10 s)")
                    done.set()
                    return
                except Exception:                                # noqa: BLE001
                    done.set()
                    return
                now = time.monotonic()
                if frame.pts is None:
                    continue
                # aiortc's video time_base is 1/90000, so pts IS the RTP
                # timestamp, and frame_ms = pts / 90.
                if frame.pts not in video_pts_at:
                    video_pts_at[frame.pts] = now

        asyncio.ensure_future(consume())

    # Insight's own view, polled fast, so "when did vf receive this frame" is a
    # measurement rather than an assumption.  Each sample pairs the wall clock
    # of the newest packet with its RTP timestamp.
    ingest_samples = []

    async def poll_ingest():
        if not ingest_url:
            return
        loop = asyncio.get_event_loop()

        def fetch():
            with urllib.request.urlopen(ingest_url, timeout=3,
                                        context=CTX) as reply:
                return json.load(reply), time.monotonic()
        while not done.is_set():
            try:
                data, at = await loop.run_in_executor(None, fetch)
            except Exception:                                    # noqa: BLE001
                await asyncio.sleep(0.2)
                continue
            for entry in data.get("channels", []):
                if entry.get("channel") != channel:
                    continue
                rtp = entry.get("rtp") or {}
                stamp = rtp.get("last_timestamp")
                # `last_timestamp` is the SOURCE clock (frame_ms * 90): vf
                # reports what it received, before its own re-stamping.  So
                # this keys directly on frame_ms, no alignment needed.
                if stamp is not None:
                    ingest_samples.append((int(round(stamp / 90.0)), at))
            await asyncio.sleep(0.02)

    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    answer = post_offer(base, channel, pc.localDescription.sdp,
                        pc.localDescription.type)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    poller = asyncio.ensure_future(poll_ingest())
    started = time.monotonic()
    try:
        await asyncio.wait_for(done.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    elapsed = time.monotonic() - started
    done.set()
    poller.cancel()
    await pc.close()

    # ---- align the video track to the pipeline's frame_ms ----------------
    #
    # Two renumberings stand between them.  vf re-stamps the forwarded stream
    # onto its own output clock (measured: a metadata message whose own
    # `timestamp` is 12539170 ms carries `_insight.rtp_timestamp` 1597773518,
    # i.e. 17753039 ms - not timestamp*90), and aiortc then renumbers that
    # track to start at zero.  The bridge is the metadata itself: vf annotates
    # every message with the OUTPUT rtp timestamp of the video frame it paired
    # it with, so `output_rtp` maps frame_ms -> output clock, and only one
    # constant shift is left to find.
    targets = set(output_rtp.values())
    offset, score = align_timelines(sorted(video_pts_at), targets,
                                    tolerance=45, window=15 * 90000)
    rtp_to_frame_ms = {rtp: ms for ms, rtp in output_rtp.items()}
    aligned_video_at = {}
    if offset is not None:
        for pts, at in video_pts_at.items():
            absolute = pts + offset
            for candidate in (absolute, absolute - 45, absolute + 45,
                              absolute - 1, absolute + 1):
                if candidate in rtp_to_frame_ms:
                    aligned_video_at[rtp_to_frame_ms[candidate]] = at
                    break
    video_at = aligned_video_at
    video_order = sorted(video_at)

    # ---- the measurement ------------------------------------------------
    shared = []
    lag_ms = []
    for key in sorted(video_at):
        for candidate in (key, key - 1, key + 1):
            if candidate in meta_at:
                shared.append(candidate)
                lag_ms.append((video_at[key] - meta_at[candidate]) * 1000.0)
                break
    meta_gaps = [(meta_at[meta_order[i]] - meta_at[meta_order[i - 1]]) * 1000.0
                 for i in range(1, len(meta_order))]
    video_gaps = [(video_at[video_order[i]] - video_at[video_order[i - 1]])
                  * 1000.0 for i in range(1, len(video_order))]
    # The capture-stamp spacing of the frames that actually arrived.
    stamp_gaps = [video_order[i] - video_order[i - 1]
                  for i in range(1, len(video_order))]

    # ---- Insight / WebRTC residency --------------------------------------
    #
    # Keyed on frame_ms, which both ends report in the SOURCE clock: vf's
    # `rtp.last_timestamp` is what it received (before its own re-stamping),
    # and the metadata message carries the same `timestamp`.  vf pairs a
    # metadata message with its video frame BEFORE forwarding either
    # (`matched_metadata_first` in /api/ingest/stats), so the metadata's
    # arrival here is the moment vf released that frame to the peer.
    #
    # Bounded below by the 20 ms poll interval, and by construction this is an
    # upper bound: vf may have received the frame just after a poll.
    vf_ms = []
    if ingest_samples:
        vf_first = {}
        for key, at in ingest_samples:
            if key not in vf_first:
                vf_first[key] = at
        for key in sorted(set(vf_first) & set(meta_at)):
            vf_ms.append((meta_at[key] - vf_first[key]) * 1000.0)

    print("=" * 72)
    print("  channel %d, %.1f s   video frames %d   metadata %d   matched %d"
          % (channel, elapsed, len(video_pts_at), len(meta_at), len(shared)))
    print("  video/metadata alignment: %s"
          % ("offset %d (90 kHz), %d frames matched" % (offset, score)
             if offset is not None else "FAILED - video stages not reported"))
    print("=" * 72)
    print("")
    print("  THE ONE THAT NEEDS NO CAPTURE CLOCK")
    print("  " + summarize("video arrival - metadata arrival", lag_ms))
    print("     Both carry the same capture stamp. The metadata is sent")
    print("     immediately after inference; the video goes through appsrc ->")
    print("     neatencoder -> rtph264pay -> udpsink. So this IS the age the")
    print("     video path adds after inference.")
    print("")
    print("  CADENCE (a queue does not change the rate, only the age)")
    print("  " + summarize("metadata inter-arrival = loop period", meta_gaps))
    print("  " + summarize("video inter-arrival", video_gaps))
    if stamp_gaps:
        print("  %-34s median %d ms  (capture-stamp spacing of arrived frames)"
              % ("frame spacing at the source",
                 statistics.median(stamp_gaps)))
    if vf_ms:
        print("")
        print("  INSIGHT / WEBRTC")
        print("  " + summarize("vf receive -> peer (paired)", vf_ms))
        print("     /api/ingest/stats polled at 50 Hz: bounded below by the")
        print("     20 ms poll interval, and an upper bound on vf's residency.")
    print("")
    if lag_ms:
        median_lag = statistics.median(lag_ms)
        period = statistics.median(meta_gaps) if meta_gaps else 50.0
        print("  READING IT: the video path holds a frame for %.0f ms, which at"
              % median_lag)
        print("  a %.0f ms loop period is %.1f frames standing in a queue."
              % (period, median_lag / period if period else 0))
        print("  A queue changes the AGE, not the RATE, which is why the frame")
        print("  rate looked healthy at every stage.")
    print("")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump({
                "channel": channel, "seconds": elapsed,
                "video_minus_metadata_ms": lag_ms,
                "metadata_gaps_ms": meta_gaps,
                "video_gaps_ms": video_gaps,
                "vf_to_peer_ms": vf_ms,
                "frame_ids": {str(k): meta_frame_id.get(k) for k in shared},
            }, handle, indent=1)
        print("  written to %s" % out_json)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base", default=None)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--ingest", default="https://127.0.0.1:9900/api/ingest/stats",
                        help="Insight's ingest stats, polled at 50 Hz; "
                             "empty string disables it")
    parser.add_argument("--json", dest="out_json", default=None)
    args = parser.parse_args()

    base = args.base
    if base is None:
        app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(app, "config", "ui_config.json"),
                  encoding="utf-8") as handle:
            base = "https://127.0.0.1:%d" % int(
                json.load(handle)["web"]["port"])
    return asyncio.new_event_loop().run_until_complete(
        run(base, args.channel, args.seconds, args.ingest or None,
            args.out_json))


if __name__ == "__main__":
    sys.exit(main())
