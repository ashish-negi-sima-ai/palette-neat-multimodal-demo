#!/usr/bin/env python3
"""
webrtc_probe.py - what a viewer actually RECEIVES, measured without a browser.

    runtime/wrtcvenv/bin/python scripts/webrtc_probe.py --channel 0 --seconds 15
    runtime/wrtcvenv/bin/python scripts/webrtc_probe.py --channel 1 --seconds 15 \
        --save runtime/forensics/wrtc-ch1

This is a real WebRTC peer: it POSTs an offer to this app's `/offer?channel=N`
exactly as `web/webrtc.js` does, so vf treats it as the channel's viewer,
negotiates the same H.264 track and opens the same `metadata` data channel.

WHY IT EXISTS

The UI's fps figure is `RTCInboundRtpStreamStats.framesPerSecond`, which is
"frames the decoder finished in the last second".  It cannot distinguish twenty
new pictures a second from one picture sent twenty times, and it says nothing
about how often the picture is PRESENTED.  The sender side can be measured on
the wire (`scripts/fps_forensics.py`); this measures the other end of the same
stream:

    4. frames delivered over WebRTC   RTP frames arriving at a real peer
    5. frames decoded                 handed to the decoder and decoded here
    2b/6. NEW pictures                per-frame hash of the decoded image;
                                      consecutive identical hashes are
                                      duplicates and are NOT new pictures
    8. metadata messages              on the data channel, with how many carry
                                      a new frame_id and how many repeat

A browser's own compositor step (presented frames) is the one thing this cannot
see; `window.demoFrameRates` in the page reports that from
requestVideoFrameCallback.

IT TAKES THE CHANNEL OVER WHILE IT RUNS
vf attaches a channel's video to ONE peer, so running this displaces a browser
viewing the same channel for as long as it runs (measured and documented in the
README). It restores nothing and changes nothing on the server; close it and
the browser's own recovery re-offers.
"""

import argparse
import asyncio
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.request

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration


def post_offer(base, channel, sdp, sdp_type):
    body = json.dumps({"sdp": sdp, "type": sdp_type}).encode("utf-8")
    request = urllib.request.Request(
        "%s/offer?channel=%d" % (base.rstrip("/"), channel), data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(request, timeout=20, context=context) as reply:
        return json.load(reply)


async def run(base, channel, seconds, save_dir, save_metadata=None):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    stats = {
        "frames": 0,              # decoded pictures
        "new_pictures": 0,        # decoded pictures whose content changed
        "repeats": 0,
        "first_frame_at": None,
        "last_frame_at": None,
        "arrival_gaps_ms": [],
        "pts": [],
        "metadata": 0,
        "metadata_new_frame_id": 0,
        "metadata_repeat_frame_id": 0,
        "metadata_first_at": None,
        "metadata_last_at": None,
        "metadata_objects": [],
        "metadata_bytes": 0,
        "metadata_sizes": [],
        "unique_object_sets": 0,
        "raw_messages": [],
        "saved": 0,
    }
    done = asyncio.Event()

    channel_obj = pc.createDataChannel("metadata", ordered=True,
                                       maxRetransmits=0)
    last_frame_id = [None]
    last_object_signature = [None]

    @channel_obj.on("message")
    def on_message(message):
        now = time.time()
        stats["metadata"] += 1
        size = len(message if isinstance(message, (bytes, bytearray))
                   else message.encode("utf-8", "replace"))
        stats["metadata_bytes"] += size
        stats["metadata_sizes"].append(size)
        if stats["metadata_first_at"] is None:
            stats["metadata_first_at"] = now
        stats["metadata_last_at"] = now
        try:
            payload = json.loads(message if isinstance(message, str)
                                 else message.decode("utf-8", "replace"))
        except (ValueError, AttributeError):
            return
        data = payload.get("data") or {}
        frame_id = payload.get("frame_id", data.get("frame_id"))
        if frame_id is not None:
            if frame_id == last_frame_id[0]:
                stats["metadata_repeat_frame_id"] += 1
            else:
                stats["metadata_new_frame_id"] += 1
            last_frame_id[0] = frame_id
        entries = data.get("objects")
        if entries is None:
            entries = data.get("segments") or []
        stats["metadata_objects"].append(len(entries))
        if save_metadata and len(stats["raw_messages"]) < 5:
            stats["raw_messages"].append(payload)
        # Does the INFERENCE OUTPUT change?  Identical input pixels give
        # identical boxes, so a repeated signature means a repeated picture
        # reached the model.
        signature = json.dumps(
            [[e.get("label"), [round(v, 2) for v in (e.get("bbox") or [])]]
             for e in entries], sort_keys=True)
        if signature != last_object_signature[0]:
            stats["unique_object_sets"] += 1
        last_object_signature[0] = signature

    @pc.on("track")
    def on_track(track):
        if track.kind != "video":
            return

        async def consume():
            last_hash = None
            last_at = None
            saved = 0
            while True:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("  (no frame for 10 s - the track went silent)")
                    done.set()
                    return
                except Exception as exc:                      # noqa: BLE001
                    print("  (track ended: %s)" % exc)
                    done.set()
                    return
                now = time.time()
                stats["frames"] += 1
                if stats["first_frame_at"] is None:
                    stats["first_frame_at"] = now
                else:
                    stats["arrival_gaps_ms"].append((now - last_at) * 1000.0)
                stats["last_frame_at"] = now
                last_at = now
                if frame.pts is not None:
                    stats["pts"].append(frame.pts)
                # The decoded luma plane, hashed directly.  `to_ndarray`
                # would need numpy in this venv for no benefit: the plane's
                # buffer IS the decoded pixels, and identical buffers are
                # identical pictures.
                digest = hashlib.md5(bytes(frame.planes[0])).hexdigest()
                if digest == last_hash:
                    stats["repeats"] += 1
                else:
                    stats["new_pictures"] += 1
                    # Best effort, and deliberately not fatal: a measurement
                    # must not be lost because a thumbnail could not be
                    # written.  (It cost two runs to learn that.)
                    if save_dir and saved < 6:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            frame.to_image().save(
                                os.path.join(save_dir, "new-%02d.png" % saved))
                            saved += 1
                            stats["saved"] = saved
                        except Exception as exc:              # noqa: BLE001
                            if saved == 0:
                                print("  (cannot save pictures to %s: %s)"
                                      % (save_dir, exc))
                            saved = 6
                last_hash = digest

        asyncio.ensure_future(consume())

    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    answer = post_offer(base, channel, pc.localDescription.sdp,
                        pc.localDescription.type)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    started = time.time()
    try:
        await asyncio.wait_for(done.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
    elapsed = time.time() - started
    await pc.close()

    print("=" * 68)
    print("  WebRTC peer on channel %d, %.2f s" % (channel, elapsed))
    print("=" * 68)
    video_span = ((stats["last_frame_at"] - stats["first_frame_at"])
                  if stats["first_frame_at"] and stats["last_frame_at"] else 0)
    print("")
    print("4/5. FRAMES DELIVERED AND DECODED")
    # The rate BETWEEN the first and last frame is the stream's rate.  Dividing
    # by the whole window would charge the stream for ICE/DTLS setup and for
    # however long the track outlived it, which is how a healthy 19.6 fps reads
    # as 16.5.
    if video_span:
        print("   %.1f fps  (%d frames across the %.2f s between the first and"
              " last one)" % ((stats["frames"] - 1) / video_span,
                              stats["frames"], video_span))
        print("   %.1f fps if the %.2f s window is used instead, which also"
              " counts connection setup"
              % (stats["frames"] / max(0.001, elapsed), elapsed))
    else:
        print("   %d frames in %.2f s = %.1f fps"
              % (stats["frames"], elapsed,
                 stats["frames"] / max(0.001, elapsed)))
    gaps = sorted(stats["arrival_gaps_ms"])
    if gaps:
        def pct(f):
            return gaps[min(len(gaps) - 1, int(round(f * (len(gaps) - 1))))]
        print("   arrival gaps: min %.0f ms  median %.0f ms  p90 %.0f ms  "
              "max %.0f ms" % (gaps[0], pct(0.5), pct(0.9), gaps[-1]))
        long_gaps = [g for g in gaps if g > 200]
        print("   gaps over 200 ms: %d  (a gap of ~1000 ms is a 1 fps picture)"
              % len(long_gaps))
    print("")
    print("6. NEW PICTURES (md5 of every decoded frame)")
    print("   new      : %d  (%.1f /s)   <-- the real visual frame rate"
          % (stats["new_pictures"],
             stats["new_pictures"] / max(0.001, video_span or elapsed)))
    print("   duplicate: %d" % stats["repeats"])
    print("")
    print("8. METADATA ON THE DATA CHANNEL")
    print("   messages : %d  (%.1f /s)"
          % (stats["metadata"], stats["metadata"] / max(0.001, elapsed)))
    print("   new frame_id      : %d" % stats["metadata_new_frame_id"])
    print("   repeated frame_id : %d" % stats["metadata_repeat_frame_id"])
    print("   changed box sets  : %d  (identical pixels give identical boxes)"
          % stats["unique_object_sets"])
    if stats["metadata_objects"]:
        print("   objects per message: min %d  max %d"
              % (min(stats["metadata_objects"]),
                 max(stats["metadata_objects"])))
    if stats["metadata_sizes"]:
        sizes = sorted(stats["metadata_sizes"])
        print("   message size: min %d B  median %d B  max %d B"
              % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))
        print("   data channel: %.2f MB in %.1f s = %.2f Mbit/s of JSON the"
              " page must parse AND draw per frame"
              % (stats["metadata_bytes"] / 1e6, elapsed,
                 stats["metadata_bytes"] * 8 / 1e6 / max(0.001, elapsed)))
    if save_dir and stats["new_pictures"] and stats.get("saved"):
        print("")
        print("   sample pictures written to %s" % save_dir)
    if save_metadata and stats["raw_messages"]:
        with open(save_metadata, "w", encoding="utf-8") as handle:
            json.dump(stats["raw_messages"], handle, ensure_ascii=False,
                      indent=1)
        print("   %d raw messages written to %s"
              % (len(stats["raw_messages"]), save_metadata))
    print("")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--base", default=None,
                        help="the UI base URL (default: this project's own "
                             "configured port on localhost)")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--save", default=None,
                        help="write the first few NEW pictures as PNG here")
    parser.add_argument("--save-metadata", default=None,
                        help="write the first few raw metadata messages to "
                             "this JSON file, for offline render timing")
    args = parser.parse_args()

    base = args.base
    if base is None:
        app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(app, "config", "ui_config.json"),
                  encoding="utf-8") as handle:
            base = "https://127.0.0.1:%d" % int(
                json.load(handle)["web"]["port"])
    return asyncio.new_event_loop().run_until_complete(
        run(base, args.channel, args.seconds, args.save, args.save_metadata))


if __name__ == "__main__":
    sys.exit(main())
