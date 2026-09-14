#!/usr/bin/env python3
"""
insight_client.py - READ-ONLY access to the installed Neat Insight.

    python3 insight_client.py --probe

Nothing in this module writes to, configures, restarts or reconfigures Insight.
It does exactly three things, all of them GETs and one POST that Insight's own
viewer makes on every page load:

    drawing_js()     GET  <vf>/static/drawing.js
                     The proven overlay renderer, served through us so the
                     browser gets the renderer of the INSTALLED Insight and
                     cannot serve a stale cached copy (a known failure mode:
                     a browser can hold an old drawing.js for days).
    offer()          POST <vf>/offer?channel=N
                     WebRTC SIGNALLING ONLY - one small SDP round trip, exactly
                     what static/viewer-react.js does.  The media that follows
                     never passes through this process: it goes straight from
                     vf's WebRTC ports to the browser.
    ingest_stats()   GET  <api>/api/ingest/stats     diagnostics
    health()         GET  <api>/api/health

WHY THE SDP GOES THROUGH HERE AT ALL

The demo page is served from this app's own origin.  Proxying the one signalling
request means the browser needs to trust one certificate instead of two and
there is no cross-origin question to answer.  It is 2 KB of text per panel, once
per connection, and it carries no media.

TLS

vf and the Insight API are served with the SDK's own certificate
(/sdk-cert/neat-sdk.pem, per ~/.insight-config/neat-port-map.json).  These calls
go to 127.0.0.1 over the loopback interface, where the certificate's subject
does not match and there is nothing on the path to intercept, so verification is
disabled for them deliberately and only for them.
"""

import argparse
import json
import ssl
import sys
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_API_BASE = "https://127.0.0.1:9900"
DEFAULT_VF_BASE = "https://127.0.0.1:8081"

# vf answers 503 until enough RTP has arrived to identify the channel's codec.
# Its own viewer retries that; 415 means the browser has no decoder and is
# permanent, so it is passed straight back and never retried.
RETRYABLE_OFFER_STATUS = (503,)


class InsightError(Exception):
    """Insight or vf could not be reached, or answered an error."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _loopback_context():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


class InsightClient:
    def __init__(self, api_base=DEFAULT_API_BASE, vf_base=DEFAULT_VF_BASE,
                 timeout=8.0):
        self.api_base = api_base.rstrip("/")
        self.vf_base = vf_base.rstrip("/")
        self.timeout = timeout
        self._context = _loopback_context()
        self._lock = threading.Lock()
        self._drawing_js = None
        self._drawing_js_at = 0.0

    # -- raw helpers -------------------------------------------------------

    def _get(self, base, path, timeout=None):
        url = base + path
        try:
            with urlrequest.urlopen(url, timeout=timeout or self.timeout,
                                    context=self._context) as resp:
                return resp.read(), resp.status
        except urlerror.HTTPError as exc:
            raise InsightError("GET %s -> HTTP %s" % (url, exc.code),
                               exc.code) from exc
        except Exception as exc:                                # noqa: BLE001
            raise InsightError("GET %s -> %s" % (url, exc)) from exc

    def _get_json(self, base, path, timeout=None):
        body, _status = self._get(base, path, timeout)
        return json.loads(body.decode("utf-8"))

    # -- the four public calls --------------------------------------------

    def health(self):
        return self._get_json(self.api_base, "/api/health", timeout=4.0)

    def ingest_stats(self, query="?all=1"):
        return self._get_json(self.api_base, "/api/ingest/stats" + query)

    def drawing_js(self, max_age_s=30.0):
        """The installed Insight's overlay renderer, briefly cached.

        Cached so a page reload is not a round trip, and re-read often enough
        that reinstalling Insight does not leave this app serving an old
        renderer.  A failure is raised, never papered over with a copy of our
        own: a silently substituted renderer is exactly the kind of divergence
        this app exists to avoid.
        """
        now = time.monotonic()
        with self._lock:
            if self._drawing_js is not None and now - self._drawing_js_at < max_age_s:
                return self._drawing_js
        body, _status = self._get(self.vf_base, "/static/drawing.js")
        with self._lock:
            self._drawing_js = body
            self._drawing_js_at = now
        return body

    def offer(self, channel, sdp_json_bytes, timeout=20.0):
        """One WebRTC SDP round trip for `channel`.  Returns (bytes, status).

        Both the request and the answer are passed through unchanged.  vf's
        error statuses are meaningful to the viewer (503 retry, 415 permanent)
        and are returned rather than translated.
        """
        url = "%s/offer?channel=%d" % (self.vf_base, int(channel))
        request = urlrequest.Request(
            url, data=sdp_json_bytes,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=timeout,
                                    context=self._context) as resp:
                return resp.read(), resp.status
        except urlerror.HTTPError as exc:
            return exc.read(), exc.code
        except Exception as exc:                                # noqa: BLE001
            raise InsightError("POST %s -> %s" % (url, exc)) from exc

    # -- startup check ------------------------------------------------------

    def preflight(self):
        """(ok, list_of_lines).  Never raises; the launcher prints the lines."""
        lines = []
        ok = True
        try:
            health = self.health()
            lines.append("Insight API  %s  %s"
                         % (self.api_base, health.get("status", "?")))
        except InsightError as exc:
            ok = False
            lines.append("Insight API  %s  UNREACHABLE (%s)" % (self.api_base, exc))
        try:
            body = self.drawing_js()
            renderer_ok = b"window.drawStrategies" in body
            lines.append("vf renderer  %s/static/drawing.js  %d bytes  "
                         "drawStrategies=%s"
                         % (self.vf_base, len(body),
                            "yes" if renderer_ok else "NO"))
            if not renderer_ok:
                ok = False
        except InsightError as exc:
            ok = False
            lines.append("vf renderer  %s  UNREACHABLE (%s)" % (self.vf_base, exc))
        return ok, lines

    def identifies_as_insight(self):
        """True only if this endpoint really is a neat-insight.

        `/api/health` returns {"service": "neat-insight", ...}, so the check is
        on that field rather than on "something answered on 9900".
        """
        try:
            return (self.health() or {}).get("service") == "neat-insight"
        except InsightError:
            return False

    def channel_summary(self):
        """Compact per-channel ingest view for the UI's diagnostics strip."""
        try:
            stats = self.ingest_stats()
        except InsightError as exc:
            return {"error": str(exc)}
        out = {}
        for channel in stats.get("channels", []):
            index = channel.get("channel")
            media = channel.get("media") or {}
            rtp = channel.get("rtp") or {}
            metadata = channel.get("metadata") or {}
            forwarding = channel.get("forwarding") or {}
            webrtc = channel.get("webrtc") or {}
            out[str(index)] = {
                "active": bool(channel.get("active")),
                "codec": media.get("codec"),
                "packets_received": rtp.get("packets_received"),
                "bitrate_bps": rtp.get("bitrate_bps"),
                "idr_count": media.get("idr_count"),
                "track_attached": forwarding.get("webrtc_track_attached"),
                # How many WebRTC peers vf currently has on this channel.  This
                # is the ONLY evidence that distinguishes "our peer was
                # displaced by another viewer" from "the source paused": vf
                # gives a channel's media to one peer, so a second peer is what
                # displacement looks like.  With only this page open it is 1.
                "peers": webrtc.get("peer_count"),
                "packets_forwarded": forwarding.get("packets_forwarded"),
                "metadata_received": metadata.get("messages_received"),
                "metadata_forwarded": metadata.get("messages_forwarded"),
                "metadata_matched": (metadata.get("matched_video_first") or 0)
                                    + (metadata.get("matched_metadata_first") or 0),
                "metadata_invalid_json": metadata.get("invalid_json"),
            }
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--vf-base", default=DEFAULT_VF_BASE)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if not args.probe:
        parser.error("nothing to do; try --probe")

    client = InsightClient(args.api_base, args.vf_base)
    ok, lines = client.preflight()
    for line in lines:
        print("  " + line)
    print("\nchannels:")
    print(json.dumps(client.channel_summary(), indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
