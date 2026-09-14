// USB (UVC) camera discovery, stable identity and capture-mode selection for the
// dual USB camera edition of the drone demo.
//
// NOTHING here is hard-coded to a device number. A UVC camera exposes more than one
// /dev/videoN (the capture node and a metadata node), the Modalix ISP publishes ~96
// more, and Linux may hand a reconnected camera a different number. So:
//
//   * candidates are the /sys/class/video4linux/videoN entries whose device is a USB
//     INTERFACE (…/usbX/…/1-3.3/1-3.3:1.0), which rejects every ISP/CSI/codec node;
//   * each candidate must report V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING in its
//     own device_caps (rejects the UVC metadata node) and offer MJPG or YUYV modes;
//   * each camera gets a STABLE identity: the USB serial number when the camera has a
//     unique one, otherwise its USB topology path (bus-port.port, e.g. "1-3.1") plus
//     vendor:product. The identity - not the node number - is what the pipeline keeps,
//     and every camera (re)start resolves it to whatever /dev/videoN it has now;
//   * cameras are ordered by USB topology, so camera 0 / camera 1 do not change
//     between starts while the cameras stay on the same ports.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace usbcam {

/// One capture mode: pixel format, frame size and integral frame rate.
struct Mode {
  std::string fourcc = "MJPG"; ///< "MJPG" or "YUYV"
  int width = 0;
  int height = 0;
  int fps = 0;

  bool operator==(const Mode& o) const {
    return fourcc == o.fourcc && width == o.width && height == o.height && fps == o.fps;
  }
};

/// "MJPG:1280x720@30"
std::string format_mode(const Mode& m);
/// Parse "MJPG:1280x720@30" (or "1280x720@30", taking `default_fourcc`).
bool parse_mode(const std::string& text, Mode& out, const std::string& default_fourcc = "MJPG");

/// One discovered USB video-capture node.
struct Camera {
  std::string video_node;   ///< current capture node, e.g. "/dev/video4"
  std::string card;         ///< V4L2 card name, e.g. "C922 Pro Stream Webcam"
  std::string driver;       ///< V4L2 driver, normally "uvcvideo"
  std::string bus_info;     ///< V4L2 bus info, e.g. "usb-0003:01:00.0-3.3"
  std::string usb_path;     ///< USB topology path from sysfs, e.g. "1-3.3"
  std::string vid_pid;      ///< "046d:085c"
  std::string serial;       ///< USB serial number (may be empty)
  std::string product;      ///< USB product string
  std::string speed_mbps;   ///< negotiated USB speed, e.g. "480"
  int interface_number = -1;
  std::string identity;     ///< stable identity, see the file comment
  std::vector<Mode> modes;  ///< every discrete MJPG/YUYV capture mode

  bool supports(const Mode& m) const;
};

/// Every USB video-capture node, ordered by USB topology. `diagnostics` receives one
/// line per /dev/videoN that was looked at and rejected, with the reason.
std::vector<Camera> discover(std::string* diagnostics = nullptr);

/// One-line description for logs.
std::string describe(const Camera& c);
/// Short summary of the modes, e.g. "MJPG 1920x1080@30,24,20,15,10,5 | 1280x720@60,30,...".
std::string describe_modes(const Camera& c);

/// Does `c` match a user selector?
///   /dev/videoN or /dev/v4l/by-*/…   the node (symlinks resolved)
///   identity:<identity>              exact stable identity
///   serial:<serial>  path:<1-3.3>  id:<vid:pid>  name:<substring>
///   anything else                    case-insensitive substring of card, product,
///                                    USB path, serial, vid:pid or identity
bool matches(const Camera& c, const std::string& selector);

/// Find the camera that currently carries `identity` (re-enumerated node included).
std::optional<Camera> resolve(const std::string& identity);

/// Per-camera outcome of `probe_streaming`.
struct ProbeStats {
  std::string node;
  int frames = 0;
  int error_frames = 0;
  double fps = 0.0;
  std::string error; ///< empty on success
};

/// Start ALL `cams` streaming `mode` at the same time for `duration_ms`, count frames,
/// stop and release them. This is the only reliable test of a mode: USB bandwidth is
/// reserved per bus when a stream starts, so a mode every camera advertises can still
/// be refused (ENOSPC) once the other camera is streaming. Returns true when every
/// camera started and delivered at least 75 % of the mode's frame rate.
bool probe_streaming(const std::vector<Camera>& cams, const Mode& mode, int duration_ms,
                     std::vector<ProbeStats>& stats);

/// Turn off the UVC "exposure dynamic framerate" (auto-exposure priority) control, so
/// the camera keeps the negotiated frame rate in low light instead of stretching the
/// exposure and silently dropping to ~20 fps. `note` says what happened.
bool disable_dynamic_framerate(const std::string& node, std::string& note);

/// PIDs (other than this process) that hold `node` open, from /proc/*/fd.
std::vector<int> node_holders(const std::string& node);

} // namespace usbcam
