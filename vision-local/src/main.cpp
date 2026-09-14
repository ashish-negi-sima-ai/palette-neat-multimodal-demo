// YOLO26m detection + YOLO26m-seg segmentation on two USB (UVC) cameras,
// streamed to Neat Insight as two CLEAN RTP/H.264 UDP streams plus two UDP metadata
// streams on adjacent ports. Insight draws the overlays from the metadata.
//
//   USB camera 0 (LEFT)  -> MJPEG -> neatdecoder NV12 -> letterbox 640 -> YOLO26m     (MLA)
//        -> on-device YoloV26 decode + NMS -> boxes
//             -> clean H.264 -> UDP video port    (camera pixels only)
//             -> object-detection JSON -> UDP metadata port
//   USB camera 1 (RIGHT) -> MJPEG -> neatdecoder NV12 -> letterbox 640 -> YOLO26m-seg (MLA)
//        -> on-device YoloV26Seg decode + NMS -> boxes+masks
//             -> clean H.264 -> UDP video port    (camera pixels only)
//             -> segmentation JSON (polygon masks) -> UDP metadata port
//
// Inference, video, metadata and MLA arbitration are the ones of the MIPI edition;
// only the camera source changed. Camera discovery, stable identity and capture-mode
// selection live in usb_camera.cpp; no /dev/videoN is ever hard-coded.

#include "mla_gate.h"
#include "stream_worker.h"
#include "usb_camera.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) {
  g_stop.store(true);
}

// ---------------------------------------------------------------- tiny config
// A flat "section.key = value" view over the small YAML subset config.yaml uses
// (two-space nesting, scalar leaves, `#` comments). Avoids a YAML dependency in
// the cross-build for what is a dozen settings.
class Config {
public:
  bool load(const std::string& path) {
    std::ifstream in(path);
    if (!in)
      return false;
    std::vector<std::string> stack;
    std::string line;
    while (std::getline(in, line)) {
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos)
        line = line.substr(0, hash);
      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
        continue;

      const std::size_t indent = line.find_first_not_of(' ');
      const std::size_t depth = indent / 2;
      std::string body = line.substr(indent);
      while (!body.empty() && (body.back() == ' ' || body.back() == '\r'))
        body.pop_back();

      const std::size_t colon = body.find(':');
      if (colon == std::string::npos)
        continue;
      std::string key = body.substr(0, colon);
      std::string value = body.substr(colon + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      while (!value.empty() && (value.back() == ' ' || value.back() == '\r'))
        value.pop_back();
      if (value.size() >= 2 && (value.front() == '"' || value.front() == '\''))
        value = value.substr(1, value.size() - 2);

      stack.resize(depth);
      if (value.empty()) {
        stack.push_back(key);
        continue;
      }
      std::string full;
      for (const auto& s : stack)
        full += s + ".";
      values_[full + key] = value;
    }
    return true;
  }

  std::string str(const std::string& k, const std::string& d) const {
    auto it = values_.find(k);
    return it == values_.end() ? d : it->second;
  }
  int integer(const std::string& k, int d) const {
    auto it = values_.find(k);
    if (it == values_.end())
      return d;
    try {
      return std::stoi(it->second);
    } catch (...) {
      return d;
    }
  }
  float real(const std::string& k, float d) const {
    auto it = values_.find(k);
    if (it == values_.end())
      return d;
    try {
      return std::stof(it->second);
    } catch (...) {
      return d;
    }
  }
  bool boolean(const std::string& k, bool d) const {
    auto it = values_.find(k);
    if (it == values_.end())
      return d;
    return it->second == "true" || it->second == "1" || it->second == "yes";
  }

private:
  std::map<std::string, std::string> values_;
};

bool arg_value(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (key == argv[i])
      return true;
  return false;
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [--config config.yaml] [options]\n"
      << "  --list-cameras          list every USB capture camera and its modes, then exit\n"
      << "  --probe-cameras         select the two cameras and a capture mode that both can\n"
      << "                          stream at the same time, print it, then exit\n"
      << "  --selection-out <file>  also write that selection as shell variables\n"
      << "  --camera0 <selector>    camera for ch0 / LEFT / detection   (USB_CAMERA0_DEVICE)\n"
      << "  --camera1 <selector>    camera for ch1 / RIGHT / segmentation (USB_CAMERA1_DEVICE)\n"
      << "                          selector: /dev/videoN, /dev/v4l/by-id/..., serial:<s>,\n"
      << "                          path:<1-3.3>, id:<vid:pid>, name:<text>, identity:<id>\n"
      << "  --camera-filter <sel>   only consider cameras matching this selector\n"
      << "  --usb-format <f>        auto (MJPG, else YUYV) | mjpeg | yuyv\n"
      << "  --usb-modes <list>      preference list, e.g. 1920x1080@30,1280x720@30\n"
      << "  --usb-mode <mode>       use exactly this mode, e.g. MJPG:1280x720@30\n"
      << "  --skip-mode-probe       trust --usb-mode without a streaming probe (run.sh has\n"
      << "                          just probed it)\n"
      << "  --probe-ms <n>          how long each mode is test-streamed (default 2500)\n"
      << "  --swap-cameras          exchange camera 0 and camera 1\n"
      << "  --host <ip>             Insight/UDP destination host\n"
      << "  --port-base <n>         first UDP port (second stream uses n+1)\n"
      << "  --frames <n>            stop after n frames per stream (0 = run forever)\n"
      << "  --det-model <path>      YOLO26m detection archive (.tar.gz)\n"
      << "  --seg-model <path>      YOLO26m-seg segmentation archive (.tar.gz)\n"
      << "  --single <det|seg>      run only one of the two streams\n"
      << "  --no-video              skip the UDP video output (debugging)\n"
      << "  --no-metadata           skip the UDP metadata output\n"
      << "  --no-source-pts         let appsrc stamp the PTS (reproduces the pairing failure)\n"
      << "  --metadata-clock <c>    frame (default) | pipeline | source\n"
      << "  --pull-timeout-ms <n>   how long one camera pull waits (poll interval)\n"
      << "  --stall-grace-ms <n>    how long the camera may deliver nothing before the\n"
      << "                          stream is declared dead (0 = retry forever)\n"
      << "  --camera-restart-ms <n> how long before a silent camera graph is rebuilt\n"
      << "  --metadata-port-base <n>  first metadata UDP port (default 9100)\n"
      << "  --video-memory <m>      ev74 (default) | system | dms0 | auto\n"
      << "  --test-image <path>     run one image through the model instead of the\n"
      << "                          cameras, print detections, write --test-out\n"
      << "  --test-out <path>       annotated output for --test-image\n"
      << "  --test-loops <n>        DIAGNOSTIC: run the still image n times back to back\n"
      << "  --arbiter-port <n>      loopback port the MLA arbiter listens on\n"
      << "  --no-arbiter            run with no MLA arbitration (standalone behaviour)\n"
      << "  --no-inference          DIAGNOSTIC: camera + video only, no Model/Runner\n"
      << "  --pause-behaviour <b>   stop-camera | video | drop (default from config) --\n"
      << "                          what a parked stream does during a voice window\n"
      << "  --pause-poll-ms <n>     camera poll interval while parked\n"
      << "  --quiesce-timeout-ms <n>  how long ACQUIRE waits for both streams to park\n"
      << "  --lease-ttl-ms <n>      backstop after which an unreleased lease resumes vision\n";
}

const char* kPanel[] = {"LEFT, ch0 detection", "RIGHT, ch1 segmentation"};

/// Pick `needed` cameras: explicit selectors first, then the remaining cameras in USB
/// topology order. More candidates than free slots is ambiguous and is refused rather
/// than guessed.
bool select_cameras(const std::vector<usbcam::Camera>& all, const std::string& filter,
                    const std::vector<std::string>& selectors, std::size_t needed,
                    std::vector<usbcam::Camera>& out, std::string& err) {
  std::vector<usbcam::Camera> pool;
  for (const auto& c : all)
    if (usbcam::matches(c, filter))
      pool.push_back(c);
  out.assign(needed, usbcam::Camera{});
  std::vector<bool> filled(needed, false), used(pool.size(), false);

  for (std::size_t i = 0; i < needed && i < selectors.size(); ++i) {
    const std::string& sel = selectors[i];
    if (sel.empty())
      continue;
    std::vector<std::size_t> hits;
    for (std::size_t j = 0; j < pool.size(); ++j)
      if (!used[j] && usbcam::matches(pool[j], sel))
        hits.push_back(j);
    if (hits.empty()) {
      err = "USB_CAMERA" + std::to_string(i) + "_DEVICE=\"" + sel +
            "\" matches no (unassigned) USB capture camera";
      return false;
    }
    if (hits.size() > 1) {
      err = "USB_CAMERA" + std::to_string(i) + "_DEVICE=\"" + sel + "\" matches " +
            std::to_string(hits.size()) +
            " cameras; use a more specific selector (serial:, path:, identity: or /dev/v4l/by-path/...)";
      return false;
    }
    out[i] = pool[hits[0]];
    used[hits[0]] = true;
    filled[i] = true;
  }

  std::vector<std::size_t> rest;
  for (std::size_t j = 0; j < pool.size(); ++j)
    if (!used[j])
      rest.push_back(j);
  std::size_t empty_slots = 0;
  for (bool f : filled)
    empty_slots += f ? 0 : 1;

  if (rest.size() < empty_slots) {
    err = "need " + std::to_string(needed) + " USB camera(s), found " + std::to_string(pool.size()) +
          (filter.empty() ? std::string() : " matching USB_CAMERA_FILTER=\"" + filter + "\"") +
          ". Connect two USB UVC cameras.";
    return false;
  }
  if (empty_slots > 0 && rest.size() > empty_slots) {
    err = std::to_string(rest.size()) + " USB cameras are available for " +
          std::to_string(empty_slots) +
          " unassigned slot(s); refusing to guess. Set USB_CAMERA0_DEVICE / USB_CAMERA1_DEVICE "
          "or USB_CAMERA_FILTER in config/local.env.";
    return false;
  }
  std::size_t k = 0;
  for (std::size_t i = 0; i < needed; ++i)
    if (!filled[i])
      out[i] = pool[rest[k++]];
  return true;
}

std::vector<usbcam::Mode> candidate_modes(const std::string& ladder, const std::string& format) {
  std::vector<std::string> fourccs;
  if (format == "mjpeg" || format == "mjpg")
    fourccs = {"MJPG"};
  else if (format == "yuyv")
    fourccs = {"YUYV"};
  else
    fourccs = {"MJPG", "YUYV"};
  std::vector<usbcam::Mode> out;
  std::stringstream ss(ladder);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.find_first_not_of(" \t") == std::string::npos)
      continue;
    usbcam::Mode m;
    if (item.find(':') != std::string::npos) {
      if (usbcam::parse_mode(item, m))
        out.push_back(m);
      continue;
    }
    for (const auto& f : fourccs)
      if (usbcam::parse_mode(item, m, f))
        out.push_back(m);
  }
  return out;
}

std::string fps_text(double v) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(1) << v;
  return os.str();
}

/// Try the modes in preference order with every selected camera streaming at once.
bool choose_mode(const std::vector<usbcam::Camera>& cams, const std::vector<usbcam::Mode>& candidates,
                 int probe_ms, usbcam::Mode& chosen, std::vector<usbcam::ProbeStats>& chosen_stats) {
  for (const auto& m : candidates) {
    std::string missing;
    for (const auto& c : cams)
      if (!c.supports(m))
        missing += (missing.empty() ? "" : ", ") + c.card;
    if (!missing.empty()) {
      if (m.fourcc == "MJPG")
        std::cout << "  mode " << usbcam::format_mode(m) << ": not offered by " << missing << "\n";
      continue;
    }
    std::vector<usbcam::ProbeStats> st;
    const bool ok = usbcam::probe_streaming(cams, m, probe_ms, st);
    std::cout << "  mode " << usbcam::format_mode(m) << ": " << (ok ? "OK     " : "REFUSED");
    for (std::size_t i = 0; i < st.size(); ++i) {
      std::cout << "  camera" << i << " " << st[i].node << " ";
      if (!st[i].error.empty() && !(ok))
        std::cout << "[" << st[i].error << "]";
      else if (!ok && st[i].frames == 0)
        std::cout << "[configured; not measured because another camera was refused]";
      else
        std::cout << fps_text(st[i].fps) << " fps (" << st[i].frames << " frames"
                  << (st[i].error_frames ? ", " + std::to_string(st[i].error_frames) + " bad" : "")
                  << ")";
    }
    std::cout << std::endl;
    if (ok) {
      chosen = m;
      chosen_stats = st;
      return true;
    }
  }
  return false;
}

void print_listing(const std::vector<usbcam::Camera>& cams, const std::string& diag) {
  std::cout << "USB capture cameras (" << cams.size() << "), in USB topology order:\n";
  for (std::size_t i = 0; i < cams.size(); ++i) {
    std::cout << "  [" << i << "] " << usbcam::describe(cams[i]) << "\n"
              << "      modes: " << usbcam::describe_modes(cams[i]) << "\n";
  }
  if (!diag.empty()) {
    std::cout << "video nodes on USB that were NOT selected:\n";
    std::stringstream ss(diag);
    std::string line;
    while (std::getline(ss, line))
      std::cout << "  " << line << "\n";
  }
  std::cout << std::flush;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

} // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGHUP, on_signal);

  if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
    usage(argv[0]);
    return 0;
  }

  std::string config_path = "config.yaml";
  arg_value(argc, argv, "--config", config_path);
  Config cfg;
  cfg.load(config_path); // absent config is fine: defaults + CLI still work

  // ---- offline verification path -------------------------------------------
  // Runs the identical Model + on-device decode + overlay code the live streams
  // use, so YOLO26 accuracy can be checked against a known scene without needing
  // the cameras to be pointed at anything in particular.
  std::string test_image;
  if (arg_value(argc, argv, "--test-image", test_image)) {
    std::string test_out;
    arg_value(argc, argv, "--test-out", test_out);
    std::string which = "det";
    arg_value(argc, argv, "--single", which);

    pipeline::StreamConfig t;
    t.task = which == "seg" ? pipeline::Task::Segment : pipeline::Task::Detect;
    t.labels_path = cfg.str("model.labels", "assets/coco_labels.txt");
    t.model_size = cfg.integer("inference.model_size", 640);
    t.min_score = cfg.real("inference.min_score", 0.35f);
    t.nms_iou = cfg.real("inference.nms_iou", 0.50f);
    t.max_detections = cfg.integer("inference.max_detections", 50);
    t.mask_alpha = cfg.real("output.mask_alpha", 0.50f);
    t.mask_threshold = cfg.real("output.mask_threshold", 0.50f);
    t.infer_timeout_ms = cfg.integer("runtime.infer_timeout_ms", 20000);
    t.model_path = t.task == pipeline::Task::Segment
                       ? cfg.str("model.segmentation", "assets/models/yolo26m-seg.tar.gz")
                       : cfg.str("model.detection", "assets/models/yolo26m.tar.gz");
    std::string mv;
    if (arg_value(argc, argv, "--det-model", mv) && t.task == pipeline::Task::Detect)
      t.model_path = mv;
    if (arg_value(argc, argv, "--seg-model", mv) && t.task == pipeline::Task::Segment)
      t.model_path = mv;
    if (arg_value(argc, argv, "--labels", mv))
      t.labels_path = mv;
    std::string loops_s = "1";
    arg_value(argc, argv, "--test-loops", loops_s);
    return pipeline::run_test_image(t, test_image, test_out, std::stoi(loops_s));
  }

  // ---- discover the USB cameras ---------------------------------------------
  std::string diag;
  const auto cameras = usbcam::discover(&diag);
  if (has_flag(argc, argv, "--list-cameras")) {
    print_listing(cameras, diag);
    return cameras.size() >= 2 ? 0 : 1;
  }

  std::string single;
  arg_value(argc, argv, "--single", single);
  const bool want_det = single.empty() || single == "det";
  const bool want_seg = single.empty() || single == "seg";
  const std::size_t needed = (want_det ? 1 : 0) + (want_seg ? 1 : 0);

  std::string sel0 = cfg.str("camera.camera0", "");
  std::string sel1 = cfg.str("camera.camera1", "");
  std::string filter = cfg.str("camera.filter", "");
  std::string format = cfg.str("camera.format", "auto");
  std::string ladder = cfg.str("camera.modes", "1920x1080@30,1280x720@30");
  std::string forced_mode;
  int probe_ms = cfg.integer("camera.probe_ms", 2500);
  arg_value(argc, argv, "--camera0", sel0);
  arg_value(argc, argv, "--camera1", sel1);
  arg_value(argc, argv, "--camera-filter", filter);
  arg_value(argc, argv, "--usb-format", format);
  arg_value(argc, argv, "--usb-modes", ladder);
  arg_value(argc, argv, "--usb-mode", forced_mode);
  std::string probe_ms_s;
  if (arg_value(argc, argv, "--probe-ms", probe_ms_s))
    probe_ms = std::stoi(probe_ms_s);
  std::transform(format.begin(), format.end(), format.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  // With --single seg the one camera is still "camera 1" for the user, so its selector applies.
  std::vector<std::string> selectors;
  if (want_det)
    selectors.push_back(sel0);
  if (want_seg)
    selectors.push_back(want_det ? sel1 : (sel1.empty() ? sel0 : sel1));

  std::vector<usbcam::Camera> chosen;
  std::string err;
  if (!select_cameras(cameras, filter, selectors, needed, chosen, err)) {
    std::cerr << "[ERR] USB camera selection: " << err << "\n";
    print_listing(cameras, diag);
    return 1;
  }
  if (has_flag(argc, argv, "--swap-cameras") && chosen.size() >= 2) {
    std::swap(chosen[0], chosen[1]);
    std::cout << "--swap-cameras: camera 0 <-> camera 1 exchanged\n";
  }

  std::cout << "discovered " << cameras.size() << " USB capture camera(s); selected:\n";
  for (std::size_t i = 0; i < chosen.size(); ++i)
    std::cout << "  camera" << i << " (" << kPanel[want_det ? i : 1] << "): "
              << usbcam::describe(chosen[i]) << "\n      modes: " << usbcam::describe_modes(chosen[i])
              << "\n";
  std::cout << std::flush;

  // ---- choose the capture mode -----------------------------------------------
  usbcam::Mode mode;
  std::vector<usbcam::ProbeStats> mode_stats;
  if (!forced_mode.empty()) {
    if (!usbcam::parse_mode(forced_mode, mode, format == "yuyv" ? "YUYV" : "MJPG")) {
      std::cerr << "[ERR] --usb-mode \"" << forced_mode << "\" is not FOURCC:WIDTHxHEIGHT@FPS\n";
      return 1;
    }
    for (const auto& c : chosen) {
      if (!c.supports(mode)) {
        std::cerr << "[ERR] " << c.card << " (" << c.video_node << ") does not offer "
                  << usbcam::format_mode(mode) << "; it offers: " << usbcam::describe_modes(c) << "\n";
        return 1;
      }
    }
    if (!has_flag(argc, argv, "--skip-mode-probe")) {
      std::cout << "capture mode probe (all selected cameras streaming at once):\n";
      if (!choose_mode(chosen, {mode}, probe_ms, mode, mode_stats)) {
        std::cerr << "[ERR] the forced mode " << usbcam::format_mode(mode)
                  << " cannot be streamed by all selected cameras at once\n";
        return 1;
      }
    } else {
      std::cout << "capture mode " << usbcam::format_mode(mode)
                << " (selected and probed by run.sh preflight)\n";
    }
  } else {
    const auto candidates = candidate_modes(ladder, format);
    std::cout << "capture mode probe, preference order " << ladder << " (format " << format
              << "), all selected cameras streaming at once:\n";
    if (!choose_mode(chosen, candidates, probe_ms, mode, mode_stats)) {
      std::cerr << "[ERR] none of the preferred capture modes (" << ladder
                << ") can be streamed by all selected cameras at the same time.\n"
                << "      Add a lower mode to USB_CAMERA_MODES, or use a separate USB bus per camera.\n";
      return 1;
    }
  }
  const usbcam::Mode preferred = [&] {
    const auto c = candidate_modes(ladder, format);
    return c.empty() ? mode : c.front();
  }();

  std::cout << "\n";
  for (std::size_t i = 0; i < chosen.size(); ++i) {
    const std::string p = "USB Camera " + std::to_string(want_det ? i : 1) + " ";
    std::cout << p << "device     : " << chosen[i].video_node << "\n"
              << p << "identity   : " << chosen[i].identity << "  (\"" << chosen[i].card
              << "\", usb " << chosen[i].usb_path << ", " << chosen[i].vid_pid << ")\n"
              << p << "format     : " << mode.fourcc
              << (mode.fourcc == "MJPG" ? " (hardware JPEG decode to NV12)" : " (CPU convert to NV12)")
              << "\n"
              << p << "resolution : " << mode.width << "x" << mode.height << "\n"
              << p << "fps        : " << mode.fps
              << (i < mode_stats.size() ? " (probe measured " + fps_text(mode_stats[i].fps) + ")" : "")
              << "\n";
  }
  if (!(mode == preferred))
    std::cout << "NOTE: preferred mode " << usbcam::format_mode(preferred)
              << " was not usable with both cameras streaming (see the probe above); using "
              << usbcam::format_mode(mode) << "\n";
  std::cout << std::endl;

  std::string selection_out;
  if (arg_value(argc, argv, "--selection-out", selection_out)) {
    std::ofstream f(selection_out);
    if (!f) {
      std::cerr << "[ERR] cannot write " << selection_out << "\n";
      return 1;
    }
    f << "# written by drone-seminar-usb-vision --probe-cameras\n";
    f << "USB_SELECTED_MODE=" << shell_quote(usbcam::format_mode(mode)) << "\n";
    f << "USB_SELECTED_PREFERRED_MODE=" << shell_quote(usbcam::format_mode(preferred)) << "\n";
    for (std::size_t i = 0; i < chosen.size(); ++i) {
      const std::string p = "USB_SELECTED_CAMERA" + std::to_string(want_det ? i : 1) + "_";
      f << p << "IDENTITY=" << shell_quote(chosen[i].identity) << "\n"
        << p << "DEVICE=" << shell_quote(chosen[i].video_node) << "\n"
        << p << "CARD=" << shell_quote(chosen[i].card) << "\n"
        << p << "USB_PATH=" << shell_quote(chosen[i].usb_path) << "\n"
        << p << "VID_PID=" << shell_quote(chosen[i].vid_pid) << "\n"
        << p << "SERIAL=" << shell_quote(chosen[i].serial) << "\n"
        << p << "PROBE_FPS=" << shell_quote(i < mode_stats.size() ? fps_text(mode_stats[i].fps) : "")
        << "\n";
    }
  }
  if (has_flag(argc, argv, "--probe-cameras"))
    return 0;

  // ---- shared settings -----------------------------------------------------
  pipeline::StreamConfig base;
  base.pixel_format = mode.fourcc;
  base.cam_width = static_cast<std::uint32_t>(mode.width);
  base.cam_height = static_cast<std::uint32_t>(mode.height);
  base.cam_fps = static_cast<std::uint32_t>(mode.fps);
  base.camera_retry_ms = cfg.integer("camera.retry_ms", 2000);
  base.camera_settle_ms = cfg.integer("camera.settle_ms", 2000);
  base.camera_build_timeout_ms = cfg.integer("camera.build_timeout_ms", 20000);
  base.model_size = cfg.integer("inference.model_size", 640);
  base.min_score = cfg.real("inference.min_score", 0.35f);
  base.nms_iou = cfg.real("inference.nms_iou", 0.50f);
  base.max_detections = cfg.integer("inference.max_detections", 50);
  base.mask_alpha = cfg.real("output.mask_alpha", 0.50f);
  base.mask_threshold = cfg.real("output.mask_threshold", 0.50f);
  base.insight_host = cfg.str("output.host", "10.42.0.1");
  base.video_port_base = cfg.integer("output.video_port_base", 9000);
  base.metadata_port_base = cfg.integer("output.metadata_port_base", 9100);
  base.metadata_enabled = cfg.boolean("output.metadata_enabled", true) &&
                          !has_flag(argc, argv, "--no-metadata");
  base.video_pts_from_source = cfg.boolean("output.video_pts_from_source", true) &&
                               !has_flag(argc, argv, "--no-source-pts");
  base.metadata_budget_bytes = cfg.integer("output.metadata_budget_bytes", 32768);
  base.metadata_clock = cfg.str("output.metadata_clock", "frame");
  base.metadata_offset_ms = cfg.integer("output.metadata_offset_ms", 0);
  base.bitrate_kbps = cfg.integer("output.bitrate_kbps", 6000);
  base.video_memory = cfg.str("output.video_memory", "ev74");
  base.video_enabled = !has_flag(argc, argv, "--no-video");
  base.frames = cfg.integer("runtime.frames", 0);
  base.pull_timeout_ms = cfg.integer("runtime.pull_timeout_ms", 15000);
  base.first_frame_timeout_ms = cfg.integer("runtime.first_frame_timeout_ms", 15000);
  base.max_runner_restarts = cfg.integer("runtime.max_runner_restarts", 5);
  base.stall_grace_ms = cfg.integer("runtime.stall_grace_ms", 0);
  base.camera_restart_ms = cfg.integer("runtime.camera_restart_ms", 5000);
  base.infer_timeout_ms = cfg.integer("runtime.infer_timeout_ms", 20000);
  base.pause_poll_ms = cfg.integer("mla.pause_poll_ms", 200);
  base.labels_path = cfg.str("model.labels", "assets/coco_labels.txt");
  base.inference_enabled = !has_flag(argc, argv, "--no-inference");

  // ---- MLA arbitration settings -------------------------------------------
  const std::string arbiter_host = cfg.str("mla.arbiter_host", "127.0.0.1");
  int arbiter_port = cfg.integer("mla.arbiter_port", 8974);
  int quiesce_timeout_ms = cfg.integer("mla.quiesce_timeout_ms", 10000);
  int lease_ttl_ms = cfg.integer("mla.lease_ttl_ms", 180000);
  std::string pause_behaviour = cfg.str("mla.pause_behaviour", "drop");
  bool use_arbiter = cfg.boolean("mla.enabled", true) && !has_flag(argc, argv, "--no-arbiter");

  std::string v;
  if (arg_value(argc, argv, "--host", v))
    base.insight_host = v;
  if (arg_value(argc, argv, "--port-base", v))
    base.video_port_base = std::stoi(v);
  if (arg_value(argc, argv, "--metadata-port-base", v))
    base.metadata_port_base = std::stoi(v);
  if (arg_value(argc, argv, "--metadata-clock", v))
    base.metadata_clock = v;
  if (arg_value(argc, argv, "--metadata-offset-ms", v))
    base.metadata_offset_ms = std::stoi(v);
  if (arg_value(argc, argv, "--pull-timeout-ms", v))
    base.pull_timeout_ms = std::stoi(v);
  if (arg_value(argc, argv, "--stall-grace-ms", v))
    base.stall_grace_ms = std::stoi(v);
  if (arg_value(argc, argv, "--camera-restart-ms", v))
    base.camera_restart_ms = std::stoi(v);
  if (arg_value(argc, argv, "--frames", v))
    base.frames = std::stoi(v);
  if (arg_value(argc, argv, "--labels", v))
    base.labels_path = v;
  if (arg_value(argc, argv, "--video-memory", v))
    base.video_memory = v;
  if (arg_value(argc, argv, "--arbiter-port", v))
    arbiter_port = std::stoi(v);
  if (arg_value(argc, argv, "--pause-behaviour", v))
    pause_behaviour = v;
  if (arg_value(argc, argv, "--pause-poll-ms", v))
    base.pause_poll_ms = std::stoi(v);
  if (arg_value(argc, argv, "--quiesce-timeout-ms", v))
    quiesce_timeout_ms = std::stoi(v);
  // Workers use the same bound when they request a maintenance pause of their own.
  base.quiesce_timeout_ms = quiesce_timeout_ms;
  if (arg_value(argc, argv, "--lease-ttl-ms", v))
    lease_ttl_ms = std::stoi(v);

  std::string det_model = cfg.str("model.detection", "assets/models/yolo26m.tar.gz");
  std::string seg_model = cfg.str("model.segmentation", "assets/models/yolo26m-seg.tar.gz");
  arg_value(argc, argv, "--det-model", det_model);
  arg_value(argc, argv, "--seg-model", seg_model);

  // ---- one stream per camera ----------------------------------------------
  // Channel index drives the UDP port (base + channel), so the two streams land
  // on adjacent ports and Insight can show them in two windows.
  std::vector<pipeline::StreamConfig> streams;
  int channel = 0;
  std::size_t cam_index = 0;
  auto fill_camera = [&](pipeline::StreamConfig& s) {
    const auto& c = chosen[cam_index++];
    s.usb_identity = c.identity;
    s.video_node = c.video_node;
    s.camera_card = c.card;
  };
  if (want_det) {
    auto s = base;
    s.task = pipeline::Task::Detect;
    s.model_path = det_model;
    fill_camera(s);
    s.channel = channel++;
    streams.push_back(std::move(s));
  }
  if (want_seg) {
    auto s = base;
    s.task = pipeline::Task::Segment;
    s.model_path = seg_model;
    fill_camera(s);
    s.channel = channel++;
    streams.push_back(std::move(s));
  }

  std::cout << "\nstreams:\n";
  for (const auto& s : streams)
    std::cout << "  ch" << s.channel << "  " << pipeline::task_name(s.task) << "  cam=\""
              << s.camera_card << "\" (" << s.video_node << ", " << s.usb_identity << ")  "
              << s.pixel_format << " " << s.cam_width << "x" << s.cam_height << "@" << s.cam_fps
              << "  model=" << s.model_path << "\n         video    udp://" << s.insight_host << ":"
              << pipeline::video_port_of(s)
              << (s.metadata_enabled
                      ? "\n         metadata udp://" + s.insight_host + ":" +
                            std::to_string(pipeline::metadata_port_of(s))
                      : std::string())
              << "\n";
  std::cout << std::endl;

  // Setup is SERIAL and the loops are concurrent. Building two Models/Runners at
  // the same time makes the MLA refuse a stage with
  // "infra.accelerator_execution_failed" while the other stream fails to map its
  // output. open() every stream first, then start the threads.
  std::vector<std::unique_ptr<pipeline::StreamRunner>> runners;
  runners.reserve(streams.size());
  for (const auto& s : streams)
    runners.push_back(std::make_unique<pipeline::StreamRunner>(s));

  // ---- MLA arbitration -----------------------------------------------------
  // One gate shared by every stream, plus a loopback server that hands the vision
  // pause to the voice server as a connection-scoped lease. Created before the
  // Runners are opened but STARTED only once both streams are streaming, so a voice
  // command can never arrive while a Runner is still being built.
  mla::PauseBehaviour behaviour = mla::PauseBehaviour::StopCamera;
  if (pause_behaviour == "drop")
    behaviour = mla::PauseBehaviour::DrainAndDrop;
  else if (pause_behaviour == "video")
    behaviour = mla::PauseBehaviour::DrainToVideo;
  mla::Gate gate(static_cast<int>(runners.size()), behaviour,
                 [](const std::string& line) { std::cout << line << std::endl; });
  std::unique_ptr<mla::Arbiter> arbiter;
  if (use_arbiter) {
    for (auto& r : runners)
      r->set_mla_gate(&gate);
  }

  for (auto& r : runners) {
    try {
      r->open();
    } catch (const std::exception& e) {
      std::cerr << "[ERR] ch" << r->config().channel << " " << pipeline::task_name(r->config().task)
                << " setup failed: " << e.what() << "\n";
      for (auto& other : runners)
        other->close();
      return 1;
    }
  }
  std::cout << std::endl;

  if (use_arbiter) {
    arbiter = std::make_unique<mla::Arbiter>(gate, arbiter_host, arbiter_port,
                                             quiesce_timeout_ms, lease_ttl_ms);
    try {
      arbiter->start();
    } catch (const std::exception& e) {
      std::cerr << "[ERR] MLA arbiter failed to start: " << e.what() << "\n"
                << "      Voice commands could not be serialised against YOLO inference,\n"
                << "      which is exactly the contention this pipeline exists to avoid.\n";
      for (auto& r : runners)
        r->close();
      return 1;
    }
    const char* behaviour_text =
        behaviour == mla::PauseBehaviour::StopCamera
            ? "stop the camera stream for the window, then restart it"
            : (behaviour == mla::PauseBehaviour::DrainToVideo
                   ? "drain camera -> video, no inference, no metadata"
                   : "drain camera and drop (cameras and video keep running)");
    std::cout << "MLA arbitration: ON  (pause behaviour: " << behaviour_text << ")"
              << std::endl;
  } else {
    std::cout << "MLA arbitration: OFF (--no-arbiter): this is the standalone behaviour and a "
                 "voice command WILL contend with YOLO inference."
              << std::endl;
  }

  std::vector<std::thread> threads;
  threads.reserve(runners.size());
  for (auto& r : runners) {
    threads.emplace_back([&r]() {
      r->run_loop(g_stop);
      if (r->result() != 0)
        g_stop.store(true); // one stream dying takes the app down, not just itself
    });
  }
  for (auto& t : threads)
    t.join();
  if (arbiter)
    arbiter->stop();
  for (auto& r : runners)
    r->close();

  int total_restarts = 0;
  for (const auto& r : runners) {
    total_restarts += r->camera_restarts();
    std::cout << "ch" << r->config().channel << " " << pipeline::task_name(r->config().task)
              << ": frames=" << r->frames_done() << " camera_restarts=" << r->camera_restarts()
              << " runner_restarts=" << r->runner_restarts() << std::endl;
  }
  std::cout << "voice windows served: " << gate.windows()
            << ", camera restarts total: " << total_restarts << std::endl;

  const bool ok = std::all_of(runners.begin(), runners.end(),
                              [](const auto& r) { return r->result() == 0; });
  std::cout << (ok ? "[OK] all streams stopped cleanly" : "[FAIL] a stream reported an error")
            << std::endl;
  if (pipeline::abandoned_builds() > 0) {
    // A graph build that hung on a vanished USB camera may still be running on a detached
    // thread; skip static destruction, which that thread could still be using.
    std::cout << "NOTE: " << pipeline::abandoned_builds()
              << " hung camera graph build(s) were abandoned during this run; exiting now"
              << std::endl;
    std::fflush(nullptr);
    std::_Exit(ok ? 0 : 1);
  }
  return ok ? 0 : 1;
}
