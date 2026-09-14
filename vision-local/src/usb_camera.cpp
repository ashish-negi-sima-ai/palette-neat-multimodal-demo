#include "usb_camera.h"

#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>

namespace usbcam {
namespace {

int xioctl(int fd, unsigned long request, void* arg) {
  int rc = 0;
  do {
    rc = ::ioctl(fd, request, arg);
  } while (rc == -1 && errno == EINTR);
  return rc;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string read_line(const std::string& path) {
  std::ifstream in(path);
  if (!in)
    return {};
  std::string s;
  std::getline(in, s);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

std::string real_path(const std::string& p) {
  char buf[PATH_MAX];
  if (::realpath(p.c_str(), buf) == nullptr)
    return {};
  return buf;
}

std::string base_name(const std::string& p) {
  const std::size_t k = p.rfind('/');
  return k == std::string::npos ? p : p.substr(k + 1);
}

std::string dir_name(const std::string& p) {
  const std::size_t k = p.rfind('/');
  return (k == std::string::npos || k == 0) ? std::string("/") : p.substr(0, k);
}

std::string cstr(const unsigned char* raw, std::size_t max) {
  const auto* chars = reinterpret_cast<const char*>(raw);
  return std::string(chars, ::strnlen(chars, max));
}

std::string fourcc_str(std::uint32_t f) {
  std::string s(4, ' ');
  for (int i = 0; i < 4; ++i)
    s[static_cast<std::size_t>(i)] = static_cast<char>((f >> (8 * i)) & 0xff);
  return s;
}

std::uint32_t fourcc_code(const std::string& s) {
  if (s.size() != 4)
    return 0;
  return v4l2_fourcc(s[0], s[1], s[2], s[3]);
}

/// "1-3.3" -> {1, 3, 3}; used for a natural topology order.
std::vector<int> numbers_in(const std::string& s) {
  std::vector<int> out;
  int cur = -1;
  for (char ch : s) {
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      cur = (cur < 0 ? 0 : cur * 10) + (ch - '0');
    } else if (cur >= 0) {
      out.push_back(cur);
      cur = -1;
    }
  }
  if (cur >= 0)
    out.push_back(cur);
  return out;
}

int node_index(const std::string& node) {
  const auto v = numbers_in(base_name(node));
  return v.empty() ? -1 : v.front();
}

void enumerate_modes(int fd, std::vector<Mode>& modes) {
  for (std::uint32_t fi = 0;; ++fi) {
    v4l2_fmtdesc desc{};
    desc.index = fi;
    desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_ENUM_FMT, &desc) < 0)
      break;
    const std::string fcc = fourcc_str(desc.pixelformat);
    if (fcc != "MJPG" && fcc != "YUYV")
      continue; // the only two formats this pipeline decodes
    for (std::uint32_t si = 0;; ++si) {
      v4l2_frmsizeenum size{};
      size.index = si;
      size.pixel_format = desc.pixelformat;
      if (xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) < 0 || size.type != V4L2_FRMSIZE_TYPE_DISCRETE)
        break;
      for (std::uint32_t ii = 0;; ++ii) {
        v4l2_frmivalenum iv{};
        iv.index = ii;
        iv.pixel_format = desc.pixelformat;
        iv.width = size.discrete.width;
        iv.height = size.discrete.height;
        if (xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &iv) < 0 ||
            iv.type != V4L2_FRMIVAL_TYPE_DISCRETE || iv.discrete.numerator == 0)
          break;
        const double fps = static_cast<double>(iv.discrete.denominator) / iv.discrete.numerator;
        Mode m;
        m.fourcc = fcc;
        m.width = static_cast<int>(size.discrete.width);
        m.height = static_cast<int>(size.discrete.height);
        m.fps = static_cast<int>(std::lround(fps));
        // Fractional rates (7.5 fps) cannot be given to the H.264 encoder as an integer.
        if (std::fabs(fps - m.fps) > 0.05 || m.fps <= 0)
          continue;
        if (std::find(modes.begin(), modes.end(), m) == modes.end())
          modes.push_back(m);
      }
    }
  }
}

std::string errno_text(int err) {
  return std::string(std::strerror(err)) + " (errno " + std::to_string(err) + ")";
}

bool disable_dynamic_framerate_fd(int fd, std::string& note) {
  v4l2_control ctrl{};
  ctrl.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
  if (xioctl(fd, VIDIOC_G_CTRL, &ctrl) < 0) {
    note = "exposure_dynamic_framerate not offered by this camera";
    return false;
  }
  if (ctrl.value == 0) {
    note = "exposure_dynamic_framerate already 0";
    return true;
  }
  ctrl.value = 0;
  if (xioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
    note = "exposure_dynamic_framerate=0 refused: " + errno_text(errno);
    return false;
  }
  note = "exposure_dynamic_framerate set 1 -> 0 (constant frame rate)";
  return true;
}

} // namespace

std::string format_mode(const Mode& m) {
  return m.fourcc + ":" + std::to_string(m.width) + "x" + std::to_string(m.height) + "@" +
         std::to_string(m.fps);
}

bool parse_mode(const std::string& text, Mode& out, const std::string& default_fourcc) {
  static const std::regex re(R"(^\s*(?:([A-Za-z0-9]{4}):)?\s*([0-9]+)x([0-9]+)@([0-9]+)\s*$)");
  std::smatch m;
  if (!std::regex_match(text, m, re))
    return false;
  std::string fcc = m[1].matched ? m[1].str() : default_fourcc;
  std::transform(fcc.begin(), fcc.end(), fcc.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (fcc == "MJPE" || fcc == "JPEG")
    fcc = "MJPG";
  if (fcc == "YUY2")
    fcc = "YUYV";
  out.fourcc = fcc;
  out.width = std::stoi(m[2].str());
  out.height = std::stoi(m[3].str());
  out.fps = std::stoi(m[4].str());
  return out.width > 0 && out.height > 0 && out.fps > 0;
}

bool Camera::supports(const Mode& m) const {
  return std::find(modes.begin(), modes.end(), m) != modes.end();
}

std::vector<Camera> discover(std::string* diagnostics) {
  std::ostringstream diag;
  std::vector<std::string> names;
  if (DIR* d = ::opendir("/sys/class/video4linux")) {
    static const std::regex node_re("^video[0-9]+$");
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (std::regex_match(n, node_re))
        names.push_back(n);
    }
    ::closedir(d);
  }
  std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
    return node_index(a) < node_index(b);
  });

  static const std::regex iface_re(R"(^([0-9]+-[0-9.]+):([0-9]+)\.([0-9]+)$)");
  std::vector<Camera> out;
  for (const std::string& name : names) {
    const std::string node = "/dev/" + name;
    const std::string dev = real_path("/sys/class/video4linux/" + name + "/device");
    std::smatch m;
    const std::string iface = base_name(dev);
    if (dev.empty() || !std::regex_match(iface, m, iface_re))
      continue; // not a USB interface: ISP, CSI capture, codec ... (not worth a diagnostic line)

    Camera c;
    c.video_node = node;
    c.usb_path = m[1].str();
    const std::string usbdev = dir_name(dev);
    c.vid_pid = read_line(usbdev + "/idVendor") + ":" + read_line(usbdev + "/idProduct");
    c.serial = read_line(usbdev + "/serial");
    c.product = read_line(usbdev + "/product");
    c.speed_mbps = read_line(usbdev + "/speed");
    const std::string ifnum = read_line(dev + "/bInterfaceNumber");
    c.interface_number = ifnum.empty() ? -1 : static_cast<int>(std::strtol(ifnum.c_str(), nullptr, 16));

    const int fd = ::open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
      diag << node << " (usb " << c.usb_path << "): cannot open: " << errno_text(errno) << "\n";
      continue;
    }
    v4l2_capability cap{};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
      diag << node << " (usb " << c.usb_path << "): VIDIOC_QUERYCAP failed: " << errno_text(errno)
           << "\n";
      ::close(fd);
      continue;
    }
    c.card = cstr(cap.card, sizeof(cap.card));
    c.driver = cstr(cap.driver, sizeof(cap.driver));
    c.bus_info = cstr(cap.bus_info, sizeof(cap.bus_info));
    // device_caps describes THIS node; capabilities describes the whole device, so a
    // UVC metadata node still lists VIDEO_CAPTURE there.
    const std::uint32_t caps =
        (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING)) {
      diag << node << " (" << c.card << ", usb " << c.usb_path
           << "): not a video-capture node (UVC metadata node)\n";
      ::close(fd);
      continue;
    }
    enumerate_modes(fd, c.modes);
    ::close(fd);
    if (c.modes.empty()) {
      diag << node << " (" << c.card << ", usb " << c.usb_path
           << "): offers no MJPG or YUYV capture mode\n";
      continue;
    }
    out.push_back(std::move(c));
  }

  std::stable_sort(out.begin(), out.end(), [](const Camera& a, const Camera& b) {
    const auto pa = numbers_in(a.usb_path), pb = numbers_in(b.usb_path);
    if (pa != pb)
      return pa < pb;
    if (a.interface_number != b.interface_number)
      return a.interface_number < b.interface_number;
    return node_index(a.video_node) < node_index(b.video_node);
  });

  // Stable identity: the serial number if it is unique among the attached cameras,
  // otherwise the USB topology path. Both carry vendor:product, so a different camera
  // plugged into the same port is never mistaken for the old one.
  std::map<std::string, int> serial_count;
  for (const auto& c : out)
    if (!c.serial.empty())
      ++serial_count[c.vid_pid + ":" + c.serial];
  std::map<std::string, int> seen;
  for (auto& c : out) {
    const std::string key = c.vid_pid + ":" + c.serial;
    if (!c.serial.empty() && serial_count[key] == 1)
      c.identity = "usb-serial:" + key;
    else
      c.identity = "usb-path:" + c.usb_path + "@" + c.vid_pid;
    const int n = seen[c.identity]++;
    if (n > 0)
      c.identity += "#" + std::to_string(n); // a second capture node on one interface
  }

  if (diagnostics != nullptr)
    *diagnostics = diag.str();
  return out;
}

std::string describe(const Camera& c) {
  std::ostringstream os;
  os << c.video_node << "  \"" << c.card << "\"  usb " << c.usb_path << "  " << c.vid_pid
     << "  serial " << (c.serial.empty() ? "(none)" : c.serial) << "  " << c.speed_mbps
     << "M  identity " << c.identity;
  return os.str();
}

std::string describe_modes(const Camera& c) {
  // fourcc -> (w,h) in first-seen order -> fps list
  std::vector<std::string> parts;
  for (const char* fcc : {"MJPG", "YUYV"}) {
    std::vector<std::pair<std::pair<int, int>, std::vector<int>>> sizes;
    for (const auto& m : c.modes) {
      if (m.fourcc != fcc)
        continue;
      auto it = std::find_if(sizes.begin(), sizes.end(), [&](const auto& s) {
        return s.first.first == m.width && s.first.second == m.height;
      });
      if (it == sizes.end()) {
        sizes.push_back({{m.width, m.height}, {}});
        it = sizes.end() - 1;
      }
      it->second.push_back(m.fps);
    }
    if (sizes.empty())
      continue;
    std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
      return static_cast<long>(a.first.first) * a.first.second >
             static_cast<long>(b.first.first) * b.first.second;
    });
    std::ostringstream os;
    os << fcc;
    for (const auto& s : sizes) {
      os << " " << s.first.first << "x" << s.first.second << "@";
      for (std::size_t i = 0; i < s.second.size(); ++i)
        os << (i ? "," : "") << s.second[i];
    }
    parts.push_back(os.str());
  }
  std::string out;
  for (std::size_t i = 0; i < parts.size(); ++i)
    out += (i ? " | " : "") + parts[i];
  return out;
}

bool matches(const Camera& c, const std::string& selector) {
  if (selector.empty())
    return true;
  auto after = [&](const char* prefix, std::string& rest) {
    const std::size_t n = std::strlen(prefix);
    if (selector.compare(0, n, prefix) != 0)
      return false;
    rest = selector.substr(n);
    return true;
  };
  std::string v;
  if (selector.rfind("/dev/", 0) == 0) {
    const std::string want = real_path(selector);
    return !want.empty() && want == real_path(c.video_node);
  }
  if (after("identity:", v))
    return c.identity == v;
  if (after("serial:", v))
    return !c.serial.empty() && c.serial == v;
  if (after("path:", v))
    return c.usb_path == v;
  if (after("id:", v))
    return lower(c.vid_pid) == lower(v);
  if (after("name:", v))
    return lower(c.card).find(lower(v)) != std::string::npos ||
           lower(c.product).find(lower(v)) != std::string::npos;
  const std::string needle = lower(selector);
  for (const std::string& hay : {c.card, c.product, c.usb_path, c.serial, c.vid_pid, c.identity})
    if (!hay.empty() && lower(hay).find(needle) != std::string::npos)
      return true;
  return false;
}

std::optional<Camera> resolve(const std::string& identity) {
  for (auto& c : discover())
    if (c.identity == identity)
      return c;
  return std::nullopt;
}

bool probe_streaming(const std::vector<Camera>& cams, const Mode& mode, int duration_ms,
                     std::vector<ProbeStats>& stats) {
  struct Open {
    int fd = -1;
    std::vector<std::pair<void*, std::size_t>> maps;
    bool streaming = false;
  };
  stats.assign(cams.size(), ProbeStats{});
  std::vector<Open> open(cams.size());
  auto cleanup = [&]() {
    for (auto& o : open) {
      if (o.fd < 0)
        continue;
      if (o.streaming) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(o.fd, VIDIOC_STREAMOFF, &type);
      }
      for (auto& mp : o.maps)
        ::munmap(mp.first, mp.second);
      v4l2_requestbuffers rb{};
      rb.count = 0;
      rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      rb.memory = V4L2_MEMORY_MMAP;
      xioctl(o.fd, VIDIOC_REQBUFS, &rb);
      ::close(o.fd);
      o.fd = -1;
    }
  };

  const std::uint32_t pix = fourcc_code(mode.fourcc);
  bool ok = true;
  for (std::size_t i = 0; i < cams.size() && ok; ++i) {
    auto& st = stats[i];
    auto& o = open[i];
    st.node = cams[i].video_node;
    o.fd = ::open(st.node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (o.fd < 0) {
      st.error = "open: " + errno_text(errno);
      ok = false;
      break;
    }
    std::string note;
    disable_dynamic_framerate_fd(o.fd, note);

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<std::uint32_t>(mode.width);
    fmt.fmt.pix.height = static_cast<std::uint32_t>(mode.height);
    fmt.fmt.pix.pixelformat = pix;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(o.fd, VIDIOC_S_FMT, &fmt) < 0) {
      st.error = "VIDIOC_S_FMT: " + errno_text(errno);
      ok = false;
      break;
    }
    if (fmt.fmt.pix.width != static_cast<std::uint32_t>(mode.width) ||
        fmt.fmt.pix.height != static_cast<std::uint32_t>(mode.height) ||
        fmt.fmt.pix.pixelformat != pix) {
      st.error = "driver adjusted the format to " + fourcc_str(fmt.fmt.pix.pixelformat) + " " +
                 std::to_string(fmt.fmt.pix.width) + "x" + std::to_string(fmt.fmt.pix.height);
      ok = false;
      break;
    }
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(o.fd, VIDIOC_G_PARM, &parm) == 0 &&
        (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) {
      parm.parm.capture.timeperframe.numerator = 1;
      parm.parm.capture.timeperframe.denominator = static_cast<std::uint32_t>(mode.fps);
      xioctl(o.fd, VIDIOC_S_PARM, &parm);
    }
    v4l2_requestbuffers rb{};
    rb.count = 4;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    if (xioctl(o.fd, VIDIOC_REQBUFS, &rb) < 0 || rb.count < 2) {
      st.error = "VIDIOC_REQBUFS: " + errno_text(errno);
      ok = false;
      break;
    }
    for (std::uint32_t b = 0; b < rb.count && ok; ++b) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = b;
      if (xioctl(o.fd, VIDIOC_QUERYBUF, &buf) < 0) {
        st.error = "VIDIOC_QUERYBUF: " + errno_text(errno);
        ok = false;
        break;
      }
      void* p = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, o.fd,
                       static_cast<off_t>(buf.m.offset));
      if (p == MAP_FAILED) {
        st.error = "mmap: " + errno_text(errno);
        ok = false;
        break;
      }
      o.maps.emplace_back(p, buf.length);
      if (xioctl(o.fd, VIDIOC_QBUF, &buf) < 0) {
        st.error = "VIDIOC_QBUF: " + errno_text(errno);
        ok = false;
        break;
      }
    }
  }

  // Start every stream: this is where the USB bus refuses a mode it cannot carry.
  for (std::size_t i = 0; i < cams.size() && ok; ++i) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(open[i].fd, VIDIOC_STREAMON, &type) < 0) {
      const int err = errno;
      stats[i].error = "VIDIOC_STREAMON: " + errno_text(err);
      if (err == ENOSPC)
        stats[i].error += " - USB bus bandwidth exhausted for this mode while the other "
                          "camera(s) stream";
      ok = false;
      break;
    }
    open[i].streaming = true;
  }

  if (ok) {
    using clock = std::chrono::steady_clock;
    std::vector<clock::time_point> first(cams.size()), last(cams.size());
    const auto t0 = clock::now();
    while (clock::now() - t0 < std::chrono::milliseconds(duration_ms)) {
      std::vector<pollfd> pfds;
      for (auto& o : open)
        pfds.push_back({o.fd, POLLIN, 0});
      if (::poll(pfds.data(), pfds.size(), 200) <= 0)
        continue;
      for (std::size_t i = 0; i < cams.size(); ++i) {
        if (!(pfds[i].revents & (POLLIN | POLLERR)))
          continue;
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(open[i].fd, VIDIOC_DQBUF, &buf) < 0) {
          if (errno != EAGAIN && stats[i].error.empty())
            stats[i].error = "VIDIOC_DQBUF: " + errno_text(errno);
          continue;
        }
        if ((buf.flags & V4L2_BUF_FLAG_ERROR) || buf.bytesused == 0) {
          ++stats[i].error_frames;
        } else {
          const auto now = clock::now();
          if (stats[i].frames == 0)
            first[i] = now;
          last[i] = now;
          ++stats[i].frames;
        }
        xioctl(open[i].fd, VIDIOC_QBUF, &buf);
      }
    }
    for (std::size_t i = 0; i < cams.size(); ++i) {
      auto& st = stats[i];
      const double span = std::chrono::duration<double>(last[i] - first[i]).count();
      st.fps = (st.frames > 1 && span > 0.0) ? (st.frames - 1) / span : 0.0;
      if (st.error.empty() && (st.frames < 3 || st.fps < 0.75 * mode.fps))
        st.error = "delivered only " + std::to_string(st.frames) + " frames (" +
                   std::to_string(static_cast<int>(std::lround(st.fps))) + " fps) in " +
                   std::to_string(duration_ms) + " ms";
      if (!st.error.empty())
        ok = false;
    }
  }
  cleanup();
  return ok;
}

bool disable_dynamic_framerate(const std::string& node, std::string& note) {
  const int fd = ::open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    note = "cannot open " + node + ": " + errno_text(errno);
    return false;
  }
  const bool ok = disable_dynamic_framerate_fd(fd, note);
  ::close(fd);
  return ok;
}

std::vector<int> node_holders(const std::string& node) {
  std::vector<int> pids;
  const std::string want = real_path(node);
  if (want.empty())
    return pids;
  const int self = static_cast<int>(::getpid());
  DIR* proc = ::opendir("/proc");
  if (proc == nullptr)
    return pids;
  while (dirent* e = ::readdir(proc)) {
    const std::string pid_s = e->d_name;
    if (pid_s.empty() || !std::all_of(pid_s.begin(), pid_s.end(),
                                      [](unsigned char ch) { return std::isdigit(ch); }))
      continue;
    const int pid = std::stoi(pid_s);
    if (pid == self)
      continue;
    const std::string fd_dir = "/proc/" + pid_s + "/fd";
    DIR* fds = ::opendir(fd_dir.c_str());
    if (fds == nullptr)
      continue;
    while (dirent* f = ::readdir(fds)) {
      if (f->d_name[0] == '.')
        continue;
      char buf[PATH_MAX];
      const std::string link = fd_dir + "/" + f->d_name;
      const ssize_t n = ::readlink(link.c_str(), buf, sizeof(buf) - 1);
      if (n > 0 && std::string(buf, static_cast<std::size_t>(n)) == want) {
        pids.push_back(pid);
        break;
      }
    }
    ::closedir(fds);
  }
  ::closedir(proc);
  return pids;
}

} // namespace usbcam
