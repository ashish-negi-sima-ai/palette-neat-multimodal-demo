#include "mla_gate.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace mla {
namespace {

std::int64_t ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                               t0)
      .count();
}

} // namespace

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

Gate::Gate(int worker_count, PauseBehaviour behaviour, LogFn log)
    : workers_(worker_count), behaviour_(behaviour), log_(std::move(log)) {}

Gate::~Gate() = default;

void Gate::log(const std::string& line) const {
  if (log_)
    log_(line);
}

bool Gate::begin_pause(const std::string& reason, int timeout_ms, int required_parked,
                       std::int64_t& quiesce_ms_out, bool* owner_busy) {
  const auto t0 = std::chrono::steady_clock::now();
  // One pause owner at a time.  A voice command and a worker-initiated camera rebuild
  // must never overlap: both stop cameras, and only one of them may put them back.
  if (owner_busy) {
    *owner_busy = false;
    if (!owner_mu_.try_lock()) {
      *owner_busy = true;
      quiesce_ms_out = ms_since(t0);
      return false;
    }
  } else if (!owner_mu_.try_lock_for(std::chrono::milliseconds(timeout_ms))) {
    quiesce_ms_out = ms_since(t0);
    log("[mla] pause request (" + reason + ") gave up after " + std::to_string(timeout_ms) +
        " ms: another pause is still in progress");
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    log("[mla] controlled vision pause REQUESTED (" + reason + ")");
    pause_started_ = t0;
    pause_owned_ = true;
    // Camera-fault maintenance only: parked workers stop their camera Graph too.
    camera_maintenance_.store(reason == "maintenance:camera-restart" ||
                                  reason == "maintenance:graph-missing",
                              std::memory_order_release);
    paused_.store(true, std::memory_order_release);
  }
  cv_.notify_all();

  std::unique_lock<std::mutex> lock(mu_);
  const bool ok = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
    return parked_.load(std::memory_order_acquire) >= required_parked;
  });
  quiesce_ms_out = ms_since(t0);

  if (!ok) {
    log("[mla] vision quiescent point NOT reached in " + std::to_string(timeout_ms) + " ms (" +
        std::to_string(parked_.load()) + "/" + std::to_string(required_parked) +
        " workers parked, inflight=" + std::to_string(inflight_.load()) + ")");
    paused_.store(false, std::memory_order_release);
    camera_maintenance_.store(false, std::memory_order_release);
    pause_owned_ = false;
    lock.unlock();
    cv_.notify_all();
    owner_mu_.unlock();
    return false;
  }

  log("[mla] YOLO quiescent point reached in " + std::to_string(quiesce_ms_out) + " ms (" +
      std::to_string(parked_.load()) + "/" + std::to_string(required_parked) +
      " workers parked, in-flight YOLO inferences=" + std::to_string(inflight_.load()) + ")");
  return true;
}

void Gate::end_pause(const std::string& reason) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!pause_owned_)
      return; // nothing to release; callers may invoke this unconditionally
    log("[mla] YOLO MLA resume REQUESTED (" + reason + ") after " +
        std::to_string(ms_since(pause_started_)) + " ms");
    paused_.store(false, std::memory_order_release);
    camera_maintenance_.store(false, std::memory_order_release);
    pause_owned_ = false;
    windows_.fetch_add(1, std::memory_order_relaxed);
  }
  cv_.notify_all();
  owner_mu_.unlock();
}

bool Gate::request_pause(const std::string& reason, int timeout_ms, std::int64_t& quiesce_ms_out) {
  const bool ok = begin_pause(reason, timeout_ms, workers_.load(std::memory_order_acquire),
                              quiesce_ms_out);
  if (ok)
    log("[mla] YOLO MLA paused - voice runtime may now use the MLA");
  return ok;
}

void Gate::release(const std::string& reason) {
  end_pause(reason);
}

bool Gate::begin_maintenance(const std::string& tag, const std::string& reason, int timeout_ms,
                             bool* gate_busy) {
  std::int64_t quiesce_ms = 0;
  // workers - 1: the caller is doing the work, not parking.
  const int required = std::max(0, workers_.load(std::memory_order_acquire) - 1);
  log(tag + " [recovery] requesting a vision pause before touching the camera (" + reason + ")");
  const bool ok = begin_pause("maintenance:" + reason, timeout_ms, required, quiesce_ms, gate_busy);
  if (gate_busy && *gate_busy) {
    log(tag + " [recovery] another voice window or camera maintenance owns the vision pause; "
               "parking with it instead of rebuilding concurrently (" + reason + ")");
    return false;
  }
  if (ok)
    log(tag + (camera_maintenance()
                   ? " [recovery] every other stream is parked with its camera stopped; "
                     "rebuilding is safe"
                   : " [recovery] every other stream is parked (cameras not stopped for " +
                         reason + ")"));
  else
    log(tag + " [recovery] WARNING: could not park the other stream(s); rebuilding anyway, "
               "because the alternative is a permanently dead stream");
  return ok;
}

void Gate::end_maintenance(const std::string& tag) {
  log(tag + " [recovery] maintenance finished; releasing the vision pause");
  end_pause("maintenance-complete");
}

void Gate::worker_parked(const std::string& tag) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    parked_.fetch_add(1, std::memory_order_acq_rel);
  }
  log(tag + " [mla] quiescent: no inference in flight, no model input being submitted");
  cv_.notify_all();
}

bool Gate::try_unpark(const std::string& tag, long dropped, long passthrough) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (paused_.load(std::memory_order_acquire))
      return false; // a new voice window started; stay parked, keep the input stopped
    parked_.fetch_sub(1, std::memory_order_acq_rel);
  }
  log(tag + " [mla] resumed: " + std::to_string(passthrough) +
      " frame(s) passed through to video, " + std::to_string(dropped) +
      " frame(s) dropped during the voice window");
  cv_.notify_all();
  return true;
}

void Gate::force_unpark(const std::string& tag) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (parked_.load(std::memory_order_acquire) > 0)
      parked_.fetch_sub(1, std::memory_order_acq_rel);
  }
  log(tag + " [mla] leaving the parked state (shutting down)");
  cv_.notify_all();
}

void Gate::worker_gone() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    workers_.fetch_sub(1, std::memory_order_acq_rel);
  }
  cv_.notify_all();
}

void Gate::wait_while_paused(int poll_ms) {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait_for(lock, std::chrono::milliseconds(poll_ms),
               [this] { return !paused_.load(std::memory_order_acquire); });
}

// ---------------------------------------------------------------------------
// Arbiter
// ---------------------------------------------------------------------------

struct Arbiter::Impl {
  Gate& gate;
  std::string host;
  int port;
  int quiesce_timeout_ms;
  int ttl_ms;

  int listen_fd = -1;
  int wake_fd[2] = {-1, -1}; ///< self-pipe so stop() interrupts poll()
  std::thread accept_thread;
  std::atomic<bool> running{false};

  /// Only one lease at a time.  Guarded so two clients cannot both believe they hold it.
  std::mutex lease_mu;
  bool lease_held = false;

  Impl(Gate& g, std::string h, int p, int q, int t)
      : gate(g), host(std::move(h)), port(p), quiesce_timeout_ms(q), ttl_ms(t) {}

  void serve_client(int fd);
  void accept_loop();
};

namespace {

bool write_all(int fd, const std::string& s) {
  std::size_t off = 0;
  while (off < s.size()) {
    const ssize_t n = ::write(fd, s.data() + off, s.size() - off);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    off += static_cast<std::size_t>(n);
  }
  return true;
}

/// Read one '\n'-terminated line.  Returns false on EOF/error/timeout-with-stop.
/// `timeout_ms` < 0 waits forever.  `*timed_out` distinguishes a TTL expiry from EOF.
bool read_line(int fd, int wake, std::string& out, int timeout_ms, bool* timed_out) {
  out.clear();
  *timed_out = false;
  char ch = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    struct pollfd pfd[2];
    pfd[0] = {fd, POLLIN, 0};
    pfd[1] = {wake, POLLIN, 0};
    int remaining = -1;
    if (timeout_ms >= 0) {
      const auto spent = ms_since(t0);
      remaining = static_cast<int>(timeout_ms - spent);
      if (remaining <= 0) {
        *timed_out = true;
        return false;
      }
    }
    const int pr = ::poll(pfd, 2, remaining);
    if (pr < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (pr == 0) {
      *timed_out = true;
      return false;
    }
    if (pfd[1].revents & POLLIN)
      return false; // server stopping
    if (!(pfd[0].revents & POLLIN))
      return false;
    const ssize_t n = ::read(fd, &ch, 1);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false; // peer closed
    }
    if (ch == '\n')
      return true;
    if (ch != '\r' && out.size() < 4096)
      out.push_back(ch);
  }
}

std::string trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t");
  if (b == std::string::npos)
    return {};
  const auto e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

} // namespace

void Arbiter::Impl::serve_client(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  bool this_conn_holds = false;
  std::chrono::steady_clock::time_point acquired_at{};

  for (;;) {
    std::string line;
    bool timed_out = false;
    // While this connection holds the lease the read is bounded by the TTL, so a
    // client that stops talking cannot keep vision paused for ever.
    const int timeout = this_conn_holds ? ttl_ms : -1;
    if (!read_line(fd, wake_fd[0], line, timeout, &timed_out)) {
      if (this_conn_holds) {
        if (timed_out)
          gate.log("[mla] lease TTL of " + std::to_string(ttl_ms) +
                   " ms expired without RELEASE - resuming vision (backstop)");
        else
          gate.log("[mla] voice client disconnected while holding the lease - resuming vision");
        gate.release(timed_out ? "lease-ttl-expired" : "client-disconnected");
        std::lock_guard<std::mutex> lk(lease_mu);
        lease_held = false;
      }
      break;
    }

    line = trim(line);
    std::string verb = line;
    std::string rest;
    const auto sp = line.find(' ');
    if (sp != std::string::npos) {
      verb = line.substr(0, sp);
      rest = trim(line.substr(sp + 1));
    }

    if (verb == "PING") {
      if (!write_all(fd, "PONG\n"))
        break;
      continue;
    }

    if (verb == "STATUS") {
      std::ostringstream os;
      os << "STATUS {\"paused\":" << (gate.paused() ? "true" : "false")
         << ",\"parked\":" << gate.parked() << ",\"inflight\":" << gate.inflight()
         << ",\"lease_held\":" << (lease_held ? "true" : "false")
         << ",\"windows\":" << gate.windows() << "}\n";
      if (!write_all(fd, os.str()))
        break;
      continue;
    }

    if (verb == "ACQUIRE") {
      if (this_conn_holds) {
        if (!write_all(fd, "ERROR this connection already holds the lease\n"))
          break;
        continue;
      }
      {
        std::lock_guard<std::mutex> lk(lease_mu);
        if (lease_held) {
          if (!write_all(fd, "ERROR lease already held by another client\n"))
            break;
          continue;
        }
        lease_held = true;
      }
      std::int64_t quiesce_ms = 0;
      const bool ok = gate.request_pause(rest.empty() ? std::string("voice") : rest,
                                         quiesce_timeout_ms, quiesce_ms);
      if (!ok) {
        {
          std::lock_guard<std::mutex> lk(lease_mu);
          lease_held = false;
        }
        if (!write_all(fd, "ERROR vision did not reach a quiescent point in " +
                               std::to_string(quiesce_timeout_ms) + " ms\n"))
          break;
        continue;
      }
      this_conn_holds = true;
      acquired_at = std::chrono::steady_clock::now();
      if (!write_all(fd, "PAUSED " + std::to_string(quiesce_ms) + "\n"))
        break;
      continue;
    }

    if (verb == "RELEASE") {
      if (!this_conn_holds) {
        if (!write_all(fd, "ERROR this connection does not hold the lease\n"))
          break;
        continue;
      }
      const std::int64_t held = ms_since(acquired_at);
      gate.release(rest.empty() ? std::string("voice-complete") : rest);
      this_conn_holds = false;
      {
        std::lock_guard<std::mutex> lk(lease_mu);
        lease_held = false;
      }
      if (!write_all(fd, "RESUMED " + std::to_string(held) + "\n"))
        break;
      continue;
    }

    if (!write_all(fd, "ERROR unknown command " + verb + "\n"))
      break;
  }

  ::close(fd);
}

void Arbiter::Impl::accept_loop() {
  // One thread per connection; finished ones are joined at every poll. Joining only at
  // shutdown kept a finished thread (and its stack mapping) alive per voice command for
  // the life of the process (measured: 86 threads after 52 lease windows).
  struct Client {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<Client> clients;
  auto reap = [&clients]() {
    for (auto it = clients.begin(); it != clients.end();) {
      if (it->done->load()) {
        it->thread.join();
        it = clients.erase(it);
      } else {
        ++it;
      }
    }
  };
  while (running.load()) {
    reap();
    struct pollfd pfd[2];
    pfd[0] = {listen_fd, POLLIN, 0};
    pfd[1] = {wake_fd[0], POLLIN, 0};
    const int pr = ::poll(pfd, 2, 500);
    if (pr < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    if (!running.load() || (pfd[1].revents & POLLIN))
      break;
    if (pr == 0 || !(pfd[0].revents & POLLIN))
      continue;
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0)
      continue;
    auto done = std::make_shared<std::atomic<bool>>(false);
    clients.push_back({std::thread([this, fd, done]() {
                         serve_client(fd);
                         done->store(true);
                       }),
                       done});
  }
  for (auto& c : clients)
    if (c.thread.joinable())
      c.thread.join();
}

Arbiter::Arbiter(Gate& gate, std::string bind_host, int port, int quiesce_timeout_ms, int ttl_ms)
    : impl_(std::make_unique<Impl>(gate, std::move(bind_host), port, quiesce_timeout_ms, ttl_ms)) {}

Arbiter::~Arbiter() {
  stop();
}

int Arbiter::port() const {
  return impl_->port;
}

void Arbiter::start() {
  Impl& s = *impl_;
  if (::pipe(s.wake_fd) != 0)
    throw std::runtime_error(std::string("arbiter pipe(): ") + std::strerror(errno));

  s.listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s.listen_fd < 0)
    throw std::runtime_error(std::string("arbiter socket(): ") + std::strerror(errno));
  int one = 1;
  ::setsockopt(s.listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(s.port));
  if (::inet_pton(AF_INET, s.host.c_str(), &addr.sin_addr) != 1)
    throw std::runtime_error("arbiter: bad bind address " + s.host);
  if (::bind(s.listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    throw std::runtime_error("arbiter: cannot bind " + s.host + ":" + std::to_string(s.port) +
                             ": " + std::strerror(errno));
  if (::listen(s.listen_fd, 8) != 0)
    throw std::runtime_error(std::string("arbiter listen(): ") + std::strerror(errno));

  s.running.store(true);
  s.accept_thread = std::thread([&s]() { s.accept_loop(); });
  s.gate.log("[mla] arbiter listening on " + s.host + ":" + std::to_string(s.port) +
             " (ACQUIRE/RELEASE, quiesce timeout " + std::to_string(s.quiesce_timeout_ms) +
             " ms, lease TTL " + std::to_string(s.ttl_ms) + " ms)");
}

void Arbiter::stop() {
  Impl& s = *impl_;
  if (!s.running.exchange(false))
    return;
  if (s.wake_fd[1] >= 0) {
    const char b = 'x';
    [[maybe_unused]] const ssize_t ignored = ::write(s.wake_fd[1], &b, 1);
  }
  if (s.accept_thread.joinable())
    s.accept_thread.join();
  if (s.listen_fd >= 0)
    ::close(s.listen_fd);
  if (s.wake_fd[0] >= 0)
    ::close(s.wake_fd[0]);
  if (s.wake_fd[1] >= 0)
    ::close(s.wake_fd[1]);
  s.listen_fd = s.wake_fd[0] = s.wake_fd[1] = -1;
  s.gate.release("arbiter-stopped");
}

} // namespace mla
