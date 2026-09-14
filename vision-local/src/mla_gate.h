// Explicit MLA arbitration between the continuous YOLO vision workload and the
// on-demand Whisper/Qwen voice workload.
//
// WHY THIS EXISTS
// ---------------
// Both workloads execute on the same Machine Learning Accelerator, through the
// same board-wide mlashmcomplex shared-memory server (/tmp/mlactrl).  That server
// serialises individual MLA jobs, but it does not serialise a *sequence* of them:
// when the LLiMa runtime starts a Whisper/Qwen generation it reprograms the MLA
// (`mlashm_load_models bulk`) while a YOLO inference may still be in flight.  The
// measured result is the camera Graph going silent with no error and the model
// Runner subsequently returning `runtime.output_timeout` - a state neither a longer
// timeout nor a retry recovers from, because nothing is actually going to arrive.
//
// So the serialisation has to happen one level up, between the two applications,
// and it has to be based on a QUIESCENT POINT rather than on a mutex: it is not
// enough to stop starting new YOLO inferences, the one already submitted has to
// have been fully consumed before the voice runtime is allowed to touch the MLA.
//
// WHAT IT GUARANTEES
// ------------------
// When `request_pause()` returns true:
//
//   * every vision worker is parked at the top of its loop,
//   * no worker is inside `Model::Runner::run()`  (`inflight() == 0`),
//   * no worker has a camera `pull()` outstanding whose output it still needs,
//   * no worker will submit new model input until `release()` is called.
//
// The workers keep consuming the camera Graph while parked (see
// `PauseBehaviour`), so the camera queue cannot build up and the Graph stays
// healthy - it is never torn down for a normal voice command.
//
// The lease is owned by a TCP connection on the loopback interface, so a voice
// server that crashes mid-command cannot leave vision paused: the socket closing
// releases the lease.  A hard TTL is a second, independent backstop.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mla {

/// What a parked vision worker does with camera frames during the voice window.
enum class PauseBehaviour {
  /// Keep pulling frames and forward the pixels to the UDP video encoder with their
  /// own timestamps, but run NO inference and send NO metadata.  The camera Graph is
  /// consumed exactly as in normal operation and the video stream does not freeze.
  DrainToVideo,
  /// Keep pulling frames and discard them.  The camera Graph is still consumed (so it
  /// cannot stall), but the video stream freezes for the duration of the window.
  DrainAndDrop,
  /// Stop the camera Run before the voice runtime is allowed to touch the MLA, and
  /// start it again on resume.  THE DEFAULT, and not a choice this project made
  /// freely - see the measurement in README.md: a LLiMa GenAI inference makes the
  /// MIPI CSI-2 receiver overflow ("IPI overflow recovery", then
  /// "frame watchdog: stalled after 3 recovery attempts; giving up (restart the
  /// stream)" in dmesg) whatever the application does, and the kernel's own recovery
  /// cannot clear it.  Stopping the sensor stream for the window is intentional and
  /// pre-emptive; the alternative is an unintentional, permanent stall.
  StopCamera,
};

/// Log sink shared with the rest of the application so pause events interleave
/// correctly with the per-stream logs.
using LogFn = std::function<void(const std::string&)>;

/// The gate itself.  One instance, shared by every StreamRunner and by the arbiter
/// server thread.
class Gate {
public:
  Gate(int worker_count, PauseBehaviour behaviour, LogFn log);
  ~Gate();

  Gate(const Gate&) = delete;
  Gate& operator=(const Gate&) = delete;

  // ---- controller side (arbiter server thread) ----------------------------

  /// Ask every worker to reach its quiescent point and block until they all have.
  /// Returns false (and leaves vision running) if they do not get there in time.
  /// `quiesce_ms_out` receives how long reaching the quiescent point took.
  bool request_pause(const std::string& reason, int timeout_ms, std::int64_t& quiesce_ms_out);

  /// Let the workers run again.  Safe to call when not paused.
  void release(const std::string& reason);

  // ---- worker-initiated maintenance ---------------------------------------

  /// Take the pause on behalf of the CALLING worker, so it can do something to its
  /// own camera that must not happen while another camera is streaming.
  ///
  /// Rebuilding one camera Graph while the other is delivering requests aborts the
  /// process on this stack:
  ///     GstLibcameraSrcState::requestCompleted(): assertion
  ///         'wrap->request_.get() == request' failed
  ///     FATAL pipeline_handler.cpp:597 assertion "!req->hasPendingBuffers()" failed
  /// so the stall-recovery rebuild goes through the same quiescent point a voice
  /// command uses: every OTHER worker parks and stops its camera first.
  ///
  /// Waits for `workers - 1` parked workers (the caller is doing the work, not
  /// parking).  Returns false on timeout; the caller may then proceed anyway, because
  /// the alternative to a risky rebuild is a permanently dead stream.
  ///
  /// FIX 4 (concurrent camera restarts): pass `gate_busy` to NOT wait for the gate. If
  /// another voice window or another stream's camera maintenance already owns the pause,
  /// *gate_busy is set and false is returned at once; the caller then parks at its loop
  /// top (camera stopped during camera-fault maintenance) instead of rebuilding its camera
  /// concurrently. Measured 2026-09-13 11:19:35: two streams restarting in the same instant
  /// each waited for the other to park, then one rebuilt anyway and the process exited.
  bool begin_maintenance(const std::string& tag, const std::string& reason, int timeout_ms,
                         bool* gate_busy = nullptr);
  /// Finish worker-initiated maintenance and let everyone run again.
  void end_maintenance(const std::string& tag);

  /// True while a pause has been requested and not yet released.
  bool paused() const { return paused_.load(std::memory_order_acquire); }
  /// Workers currently parked at their quiescent point.
  int parked() const { return parked_.load(std::memory_order_acquire); }
  /// Model inferences currently submitted to the MLA and not yet consumed.
  int inflight() const { return inflight_.load(std::memory_order_acquire); }
  /// Total completed voice windows.
  std::uint64_t windows() const { return windows_.load(std::memory_order_relaxed); }

  PauseBehaviour behaviour() const { return behaviour_; }

  /// True while the current pause is a CAMERA-FAULT maintenance window
  /// (begin_maintenance with reason "camera-restart" or "graph-missing").
  ///
  /// drone seminar: voice windows use the configured behaviour (drop,
  /// cameras keep running), but a camera rebuild while another camera is streaming is
  /// exactly what aborts the process on this stack (see begin_maintenance). So during
  /// a camera-fault maintenance window every parked worker stops its camera Graph, as
  /// StopCamera does, whatever the configured voice behaviour is. Measured A/B on
  /// 2026-09-13: the fallback (stop-camera) survived a 40c3000 watchdog give-up; this
  /// project with drop exited in 4 of 4 identical faults.
  bool camera_maintenance() const { return camera_maintenance_.load(std::memory_order_acquire); }

  // ---- worker side --------------------------------------------------------

  /// True when the worker must stop submitting inference input and park.
  bool pause_requested() const { return paused_.load(std::memory_order_acquire); }

  /// Mark this worker as parked.  Call ONLY from the loop top, i.e. with no
  /// inference in flight and no camera output owed.
  void worker_parked(const std::string& tag);
  /// Leave the parked state, but ONLY if the pause really is over.
  ///
  /// Checked and applied under the gate's lock, which is what closes the window
  /// between "the pause was released" and "this worker has actually restarted its
  /// camera": without it a second ACQUIRE arriving in that window would find the
  /// worker still counted as parked and would be granted the MLA while the camera
  /// was coming back up.  Returns false when a new pause has already been requested
  /// - the caller then stays parked, keeps its input stopped and waits again.
  bool try_unpark(const std::string& tag, long dropped, long passthrough);

  /// Unconditionally leave the parked state.  Only for shutdown, where the worker is
  /// leaving its loop for good.
  void force_unpark(const std::string& tag);

  /// Block until the pause is released or `stop` becomes true.  `poll_ms` bounds how
  /// long the caller sleeps before it gets a chance to drain the camera again.
  void wait_while_paused(int poll_ms);

  /// RAII marker for "an inference is submitted to the MLA right now".
  class InFlight {
  public:
    explicit InFlight(Gate& gate) : gate_(gate) {
      gate_.inflight_.fetch_add(1, std::memory_order_acq_rel);
    }
    ~InFlight() { gate_.inflight_.fetch_sub(1, std::memory_order_acq_rel); }
    InFlight(const InFlight&) = delete;
    InFlight& operator=(const InFlight&) = delete;

  private:
    Gate& gate_;
  };

  /// Called when a worker thread exits, so a pause request cannot wait for a
  /// worker that will never park again.
  void worker_gone();

  void log(const std::string& line) const;

private:
  /// Shared by `request_pause()` and `begin_maintenance()`: exactly one pause owner
  /// at a time, so a voice command and a camera rebuild can never overlap.
  bool begin_pause(const std::string& reason, int timeout_ms, int required_parked,
                   std::int64_t& quiesce_ms_out, bool* owner_busy = nullptr);
  void end_pause(const std::string& reason);

  mutable std::mutex mu_;
  std::timed_mutex owner_mu_;
  /// True while `owner_mu_` is held by a pause.  Lets `end_pause()` be called
  /// unconditionally (Arbiter::stop() does) without unlocking a mutex nobody owns.
  bool pause_owned_ = false;
  std::condition_variable cv_;
  std::atomic<bool> paused_{false};
  std::atomic<bool> camera_maintenance_{false};
  std::atomic<int> parked_{0};
  std::atomic<int> inflight_{0};
  std::atomic<int> workers_;
  std::atomic<std::uint64_t> windows_{0};
  PauseBehaviour behaviour_;
  LogFn log_;
  std::chrono::steady_clock::time_point pause_started_{};
};

/// Loopback TCP server that hands out the vision pause as a connection-scoped lease.
///
/// Line protocol, one command per line, ASCII:
///
///     ACQUIRE <reason>   ->  PAUSED <quiesce_ms>       | ERROR <why>
///     RELEASE            ->  RESUMED <held_ms>         | ERROR <why>
///     STATUS             ->  STATUS <json>
///     PING               ->  PONG
///
/// The lease belongs to the connection that took it.  Closing the socket - including
/// by crashing - releases it.  `ttl_ms` is an independent backstop for a client that
/// stays connected but never releases.
class Arbiter {
public:
  Arbiter(Gate& gate, std::string bind_host, int port, int quiesce_timeout_ms, int ttl_ms);
  ~Arbiter();

  /// Bind, listen and start serving.  Throws on bind failure.
  void start();
  /// Stop serving and join the threads.  Idempotent.
  void stop();

  int port() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mla
