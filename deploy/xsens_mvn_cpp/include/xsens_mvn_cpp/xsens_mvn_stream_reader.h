#ifndef DEPLOY_XSENS_MVN_CPP_XSENS_MVN_STREAM_READER_H
#define DEPLOY_XSENS_MVN_CPP_XSENS_MVN_STREAM_READER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "xsens_mvn_cpp/latest_frame_store.h"
#include "xsens_mvn_cpp/udp_datagram_socket.h"
#include "xsens_mvn_cpp/xsens_datagram_parser.h"

namespace xsens_mvn_cpp
{

/**
 * @brief Configuration for Xsens MVN UDP stream ingestion.
 *
 * Preconditions:
 * - udp_port is the MVN UDP streaming port.
 *
 * Postconditions:
 * - Values are immutable once passed into XsensMvnStreamReader.
 */
struct XsensMvnStreamConfig
{
  std::uint16_t udp_port = 8001;
  int receive_timeout_ms = 2;
  std::size_t max_datagram_size = 5000;
};

/**
 * @brief High-frequency Xsens MVN stream reader that stores only the latest frame.
 *
 * Preconditions:
 * - Xsens MVN is configured to stream UDP datagrams to udp_port.
 *
 * Postconditions:
 * - No datagrams are forwarded.
 * - No multi-frame buffering is performed.
 * - latestFrame() returns only the latest assembled frame snapshot.
 */
class XsensMvnStreamReader
{
public:
  /**
   * @brief Construct a reader from configuration and default dependencies.
   *
   * Preconditions:
   * - config.udp_port is a valid UDP port.
   *
   * Postconditions:
   * - The reader is not running until start() is called.
   */
  explicit XsensMvnStreamReader(const XsensMvnStreamConfig& config);

  /**
   * @brief Stop the reader thread and release socket resources.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - isRunning() returns false.
   */
  ~XsensMvnStreamReader();

  XsensMvnStreamReader(const XsensMvnStreamReader&) = delete;
  XsensMvnStreamReader& operator=(const XsensMvnStreamReader&) = delete;

  /**
   * @brief Start the background UDP parsing loop.
   *
   * Preconditions:
   * - The reader is not already running.
   *
   * Postconditions:
   * - isRunning() returns true after successful start.
   */
  void start();

  /**
   * @brief Stop the background UDP parsing loop.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - The worker thread is joined if it was running.
   */
  void stop();

  /**
   * @brief Return whether the reader loop is active.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  bool isRunning() const;

  /**
   * @brief Return whether any datagram has produced a frame snapshot.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  bool hasFrame() const;

  /**
   * @brief Return the latest frame sequence number.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  std::uint64_t sequence() const;

  /**
   * @brief Return the latest raw frame snapshot.
   *
   * Preconditions:
   * - hasFrame() is true for meaningful data.
   *
   * Postconditions:
   * - No internal buffers are exposed to the caller.
   */
  XsensRawFrame latestFrame() const;

private:
  void runLoop();

  XsensMvnStreamConfig config_;
  std::atomic<bool> running_{false};
  LatestFrameStore latest_frame_store_;
  std::unique_ptr<UdpDatagramSocket> socket_;
  std::unique_ptr<XsensDatagramParser> parser_;
  std::unique_ptr<XsensRawFrameAssembler> assembler_;
  std::thread worker_thread_;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_XSENS_MVN_STREAM_READER_H
