#include "xsens_mvn_cpp/xsens_mvn_stream_reader.h"

#include <stdexcept>

namespace xsens_mvn_cpp
{

XsensMvnStreamReader::XsensMvnStreamReader(const XsensMvnStreamConfig& config)
  : config_(config),
    socket_(new UdpDatagramSocket()),
    parser_(new XsensDatagramParser()),
    assembler_(new XsensRawFrameAssembler())
{
  if (config_.udp_port == 0)
  {
    throw std::invalid_argument("udp_port must be greater than zero");
  }
  if (config_.receive_timeout_ms < 0)
  {
    throw std::invalid_argument("receive_timeout_ms must be non-negative");
  }
  if (config_.max_datagram_size == 0)
  {
    throw std::invalid_argument("max_datagram_size must be greater than zero");
  }
}

XsensMvnStreamReader::~XsensMvnStreamReader()
{
  stop();
}

void XsensMvnStreamReader::start()
{
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true))
  {
    throw std::logic_error("XsensMvnStreamReader is already running");
  }

  socket_->bindToPort(config_.udp_port);
  socket_->setReceiveTimeout(config_.receive_timeout_ms);
  worker_thread_ = std::thread(&XsensMvnStreamReader::runLoop, this);
}

void XsensMvnStreamReader::stop()
{
  const bool was_running = running_.exchange(false);
  if (socket_)
  {
    socket_->closeSocket();
  }
  if (was_running && worker_thread_.joinable())
  {
    worker_thread_.join();
  }
  else if (worker_thread_.joinable())
  {
    worker_thread_.join();
  }
}

bool XsensMvnStreamReader::isRunning() const
{
  return running_.load();
}

bool XsensMvnStreamReader::hasFrame() const
{
  return latest_frame_store_.hasFrame();
}

std::uint64_t XsensMvnStreamReader::sequence() const
{
  return latest_frame_store_.sequence();
}

XsensRawFrame XsensMvnStreamReader::latestFrame() const
{
  return latest_frame_store_.snapshot();
}

void XsensMvnStreamReader::runLoop()
{
  std::vector<char> buffer(config_.max_datagram_size);
  while (running_.load())
  {
    try
    {
      const int bytes_received = socket_->receive(buffer.data(), buffer.size());
      if (bytes_received <= 0)
      {
        continue;
      }

      const bool updated = parser_->parse(
        buffer.data(), static_cast<std::size_t>(bytes_received), *assembler_);
      if (updated)
      {
        latest_frame_store_.update(assembler_->snapshot());
      }
    }
    catch (const std::exception&)
    {
      if (running_.load())
      {
        running_.store(false);
      }
    }
  }
}

}  // namespace xsens_mvn_cpp
