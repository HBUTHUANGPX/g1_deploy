#include "xsens_mvn_cpp/latest_frame_store.h"

namespace xsens_mvn_cpp
{

void LatestFrameStore::update(const XsensRawFrame& frame)
{
  std::lock_guard<std::mutex> lock(mutex_);
  latest_frame_ = frame;
  ++sequence_;
  latest_frame_.sequence = sequence_;
  has_frame_ = true;
}

bool LatestFrameStore::hasFrame() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return has_frame_;
}

std::uint64_t LatestFrameStore::sequence() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return sequence_;
}

XsensRawFrame LatestFrameStore::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_frame_;
}

}  // namespace xsens_mvn_cpp
