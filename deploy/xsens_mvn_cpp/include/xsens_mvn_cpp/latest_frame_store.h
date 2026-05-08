#ifndef DEPLOY_XSENS_MVN_CPP_LATEST_FRAME_STORE_H
#define DEPLOY_XSENS_MVN_CPP_LATEST_FRAME_STORE_H

#include <cstdint>
#include <mutex>

#include "xsens_mvn_cpp/raw_frame_types.h"

namespace xsens_mvn_cpp
{

/**
 * @brief Thread-safe holder for the most recent Xsens frame only.
 *
 * Preconditions:
 * - Writers provide complete value-type frame snapshots.
 *
 * Postconditions:
 * - Only the latest frame is retained.
 * - Readers receive a copy and cannot mutate internal state.
 */
class LatestFrameStore
{
public:
  /**
   * @brief Construct an empty latest-frame store.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - hasFrame() returns false and sequence() returns 0.
   */
  LatestFrameStore() = default;

  /**
   * @brief Replace the currently stored frame with a new snapshot.
   *
   * Preconditions:
   * - frame is a value snapshot owned by the caller.
   *
   * Postconditions:
   * - hasFrame() returns true.
   * - sequence() increments by one.
   */
  void update(const XsensRawFrame& frame);

  /**
   * @brief Return whether at least one frame has been stored.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  bool hasFrame() const;

  /**
   * @brief Return the latest store-level sequence number.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  std::uint64_t sequence() const;

  /**
   * @brief Return a copy of the latest frame.
   *
   * Preconditions:
   * - hasFrame() is true for meaningful data.
   *
   * Postconditions:
   * - Internal state is unchanged.
   * - The returned frame is independent of the store.
   */
  XsensRawFrame snapshot() const;

private:
  mutable std::mutex mutex_;
  bool has_frame_ = false;
  std::uint64_t sequence_ = 0;
  XsensRawFrame latest_frame_;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_LATEST_FRAME_STORE_H
