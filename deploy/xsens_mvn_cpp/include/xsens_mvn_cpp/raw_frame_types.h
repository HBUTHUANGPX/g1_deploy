#ifndef DEPLOY_XSENS_MVN_CPP_RAW_FRAME_TYPES_H
#define DEPLOY_XSENS_MVN_CPP_RAW_FRAME_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace xsens_mvn_cpp
{

/**
 * @brief Three-dimensional vector transported without coordinate conversion.
 *
 * Preconditions:
 * - Values are expressed in the source Xsens stream convention.
 *
 * Postconditions:
 * - The struct contains exactly three scalar components.
 */
struct Vector3
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

/**
 * @brief Quaternion transported in w, x, y, z component order.
 *
 * Preconditions:
 * - Values are copied from Xsens datagrams without changing the represented
 *   rotation.
 *
 * Postconditions:
 * - The struct contains exactly four scalar components.
 */
struct QuaternionWxyz
{
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

/**
 * @brief Latest raw state for one Xsens segment/link.
 *
 * Preconditions:
 * - The segment name is resolved from the Xsens segment id.
 *
 * Postconditions:
 * - Pose and kinematic fields reflect the most recently parsed datagrams for
 *   this segment.
 */
struct XsensRawSegmentState
{
  std::string name;
  int segment_id = 0;
  Vector3 position;
  QuaternionWxyz orientation;
  Vector3 linear_velocity;
  Vector3 angular_velocity;
  Vector3 linear_acceleration;
  Vector3 angular_acceleration;
};

/**
 * @brief Latest raw state for one Xsens joint angle entry.
 *
 * Preconditions:
 * - Angles are read from the Xsens joint-angle datagram.
 *
 * Postconditions:
 * - No Euler-to-quaternion or coordinate-frame conversion is performed.
 */
struct XsensRawJointState
{
  std::string name;
  int parent_segment_id = 0;
  int child_segment_id = 0;
  Vector3 angles;
};

/**
 * @brief Single latest-frame snapshot assembled from Xsens MVN datagrams.
 *
 * Preconditions:
 * - The frame is assembled from one or more parsed datagrams.
 *
 * Postconditions:
 * - The frame owns its data and can be safely copied across thread or Python
 *   boundaries.
 */
struct XsensRawFrame
{
  std::uint64_t sequence = 0;
  int sample_counter = 0;
  int frame_time = 0;
  std::vector<XsensRawSegmentState> segments;
  std::vector<XsensRawJointState> joints;
  Vector3 center_of_mass;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_RAW_FRAME_TYPES_H
