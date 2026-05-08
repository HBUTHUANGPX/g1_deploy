#ifndef DEPLOY_XSENS_MVN_CPP_XSENS_RAW_FRAME_ASSEMBLER_H
#define DEPLOY_XSENS_MVN_CPP_XSENS_RAW_FRAME_ASSEMBLER_H

#include <map>

#include "xsens_mvn_cpp/raw_frame_types.h"

struct JointAngle;
struct quaternionKinematics;
struct linearSegmentKinematics;
struct angularSegmentKinematics;

namespace xsens_mvn_cpp
{

/**
 * @brief Incrementally assembles latest raw Xsens segment and joint states.
 *
 * Preconditions:
 * - Datagram items have already been parsed by the Xsens MVN SDK parser.
 *
 * Postconditions:
 * - Only the most recent value per segment/joint name is retained.
 * - No coordinate-frame or posture-description conversion is performed.
 */
class XsensRawFrameAssembler
{
public:
  /**
   * @brief Construct an empty raw frame assembler.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - snapshot() returns an empty frame.
   */
  XsensRawFrameAssembler() = default;

  /**
   * @brief Update segment pose from a quaternion datagram item.
   *
   * Preconditions:
   * - item contains a valid Xsens segment id.
   *
   * Postconditions:
   * - The segment's position and orientation fields are updated.
   */
  void updateSegmentPose(const quaternionKinematics& item);

  /**
   * @brief Update segment linear velocity and acceleration.
   *
   * Preconditions:
   * - item contains a valid Xsens segment id.
   *
   * Postconditions:
   * - The segment's linear kinematic fields are updated.
   */
  void updateSegmentLinearKinematics(const linearSegmentKinematics& item);

  /**
   * @brief Update segment angular velocity and acceleration.
   *
   * Preconditions:
   * - item contains a valid Xsens segment id.
   *
   * Postconditions:
   * - The segment's angular kinematic fields are updated.
   */
  void updateSegmentAngularKinematics(const angularSegmentKinematics& item);

  /**
   * @brief Update joint angle state from a joint-angle datagram item.
   *
   * Preconditions:
   * - item contains parent and child segment identifiers.
   *
   * Postconditions:
   * - The matching joint angle state is updated.
   */
  void updateJointAngles(const JointAngle& item);

  /**
   * @brief Update center of mass.
   *
   * Preconditions:
   * - center_of_mass contains Xsens stream coordinates.
   *
   * Postconditions:
   * - The latest frame center of mass is updated.
   */
  void updateCenterOfMass(const Vector3& center_of_mass);

  /**
   * @brief Update source datagram timing metadata.
   *
   * Preconditions:
   * - Values are copied from a parsed Xsens datagram header.
   *
   * Postconditions:
   * - The latest frame metadata is updated.
   */
  void updateDatagramMetadata(int sample_counter, int frame_time);

  /**
   * @brief Return a value snapshot of the assembled latest frame.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  XsensRawFrame snapshot() const;

private:
  XsensRawSegmentState& mutableSegment(int segment_id);
  XsensRawJointState& mutableJoint(int parent_segment_id, int child_segment_id);
  std::string segmentName(int segment_id) const;
  std::string jointName(int parent_segment_id, int child_segment_id) const;

  int prop_count_ = 0;
  XsensRawFrame latest_frame_;
  std::map<int, XsensRawSegmentState> segments_by_id_;
  std::map<std::pair<int, int>, XsensRawJointState> joints_by_pair_;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_XSENS_RAW_FRAME_ASSEMBLER_H
