#include "xsens_mvn_cpp/xsens_raw_frame_assembler.h"

#include <algorithm>

#include "xsens_mvn_sdk/MvnModel.h"
#include "xsens_mvn_sdk/angularsegmentkinematicsdatagram.h"
#include "xsens_mvn_sdk/jointanglesdatagram.h"
#include "xsens_mvn_sdk/linearsegmentkinematicsdatagram.h"
#include "xsens_mvn_sdk/quaterniondatagram.h"

namespace xsens_mvn_cpp
{

namespace
{
Vector3 makeVector3(const float values[3])
{
  return Vector3{values[0], values[1], values[2]};
}
}  // namespace

void XsensRawFrameAssembler::updateSegmentPose(const quaternionKinematics& item)
{
  prop_count_ = std::max(prop_count_, item.segmentId >= 24 && item.segmentId <= 27 ? item.segmentId - 23 : prop_count_);
  auto& segment = mutableSegment(item.segmentId);
  segment.position = makeVector3(item.sensorPos);
  segment.orientation = QuaternionWxyz{
    item.quatRotation[0],
    item.quatRotation[1],
    item.quatRotation[2],
    item.quatRotation[3],
  };
}

void XsensRawFrameAssembler::updateSegmentLinearKinematics(const linearSegmentKinematics& item)
{
  auto& segment = mutableSegment(item.segmentId);
  segment.linear_velocity = makeVector3(item.velocity);
  segment.linear_acceleration = makeVector3(item.acceleration);
}

void XsensRawFrameAssembler::updateSegmentAngularKinematics(const angularSegmentKinematics& item)
{
  auto& segment = mutableSegment(item.segmentId);
  segment.angular_velocity = makeVector3(item.angularVeloc);
  segment.angular_acceleration = makeVector3(item.angularAccel);
}

void XsensRawFrameAssembler::updateJointAngles(const JointAngle& item)
{
  auto& joint = mutableJoint(item.parentSegmentId, item.childSegmentId);
  joint.angles = Vector3{item.rotation[0], item.rotation[1], item.rotation[2]};
}

void XsensRawFrameAssembler::updateCenterOfMass(const Vector3& center_of_mass)
{
  latest_frame_.center_of_mass = center_of_mass;
}

void XsensRawFrameAssembler::updateDatagramMetadata(int sample_counter, int frame_time)
{
  latest_frame_.sample_counter = sample_counter;
  latest_frame_.frame_time = frame_time;
}

bool XsensRawFrameAssembler::markSegmentPoseDatagram(int sample_counter, int frame_time)
{
  updateDatagramMetadata(sample_counter, frame_time);
  latest_segment_pose_sample_counter_ = sample_counter;
  return markPublishableMotionSampleIfComplete(sample_counter);
}

bool XsensRawFrameAssembler::markJointAnglesDatagram(int sample_counter, int frame_time)
{
  updateDatagramMetadata(sample_counter, frame_time);
  latest_joint_angles_sample_counter_ = sample_counter;
  return markPublishableMotionSampleIfComplete(sample_counter);
}

XsensRawFrame XsensRawFrameAssembler::snapshot() const
{
  XsensRawFrame frame = latest_frame_;
  frame.segments.clear();
  frame.joints.clear();

  for (const auto& entry : segments_by_id_)
  {
    frame.segments.push_back(entry.second);
  }
  for (const auto& entry : joints_by_pair_)
  {
    frame.joints.push_back(entry.second);
  }
  return frame;
}

XsensRawSegmentState& XsensRawFrameAssembler::mutableSegment(int segment_id)
{
  auto& segment = segments_by_id_[segment_id];
  segment.segment_id = segment_id;
  segment.name = segmentName(segment_id);
  return segment;
}

XsensRawJointState& XsensRawFrameAssembler::mutableJoint(int parent_segment_id, int child_segment_id)
{
  const auto key = std::make_pair(parent_segment_id, child_segment_id);
  auto& joint = joints_by_pair_[key];
  joint.parent_segment_id = parent_segment_id;
  joint.child_segment_id = child_segment_id;
  joint.name = jointName(parent_segment_id, child_segment_id);
  return joint;
}

bool XsensRawFrameAssembler::markPublishableMotionSampleIfComplete(int sample_counter)
{
  const bool has_segment_pose = latest_segment_pose_sample_counter_ == sample_counter;
  const bool has_joint_angles = latest_joint_angles_sample_counter_ == sample_counter;
  const bool already_published = published_motion_sample_counter_ == sample_counter;
  if (!has_segment_pose || !has_joint_angles || already_published)
  {
    return false;
  }
  published_motion_sample_counter_ = sample_counter;
  return true;
}

std::string XsensRawFrameAssembler::segmentName(int segment_id) const
{
  return MvnModelNames{}.getSegmentNameFromId(segment_id, prop_count_);
}

std::string XsensRawFrameAssembler::jointName(int parent_segment_id, int child_segment_id) const
{
  const auto parent_name = segmentName(parent_segment_id);
  const auto child_name = segmentName(child_segment_id);
  return parent_name + "_to_" + child_name;
}

}  // namespace xsens_mvn_cpp
