#include <gtest/gtest.h>

#include "xsens_mvn_cpp/latest_frame_store.h"
#include "xsens_mvn_cpp/xsens_raw_frame_assembler.h"

TEST(LatestFrameStoreTest, StartsWithoutFrame)
{
  xsens_mvn_cpp::LatestFrameStore store;

  EXPECT_FALSE(store.hasFrame());
  EXPECT_EQ(store.sequence(), 0U);
}

TEST(LatestFrameStoreTest, KeepsOnlyLatestSnapshot)
{
  xsens_mvn_cpp::LatestFrameStore store;

  xsens_mvn_cpp::XsensRawFrame first_frame;
  first_frame.segments.push_back(xsens_mvn_cpp::XsensRawSegmentState{});
  first_frame.segments.back().name = "pelvis";
  first_frame.segments.back().position = {1.0, 2.0, 3.0};

  xsens_mvn_cpp::XsensRawFrame second_frame;
  second_frame.segments.push_back(xsens_mvn_cpp::XsensRawSegmentState{});
  second_frame.segments.back().name = "head";
  second_frame.segments.back().position = {4.0, 5.0, 6.0};

  store.update(first_frame);
  store.update(second_frame);

  ASSERT_TRUE(store.hasFrame());
  EXPECT_EQ(store.sequence(), 2U);

  const auto snapshot = store.snapshot();
  ASSERT_EQ(snapshot.segments.size(), 1U);
  EXPECT_EQ(snapshot.segments.front().name, "head");
  EXPECT_DOUBLE_EQ(snapshot.segments.front().position.x, 4.0);
}

TEST(LatestFrameStoreTest, SnapshotIsACopy)
{
  xsens_mvn_cpp::LatestFrameStore store;
  xsens_mvn_cpp::XsensRawFrame frame;
  frame.joints.push_back(xsens_mvn_cpp::XsensRawJointState{});
  frame.joints.back().name = "left_knee";

  store.update(frame);
  auto snapshot = store.snapshot();
  snapshot.joints.front().name = "mutated";

  EXPECT_EQ(store.snapshot().joints.front().name, "left_knee");
}

TEST(XsensRawFrameAssemblerTest, PublishesOnlyWhenPoseAndJointAnglesShareSampleCounter)
{
  xsens_mvn_cpp::XsensRawFrameAssembler assembler;

  EXPECT_FALSE(assembler.markSegmentPoseDatagram(10, 1000));
  EXPECT_FALSE(assembler.markJointAnglesDatagram(11, 1004));
  EXPECT_TRUE(assembler.markSegmentPoseDatagram(11, 1004));
  EXPECT_FALSE(assembler.markJointAnglesDatagram(11, 1004));

  const auto snapshot = assembler.snapshot();
  EXPECT_EQ(snapshot.sample_counter, 11);
  EXPECT_EQ(snapshot.frame_time, 1004);
}
