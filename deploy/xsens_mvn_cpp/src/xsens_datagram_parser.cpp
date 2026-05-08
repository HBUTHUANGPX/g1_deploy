#include "xsens_mvn_cpp/xsens_datagram_parser.h"

#include <stdexcept>

#include "xsens_mvn_sdk/datagram.h"

namespace xsens_mvn_cpp
{

XsensDatagramParser::XsensDatagramParser()
  : parser_manager_(false, false)
{
}

bool XsensDatagramParser::parse(
  const char* data,
  std::size_t data_size,
  XsensRawFrameAssembler& assembler)
{
  if (data == nullptr)
  {
    throw std::invalid_argument("data must not be null");
  }
  if (data_size == 0)
  {
    return false;
  }

  const auto datagram_type = static_cast<StreamingProtocol>(Datagram::messageType(data));
  parser_manager_.readDatagram(data);

  switch (datagram_type)
  {
    case SPPoseQuaternion:
      return parseQuaternionDatagram(assembler);
    case SPJointAngles:
      return parseJointAnglesDatagram(assembler);
    case SPLinearSegmentKinematics:
      return parseLinearSegmentKinematicsDatagram(assembler);
    case SPAngularSegmentKinematics:
      return parseAngularSegmentKinematicsDatagram(assembler);
    case SPCenterOfMass:
      return parseCenterOfMassDatagram(assembler);
    default:
      return false;
  }
}

bool XsensDatagramParser::parseQuaternionDatagram(XsensRawFrameAssembler& assembler)
{
  auto* datagram = parser_manager_.getQuaternionDatagram();
  if (datagram == nullptr)
  {
    return false;
  }
  for (const auto& item : datagram->getData())
  {
    assembler.updateSegmentPose(item);
  }
  return assembler.markSegmentPoseDatagram(
    datagram->sampleCounter(), datagram->frameTime());
}

bool XsensDatagramParser::parseJointAnglesDatagram(XsensRawFrameAssembler& assembler)
{
  auto* datagram = parser_manager_.getJointAnglesDatagram();
  if (datagram == nullptr)
  {
    return false;
  }
  for (const auto& item : datagram->getData())
  {
    assembler.updateJointAngles(item);
  }
  return assembler.markJointAnglesDatagram(
    datagram->sampleCounter(), datagram->frameTime());
}

bool XsensDatagramParser::parseLinearSegmentKinematicsDatagram(
  XsensRawFrameAssembler& assembler)
{
  auto* datagram = parser_manager_.getLinearSegmentKinematicsDatagram();
  if (datagram == nullptr)
  {
    return false;
  }
  assembler.updateDatagramMetadata(datagram->sampleCounter(), datagram->frameTime());
  for (const auto& item : datagram->getData())
  {
    assembler.updateSegmentLinearKinematics(item);
  }
  return false;
}

bool XsensDatagramParser::parseAngularSegmentKinematicsDatagram(
  XsensRawFrameAssembler& assembler)
{
  auto* datagram = parser_manager_.getAngularSegmentKinematicsDatagram();
  if (datagram == nullptr)
  {
    return false;
  }
  assembler.updateDatagramMetadata(datagram->sampleCounter(), datagram->frameTime());
  for (const auto& item : datagram->getData())
  {
    assembler.updateSegmentAngularKinematics(item);
  }
  return false;
}

bool XsensDatagramParser::parseCenterOfMassDatagram(XsensRawFrameAssembler& assembler)
{
  auto* datagram = parser_manager_.getCenterOfMassDatagram();
  if (datagram == nullptr)
  {
    return false;
  }
  assembler.updateDatagramMetadata(datagram->sampleCounter(), datagram->frameTime());
  const float* center_of_mass = datagram->getData();
  assembler.updateCenterOfMass(
    Vector3{center_of_mass[0], center_of_mass[1], center_of_mass[2]});
  return false;
}

}  // namespace xsens_mvn_cpp
