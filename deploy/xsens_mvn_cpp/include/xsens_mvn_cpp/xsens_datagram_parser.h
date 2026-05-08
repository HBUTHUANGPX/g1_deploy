#ifndef DEPLOY_XSENS_MVN_CPP_XSENS_DATAGRAM_PARSER_H
#define DEPLOY_XSENS_MVN_CPP_XSENS_DATAGRAM_PARSER_H

#include <cstddef>

#include "xsens_mvn_cpp/xsens_raw_frame_assembler.h"
#include "xsens_mvn_sdk/parsermanager.h"

namespace xsens_mvn_cpp
{

/**
 * @brief Parses raw Xsens MVN datagrams into an incremental raw frame assembler.
 *
 * Preconditions:
 * - Input buffers contain complete Xsens MVN datagrams.
 *
 * Postconditions:
 * - The assembler is updated only with fields present in the parsed datagram.
 * - No forwarding, buffering, or posture-description conversion is performed.
 */
class XsensDatagramParser
{
public:
  /**
   * @brief Construct a parser with printing disabled.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - The parser is ready to parse datagrams.
   */
  XsensDatagramParser();

  /**
   * @brief Parse one datagram and update the provided assembler.
   *
   * Preconditions:
   * - data points to a complete Xsens datagram.
   * - data_size is the number of valid bytes in data.
   *
   * Postconditions:
   * - Returns true if the datagram type was supported and applied.
   * - Returns false for unsupported datagram types.
   */
  bool parse(const char* data, std::size_t data_size, XsensRawFrameAssembler& assembler);

private:
  bool parseQuaternionDatagram(XsensRawFrameAssembler& assembler);
  bool parseJointAnglesDatagram(XsensRawFrameAssembler& assembler);
  bool parseLinearSegmentKinematicsDatagram(XsensRawFrameAssembler& assembler);
  bool parseAngularSegmentKinematicsDatagram(XsensRawFrameAssembler& assembler);
  bool parseCenterOfMassDatagram(XsensRawFrameAssembler& assembler);

  ParserManager parser_manager_;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_XSENS_DATAGRAM_PARSER_H
