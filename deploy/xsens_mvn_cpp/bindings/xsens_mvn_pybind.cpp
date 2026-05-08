#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xsens_mvn_cpp/raw_frame_types.h"
#include "xsens_mvn_cpp/xsens_mvn_stream_reader.h"

namespace py = pybind11;

PYBIND11_MODULE(xsens_mvn_cpp_py, module)
{
  module.doc() =
    "High-frequency Xsens MVN UDP parser. The module stores only the latest raw frame.";

  py::class_<xsens_mvn_cpp::Vector3>(
    module,
    "Vector3",
    "Three-dimensional vector copied from the Xsens stream without coordinate conversion.")
    .def(py::init<>())
    .def_readwrite("x", &xsens_mvn_cpp::Vector3::x)
    .def_readwrite("y", &xsens_mvn_cpp::Vector3::y)
    .def_readwrite("z", &xsens_mvn_cpp::Vector3::z);

  py::class_<xsens_mvn_cpp::QuaternionWxyz>(
    module,
    "QuaternionWxyz",
    "Quaternion represented as w, x, y, z scalars.")
    .def(py::init<>())
    .def_readwrite("w", &xsens_mvn_cpp::QuaternionWxyz::w)
    .def_readwrite("x", &xsens_mvn_cpp::QuaternionWxyz::x)
    .def_readwrite("y", &xsens_mvn_cpp::QuaternionWxyz::y)
    .def_readwrite("z", &xsens_mvn_cpp::QuaternionWxyz::z);

  py::class_<xsens_mvn_cpp::XsensRawSegmentState>(
    module,
    "XsensRawSegmentState",
    "Latest raw state for one Xsens segment.")
    .def(py::init<>())
    .def_readwrite("name", &xsens_mvn_cpp::XsensRawSegmentState::name)
    .def_readwrite("segment_id", &xsens_mvn_cpp::XsensRawSegmentState::segment_id)
    .def_readwrite("position", &xsens_mvn_cpp::XsensRawSegmentState::position)
    .def_readwrite("orientation", &xsens_mvn_cpp::XsensRawSegmentState::orientation)
    .def_readwrite("linear_velocity", &xsens_mvn_cpp::XsensRawSegmentState::linear_velocity)
    .def_readwrite("angular_velocity", &xsens_mvn_cpp::XsensRawSegmentState::angular_velocity)
    .def_readwrite(
      "linear_acceleration", &xsens_mvn_cpp::XsensRawSegmentState::linear_acceleration)
    .def_readwrite(
      "angular_acceleration", &xsens_mvn_cpp::XsensRawSegmentState::angular_acceleration);

  py::class_<xsens_mvn_cpp::XsensRawJointState>(
    module,
    "XsensRawJointState",
    "Latest raw state for one Xsens joint-angle entry.")
    .def(py::init<>())
    .def_readwrite("name", &xsens_mvn_cpp::XsensRawJointState::name)
    .def_readwrite("parent_segment_id", &xsens_mvn_cpp::XsensRawJointState::parent_segment_id)
    .def_readwrite("child_segment_id", &xsens_mvn_cpp::XsensRawJointState::child_segment_id)
    .def_readwrite("angles", &xsens_mvn_cpp::XsensRawJointState::angles);

  py::class_<xsens_mvn_cpp::XsensRawFrame>(
    module,
    "XsensRawFrame",
    "Latest raw Xsens frame snapshot.")
    .def(py::init<>())
    .def_readwrite("sequence", &xsens_mvn_cpp::XsensRawFrame::sequence)
    .def_readwrite("sample_counter", &xsens_mvn_cpp::XsensRawFrame::sample_counter)
    .def_readwrite("frame_time", &xsens_mvn_cpp::XsensRawFrame::frame_time)
    .def_readwrite("segments", &xsens_mvn_cpp::XsensRawFrame::segments)
    .def_readwrite("joints", &xsens_mvn_cpp::XsensRawFrame::joints)
    .def_readwrite("center_of_mass", &xsens_mvn_cpp::XsensRawFrame::center_of_mass);

  py::class_<xsens_mvn_cpp::XsensMvnStreamConfig>(
    module,
    "XsensMvnStreamConfig",
    "Configuration for Xsens MVN UDP stream ingestion.")
    .def(py::init<>())
    .def_readwrite("udp_port", &xsens_mvn_cpp::XsensMvnStreamConfig::udp_port)
    .def_readwrite(
      "receive_timeout_ms", &xsens_mvn_cpp::XsensMvnStreamConfig::receive_timeout_ms)
    .def_readwrite("max_datagram_size", &xsens_mvn_cpp::XsensMvnStreamConfig::max_datagram_size);

  py::class_<xsens_mvn_cpp::XsensMvnStreamReader>(
    module,
    "XsensMvnStreamReader",
    "Threaded Xsens MVN UDP reader that parses datagrams and keeps only the latest frame.")
    .def(py::init<const xsens_mvn_cpp::XsensMvnStreamConfig&>(), py::arg("config"))
    .def(
      py::init([](std::uint16_t udp_port, int receive_timeout_ms, std::size_t max_datagram_size) {
        xsens_mvn_cpp::XsensMvnStreamConfig config;
        config.udp_port = udp_port;
        config.receive_timeout_ms = receive_timeout_ms;
        config.max_datagram_size = max_datagram_size;
        return new xsens_mvn_cpp::XsensMvnStreamReader(config);
      }),
      py::arg("udp_port") = 8001,
      py::arg("receive_timeout_ms") = 2,
      py::arg("max_datagram_size") = 5000)
    .def("start", &xsens_mvn_cpp::XsensMvnStreamReader::start, "Start UDP parsing.")
    .def("stop", &xsens_mvn_cpp::XsensMvnStreamReader::stop, "Stop UDP parsing.")
    .def("is_running", &xsens_mvn_cpp::XsensMvnStreamReader::isRunning)
    .def("has_frame", &xsens_mvn_cpp::XsensMvnStreamReader::hasFrame)
    .def("sequence", &xsens_mvn_cpp::XsensMvnStreamReader::sequence)
    .def("latest_frame", &xsens_mvn_cpp::XsensMvnStreamReader::latestFrame)
    .def(
      "__enter__",
      [](xsens_mvn_cpp::XsensMvnStreamReader& self)
        -> xsens_mvn_cpp::XsensMvnStreamReader& {
        self.start();
        return self;
      },
      py::return_value_policy::reference_internal)
    .def(
      "__exit__",
      [](xsens_mvn_cpp::XsensMvnStreamReader& self, py::object, py::object, py::object) {
        self.stop();
      });
}
