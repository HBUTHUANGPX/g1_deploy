# Xsens MVN C++ Latest-Frame Receiver

## Goal

This module receives Xsens MVN UDP datagrams at high frequency, parses them in C++, and keeps only the latest raw frame. It does not forward data, does not keep multi-frame buffers, and does not convert posture descriptions.

The Python consumer obtains the latest frame through pybind11 and is responsible for:

- buffering historical or future windows;
- mapping Xsens names to policy names;
- converting Euler angles, local quaternions, or other posture descriptions;
- deciding whether to consume human motion or robot motion.

## Vendored Parser Code

The module vendors the MVN SDK parser sources under:

`deploy/xsens_mvn_cpp/third_party/xsens_mvn_sdk`

It intentionally does not depend on the old `xsens_mvn_refactor_build` project or
its ZMQ/protobuf transport layer. The vendored parser can be upgraded by copying a
new parser snapshot into `third_party/xsens_mvn_sdk` and rebuilding this module.

## Class Design

- `UdpDatagramSocket`: RAII UDP socket wrapper with receive timeout.
- `XsensDatagramParser`: dispatches one datagram into supported parser outputs.
- `XsensRawFrameAssembler`: incrementally updates latest segment, joint, and COM fields.
- `LatestFrameStore`: thread-safe latest-frame-only storage.
- `XsensMvnStreamReader`: owns the worker thread and composes the socket, parser, assembler, and frame store.

The design favors composition over inheritance. Each class has one responsibility and exposes a narrow interface.

## Data Semantics

`XsensRawFrame` contains:

- `segments`: raw segment/link pose and kinematics.
- `joints`: raw joint-angle entries from the Xsens joint-angle datagram.
- `center_of_mass`: raw center of mass when available.
- `sample_counter` and `frame_time`: metadata copied from the latest parsed datagram.
- `sequence`: local latest-frame-store sequence, incremented whenever a supported datagram updates the frame.

Quaternion fields are exposed as `w, x, y, z` components. The represented orientation is not transformed.

## Build

From the repository root, using the required conda environment:

```bash
source ~/miniconda3/bin/activate hpx_g1
cmake -S deploy/xsens_mvn_cpp -B build/xsens_mvn_cpp \
  -DXSENS_MVN_CPP_BUILD_PYTHON=ON \
  -DXSENS_MVN_CPP_BUILD_TESTS=ON
cmake --build build/xsens_mvn_cpp -j2
ctest --test-dir build/xsens_mvn_cpp --output-on-failure
```

If pybind11 is not installed in `hpx_g1`:

```bash
source ~/miniconda3/bin/activate hpx_g1
python -m pip install pybind11
```

## Python Usage

```python
import xsens_mvn_cpp_py

with xsens_mvn_cpp_py.XsensMvnStreamReader(
    udp_port=8001,
    receive_timeout_ms=2,
    max_datagram_size=5000,
) as reader:
    if reader.has_frame():
        frame = reader.latest_frame()
        print(frame.sequence, len(frame.segments), len(frame.joints))
```

The module path is the CMake build output directory unless it is copied or installed into a Python package path.
