# Xsens MVN Streamed Data Mapping

## Purpose

This document describes how Xsens MVN real-time network streaming packets map to
the `xsens_mvn_cpp` latest-frame receiver. It is intentionally focused on raw
stream parsing. The receiver keeps only the newest parsed frame and does not
perform posture conversion, retargeting, buffering, or forwarding.

The protocol reference is:

`MVN_real-time_network_streaming_protocol_specification-1.pdf`

## Receiver Boundary

`xsens_mvn_cpp` owns these responsibilities:

- receive complete UDP datagrams from MVN Analyze/Animate;
- parse supported datagram payloads as quickly as possible;
- copy parsed values into a latest-frame snapshot;
- expose the latest snapshot through pybind11;
- preserve source coordinate and posture conventions.

Consumers own these responsibilities:

- multi-frame buffering and interpolation;
- mapping Xsens segment names or ids to policy or robot names;
- converting joint angles to quaternions;
- converting coordinate systems or applying fixed convention offsets;
- filtering body, prop, finger, or object tracking segments;
- deciding whether a frame should be interpreted as human motion or robot motion.

## Protocol Header

Each MVN datagram starts with the `MXTP` protocol header. The receiver uses the
header message type to dispatch packets. Important header fields include:

- id string;
- sample counter;
- datagram counter;
- number of items;
- time code in milliseconds;
- character id;
- number of body segments;
- number of props;
- number of finger tracking data segments;
- payload size.

The current frame model exposes `sample_counter` and `frame_time`. Character id,
body segment count, prop count, finger segment count, and raw payload size are
kept as future compatibility points and should not be assumed constant by
consumer code.

## Supported Datagram Types

| MVN option | Type | Current parser status | Receiver output |
| --- | --- | --- | --- |
| Position + Orientation (Quaternion) | `02` | Supported | `segments[].position`, `segments[].orientation` |
| Joint Angles | `20` | Supported | `joints[].parent_segment_id`, `joints[].child_segment_id`, `joints[].angles` |
| Linear Segment Kinematics | `21` | Supported | `segments[].position`, `segments[].linear_velocity`, `segments[].linear_acceleration` |
| Angular Segment Kinematics | `22` | Supported | `segments[].orientation`, `segments[].angular_velocity`, `segments[].angular_acceleration` |
| Center of Mass | `24` | Supported | `center_of_mass` |
| Position + Orientation (Euler) | `01` | Not exposed | Future optional parser |
| Virtual Optical Marker Set | `03` | Not exposed | Future optional parser |
| Unity 3D | `05` | Not exposed | Future optional parser |
| Character Meta Data | `12` | Not exposed | Future metadata parser |
| Scaling Data | `13` | Not exposed | Future skeleton or mesh-scale parser |
| Motion Tracker Kinematics | `23` | Not exposed | Future sensor-level parser |
| Time Code string | `25` | Not exposed | Future timestamp parser |
| Network Sync JSON/XML | N/A | Not handled | Out of scope for latest-frame motion parsing |
| Siemens Tecnomatix | N/A | Not handled | Out of scope |

## Segment World Pose

MVN `Position + Orientation (Quaternion)` packets are the preferred source for
human segment world poses.

For each segment, MVN sends:

- segment id;
- world position `x, y, z`, in meters;
- world orientation quaternion `q1, q2, q3, q4`.

The receiver exposes this as:

- `XsensRawSegmentState.segment_id`;
- `XsensRawSegmentState.name`;
- `XsensRawSegmentState.position`;
- `XsensRawSegmentState.orientation`.

Quaternion values are exposed in `w, x, y, z` order. The receiver does not
renormalize, rotate axes, convert handedness, or convert to Euler angles.

The live test on the current LAN stream received 63 segments. Consumers must not
hard-code the body segment count as 23. A stream may include regular body
segments, props, finger tracking segments, or object tracking data depending on
MVN streamer settings.

## Joint Angles

MVN `Joint Angles` packets contain parent and child point ids plus three angle
components.

For each joint entry, MVN sends:

- parent point id;
- child point id;
- rotation around segment x axis;
- rotation around segment y axis;
- rotation around segment z axis.

The receiver exposes this as:

- `XsensRawJointState.parent_segment_id`;
- `XsensRawJointState.child_segment_id`;
- `XsensRawJointState.name`;
- `XsensRawJointState.angles`.

The Python online human adapter maps these raw entries into
`HumanMotionSample.human_joint_angles` using explicit MVN parent-child segment
pairs. The mapped angles remain raw streamed angle triplets.

These are raw `x, y, z` joint angle values from the stream. They are not
quaternions. Any `joint_relative_quaternion` field should be derived in the
consumer from either:

- parent and child segment world quaternions; or
- the raw joint angle triplet and a consumer-owned rotation convention.

This boundary is deliberate. It keeps the receiver independent from retargeting
policy assumptions and makes online human-motion and robot-motion consumers
easier to evolve separately.

The current Python human-motion consumer derives `human_joint_quat` from the
raw streamed angle triplets. Any fixed coordinate or quaternion convention
adjustment should be provided as a consumer-owned sample transform. The raw
`human_joint_angles` field must not be changed by that step.

Ergonomic joint angles are part of the same MVN stream. They have the same data
layout as regular joint angles, but the local point id is always `0`. Consumers
should filter or map them explicitly instead of assuming every entry maps to a
robot joint.

## Kinematic Streams

`Linear Segment Kinematics` and `Angular Segment Kinematics` are additional
streams. They do not define a complete pose by themselves, but they can enrich a
latest frame when MVN is configured to send them.

Linear segment kinematics provides:

- segment id;
- position;
- global linear velocity;
- global linear acceleration.

Angular segment kinematics provides:

- segment id;
- orientation quaternion;
- global angular velocity;
- global angular acceleration.

Consumers should treat these fields as optional. A valid online motion loader
should work when only type `02` and type `20` are available.

## Center of Mass

The receiver currently stores the center of mass position in `center_of_mass`.
The MVN protocol can also send center of mass velocity and acceleration in newer
MVN versions. Those fields are not exposed yet and should be treated as a future
extension point.

## Unsupported But Important Future Data

`Scaling Data` is important for visualization and mesh fitting. It is not
required for the first online human-motion loader if the consumer only uses
segment poses and joint angles. It should remain a future parser extension,
especially if visualization or skeleton fitting becomes part of the runtime
pipeline.

`Character Meta Data` can matter in multi-person streaming. The current receiver
does not expose character metadata, so online consumers should not assume
multi-character identity handling is solved yet.

`Motion Tracker Kinematics` is sensor-level data rather than body-segment pose.
It may become useful for diagnostics or raw sensor fusion, but it should not be
mixed into the first human-motion consumer path.

## Compatibility Guidelines For Online Motion Loader

- Prefer `Position + Orientation (Quaternion)` as the authoritative world pose
  source for human segments.
- Treat `Joint Angles` as raw angle triplets, not quaternions.
- Keep derived quaternion conversion in the consumer layer.
- Keep fixed 90-degree or 180-degree convention rotations in the consumer
  layer, not in the pybind receiver.
- Do not assume every frame contains every datagram type.
- Do not assume segment count, joint count, or segment ordering is constant.
- Use stable ids or names for mapping, not vector indices alone.
- Preserve unknown or unsupported datagrams by ignoring them safely.
- Keep robot-motion and human-motion consumers behind separate observation
  builders, even if they share the same latest-frame source interface.
