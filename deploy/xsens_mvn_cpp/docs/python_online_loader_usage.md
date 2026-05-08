# Python Online Loader Usage

## Runtime Path

The online human-motion path is:

1. `xsens_mvn_cpp_py.XsensMvnStreamReader` receives and parses UDP datagrams in
   C++ and keeps only the latest raw frame.
2. `XsensPybindLatestFrameSource` wraps the pybind reader lifecycle for Python.
3. `XsensRawFrameHumanMotionAdapter` maps raw segment world poses and raw
   streamed joint angles into the desired human joint order.
4. `OnlineHumanMotionLoader` samples the latest frame at the consumer rate and
   maintains a fixed-size window.
5. `OnlineHumanMotionSimulator` consumes the latest sample and fixed window in
   the existing `_obs_*` methods.

The pybind layer does not expose a consumed/unconsumed frame contract. The
online loader intentionally reads the latest frame every refresh, even if the
underlying Xsens sequence number did not change.

## Optional Sample Transform

The default online path does not load offline motion files and does not apply a
runtime pose alignment. If a fixed 90-degree or 180-degree axis rotation is
needed later, pass an object with `apply(sample) -> HumanMotionSample` through
`OnlineHumanMotionLoader(sample_transformer=...)` or
`OnlineHumanMotionSimulator(human_sample_transformer=...)`.

That transform belongs in the consumer layer. The pybind receiver still keeps
only the latest raw frame, and the adapter still preserves source Xsens
semantics.

Each sampled human frame contains the three arrays expected by the online human
observation path:

- `human_body_pos_w`: segment world positions;
- `human_body_quat_w`: segment world quaternions;
- `human_joint_quat`: local joint quaternions converted from streamed MVN
  joint angles;
- `human_joint_angles`: raw streamed MVN joint-angle triplets.

It also contains `human_joint_angles`, which is mapped directly from MVN
`Joint Angles` type-20 datagrams by parent-child segment id. These values are
raw streamed angle triplets and are not converted by the pybind layer.

`human_joint_quat` is not produced by the pybind layer. It is a consumer-layer
conversion from `human_joint_angles`, which keeps the C++ stream parser free of
retargeting policy. `human_joint_angles` is the direct streamed joint-angle
description and should be used when comparing against Xsens joint-angle
semantics.

## Window Semantics

At initialization, the loader waits for the first valid frame and fills the
entire window with that sample:

```text
[frame0, frame0, frame0, ..., frame0]
```

Before each observation update, the loader refreshes the window by shifting left
and appending the current latest sample:

```text
old: [a, b, c]
new: [b, c, latest]
```

This means the online time base is the simulator or policy loop, not the Xsens
packet sequence counter.

## CLI Probe

Build the pybind module, then run:

```bash
source ~/miniconda3/bin/activate hpx_g1
PYTHONPATH=build/xsens_mvn_cpp python -m deploy.xsens_mvn_cpp.cli \
  --udp-port 8001 \
  --samples 5
```

The CLI prints frame id, timestamp, desired-joint valid ratio, and the configured
human anchor pose.

## Simulator Entry

Run the online MuJoCo simulator with the pybind build directory on `PYTHONPATH`:

```bash
source ~/miniconda3/bin/activate hpx_g1
PYTHONPATH=build/xsens_mvn_cpp python deploy/deploy_mujoco/deploy_g1_mujoco_online.py
```

The simulator keeps the original base `simulator` policy-loop semantics as much
as possible. Only the online human motion source and human-reference observation
terms are overridden.
