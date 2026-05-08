from __future__ import annotations

import argparse
import time

from deploy.utils.cfg import cfg
from deploy.xsens_mvn_cpp import (
    OnlineHumanMotionLoader,
    XsensPybindLatestFrameSource,
    XsensRawFrameHumanMotionAdapter,
)


class XsensMvnCppCli:
    """Command line probe for the pybind-backed Xsens latest-frame receiver.

    Preconditions:
    - The `xsens_mvn_cpp_py` extension module is importable.

    Postconditions:
    - The command prints online loader status and sample contents.
    """

    def run(self, argv: list[str] | None = None) -> int:
        """Run the CLI.

        Preconditions:
        - `argv` is either `None` or a list of command line arguments.

        Postconditions:
        - Returns a process-style integer exit code.
        """

        parser = argparse.ArgumentParser(
            description="Inspect Xsens MVN raw frames through xsens_mvn_cpp_py."
        )
        parser.add_argument("--udp-port", type=int, default=8001)
        parser.add_argument("--receive-timeout-ms", type=int, default=2)
        parser.add_argument("--max-datagram-size", type=int, default=5000)
        parser.add_argument("--init-timeout-s", type=float, default=5.0)
        parser.add_argument("--samples", type=int, default=5)
        parser.add_argument("--period-s", type=float, default=0.02)
        args = parser.parse_args(argv)

        source = XsensPybindLatestFrameSource(
            udp_port=args.udp_port,
            receive_timeout_ms=args.receive_timeout_ms,
            max_datagram_size=args.max_datagram_size,
        )
        adapter = XsensRawFrameHumanMotionAdapter(cfg.desire_human_joint_names)
        loader = OnlineHumanMotionLoader(
            source=source,
            adapter=adapter,
            window_size=cfg.history_frames + cfg.future_frames + 1,
            initialization_timeout_s=args.init_timeout_s,
        )

        loader.start()
        try:
            loader.initialize()
            previous_sample = None
            for _ in range(args.samples):
                loader.refresh()
                sample = loader.latest_sample()
                valid_ratio = float(sample.valid_mask.mean())
                joint_angle_valid_ratio = float(sample.joint_angle_valid_mask.mean())
                hips_index = sample.joint_names.index(cfg.human_anchor_name)
                hips_pos = sample.human_body_pos_w[hips_index]
                hips_quat = sample.human_body_quat_w[hips_index]
                hips_joint_angles = sample.human_joint_angles[hips_index]
                print(f"frame_id={sample.frame_id}")
                print(f"sample_counter={sample.source_sample_counter}")
                print(f"datagram_sequence={sample.source_datagram_sequence}")
                print(f"timestamp_ns={sample.timestamp_ns}")
                if previous_sample is not None:
                    timestamp_delta_ms = (
                        sample.timestamp_ns - previous_sample.timestamp_ns
                    ) / 1_000_000.0
                    sample_counter_delta = (
                        sample.source_sample_counter
                        - previous_sample.source_sample_counter
                    )
                    datagram_sequence_delta = (
                        sample.source_datagram_sequence
                        - previous_sample.source_datagram_sequence
                    )
                    print(f"delta_timestamp_ms={timestamp_delta_ms:.3f}")
                    print(f"delta_sample_counter={sample_counter_delta}")
                    print(f"delta_datagram_sequence={datagram_sequence_delta}")
                print(f"valid_ratio={valid_ratio:.3f}")
                print(f"joint_angle_valid_ratio={joint_angle_valid_ratio:.3f}")
                print(f"{cfg.human_anchor_name}_pos={hips_pos.tolist()}")
                print(f"{cfg.human_anchor_name}_quat_wxyz={hips_quat.tolist()}")
                print(f"{cfg.human_anchor_name}_joint_angles_xyz={hips_joint_angles.tolist()}")
                previous_sample = sample
                time.sleep(args.period_s)
        finally:
            loader.close()
        return 0


def main(argv: list[str] | None = None) -> int:
    """Run the pybind Xsens receiver CLI.

    Preconditions:
    - See `XsensMvnCppCli.run`.

    Postconditions:
    - Returns a process-style integer exit code.
    """

    return XsensMvnCppCli().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
