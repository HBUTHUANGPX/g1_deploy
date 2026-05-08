from __future__ import annotations

import time

import numpy as np

from deploy.xsens_mvn_cpp.types import HumanMotionSample, HumanMotionWindow


class OnlineHumanMotionLoader:
    """Fixed-rate online human-motion loader backed by a latest-frame source.

    Preconditions:
    - `source` exposes `start`, `close`, `has_frame`, and `latest_frame`.
    - `adapter` exposes `to_human_sample(frame)`.

    Postconditions:
    - The loader owns a fixed-size window sampled at the consumer's refresh rate.
    - It does not depend on whether the source sequence changed between refreshes.
    """

    def __init__(
        self,
        source,
        adapter,
        window_size: int,
        initialization_timeout_s: float = 5.0,
        poll_interval_s: float = 0.002,
    ):
        """Create an online loader.

        Preconditions:
        - `window_size` is positive.
        - `initialization_timeout_s` is non-negative.

        Postconditions:
        - No data is consumed until `initialize` or `refresh` is called.
        """

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if initialization_timeout_s < 0:
            raise ValueError("initialization_timeout_s must be non-negative")

        self.source = source
        self.adapter = adapter
        self.window_size = int(window_size)
        self.initialization_timeout_s = float(initialization_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self._window_samples: list[HumanMotionSample] = []
        self._started = False

    def start(self) -> None:
        """Start the underlying latest-frame source.

        Preconditions:
        - The source is not required to be stopped.

        Postconditions:
        - The source has received a `start` call exactly once from this loader.
        """

        if not self._started:
            self.source.start()
            self._started = True

    def close(self) -> None:
        """Close the underlying latest-frame source.

        Preconditions:
        - None.

        Postconditions:
        - Source resources are released.
        """

        self.source.close()
        self._started = False

    def initialize(self) -> None:
        """Fill the whole window with the first available latest frame.

        Preconditions:
        - `start` has been called or the source is otherwise active.

        Postconditions:
        - `is_initialized` returns true.
        - Every slot in the window contains the first sampled frame.
        """

        sample = self._wait_for_sample()
        self._window_samples = [sample] * self.window_size

    def refresh(self) -> None:
        """Sample the latest frame and shift the fixed window left by one slot.

        Preconditions:
        - `initialize` has completed.

        Postconditions:
        - The window length remains constant.
        - The newest sampled frame is stored in the last slot.
        """

        if not self.is_initialized:
            self.initialize()
            return

        sample = self._read_latest_sample()
        self._window_samples = self._window_samples[1:] + [sample]

    @property
    def is_initialized(self) -> bool:
        """Return whether the fixed-size window has been initialized.

        Preconditions:
        - None.

        Postconditions:
        - Internal state is unchanged.
        """

        return len(self._window_samples) == self.window_size

    def latest_sample(self) -> HumanMotionSample | None:
        """Return the newest sample in the online window.

        Preconditions:
        - None.

        Postconditions:
        - Returns `None` if the loader has not been initialized.
        """

        if not self._window_samples:
            return None
        return self._window_samples[-1]

    def window(self) -> HumanMotionWindow:
        """Return the current fixed-size human-motion window.

        Preconditions:
        - `initialize` has completed.

        Postconditions:
        - Returned arrays are copies assembled from the current sample window.
        """

        if not self.is_initialized:
            raise RuntimeError("online human motion loader is not initialized")
        return self._window_from_samples(self._window_samples)

    def _wait_for_sample(self) -> HumanMotionSample:
        deadline = time.monotonic() + self.initialization_timeout_s
        while True:
            if self.source.has_frame():
                return self._read_latest_sample()
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the first Xsens frame")
            time.sleep(self.poll_interval_s)

    def _read_latest_sample(self) -> HumanMotionSample:
        return self.adapter.to_human_sample(self.source.latest_frame())

    @staticmethod
    def _window_from_samples(samples: list[HumanMotionSample]) -> HumanMotionWindow:
        latest = samples[-1]
        return HumanMotionWindow(
            joint_names=list(latest.joint_names),
            human_body_pos_w=np.stack(
                [sample.human_body_pos_w for sample in samples],
                axis=0,
            ),
            human_body_quat_w=np.stack(
                [sample.human_body_quat_w for sample in samples],
                axis=0,
            ),
            human_joint_quat=np.stack(
                [sample.human_joint_quat for sample in samples],
                axis=0,
            ),
            valid_mask=np.stack([sample.valid_mask for sample in samples], axis=0),
        )
