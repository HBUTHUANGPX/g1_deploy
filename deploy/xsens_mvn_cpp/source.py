from __future__ import annotations


class XsensPybindLatestFrameSource:
    """Lifecycle wrapper around `xsens_mvn_cpp_py.XsensMvnStreamReader`.

    Preconditions:
    - The `xsens_mvn_cpp_py` extension module is importable.
    - MVN streams UDP datagrams to `udp_port`.

    Postconditions:
    - Consumers can read the latest raw frame through a stable Python interface.
    """

    def __init__(
        self,
        udp_port: int = 8001,
        receive_timeout_ms: int = 2,
        max_datagram_size: int = 5000,
    ):
        """Create the latest-frame source.

        Preconditions:
        - `udp_port` is a valid UDP port.

        Postconditions:
        - The underlying reader is constructed but not started.
        """

        self.udp_port = int(udp_port)
        self.receive_timeout_ms = int(receive_timeout_ms)
        self.max_datagram_size = int(max_datagram_size)
        self._reader = self._create_reader()

    def start(self) -> None:
        """Start the pybind reader.

        Preconditions:
        - The UDP port is available.

        Postconditions:
        - The reader's background parsing loop is active.
        """

        self._reader.start()

    def close(self) -> None:
        """Stop the pybind reader.

        Preconditions:
        - None.

        Postconditions:
        - The reader's background parsing loop is stopped.
        """

        self._reader.stop()

    def has_frame(self) -> bool:
        """Return whether the pybind reader has received any supported frame.

        Preconditions:
        - None.

        Postconditions:
        - Internal reader state is unchanged.
        """

        return bool(self._reader.has_frame())

    def latest_frame(self):
        """Return the latest raw frame from the pybind reader.

        Preconditions:
        - `has_frame` is true for meaningful data.

        Postconditions:
        - No buffering or conversion is performed by this wrapper.
        """

        return self._reader.latest_frame()

    def _create_reader(self):
        try:
            import xsens_mvn_cpp_py
        except ImportError as exc:
            raise ImportError(
                "xsens_mvn_cpp_py is not importable. Build deploy/xsens_mvn_cpp "
                "with XSENS_MVN_CPP_BUILD_PYTHON=ON and add the build directory "
                "to PYTHONPATH."
            ) from exc

        return xsens_mvn_cpp_py.XsensMvnStreamReader(
            udp_port=self.udp_port,
            receive_timeout_ms=self.receive_timeout_ms,
            max_datagram_size=self.max_datagram_size,
        )
