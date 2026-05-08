#ifndef DEPLOY_XSENS_MVN_CPP_UDP_DATAGRAM_SOCKET_H
#define DEPLOY_XSENS_MVN_CPP_UDP_DATAGRAM_SOCKET_H

#include <cstddef>
#include <cstdint>

namespace xsens_mvn_cpp
{

/**
 * @brief RAII UDP socket for receiving Xsens MVN datagrams.
 *
 * Preconditions:
 * - bindToPort() is called before receive().
 *
 * Postconditions:
 * - The socket descriptor is closed when the object is destroyed.
 */
class UdpDatagramSocket
{
public:
  /**
   * @brief Construct an unopened UDP socket wrapper.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - isOpen() returns false.
   */
  UdpDatagramSocket() = default;

  /**
   * @brief Close the socket if it is open.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Any owned socket descriptor is released.
   */
  ~UdpDatagramSocket();

  UdpDatagramSocket(const UdpDatagramSocket&) = delete;
  UdpDatagramSocket& operator=(const UdpDatagramSocket&) = delete;

  /**
   * @brief Bind the UDP socket to a local port.
   *
   * Preconditions:
   * - port is greater than 0.
   * - The socket is not already open.
   *
   * Postconditions:
   * - isOpen() returns true on success.
   */
  void bindToPort(std::uint16_t port);

  /**
   * @brief Configure receive timeout in milliseconds.
   *
   * Preconditions:
   * - The socket is open.
   * - timeout_ms is non-negative.
   *
   * Postconditions:
   * - receive() returns 0 on timeout.
   */
  void setReceiveTimeout(int timeout_ms);

  /**
   * @brief Receive one datagram into the provided buffer.
   *
   * Preconditions:
   * - The socket is open.
   * - buffer points to at least buffer_size bytes.
   *
   * Postconditions:
   * - Returns number of bytes received.
   * - Returns 0 on timeout.
   */
  int receive(char* buffer, std::size_t buffer_size);

  /**
   * @brief Close the socket.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - isOpen() returns false.
   */
  void closeSocket();

  /**
   * @brief Report whether a socket descriptor is currently open.
   *
   * Preconditions:
   * - None.
   *
   * Postconditions:
   * - Internal state is unchanged.
   */
  bool isOpen() const;

private:
  int socket_fd_ = -1;
};

}  // namespace xsens_mvn_cpp

#endif  // DEPLOY_XSENS_MVN_CPP_UDP_DATAGRAM_SOCKET_H
