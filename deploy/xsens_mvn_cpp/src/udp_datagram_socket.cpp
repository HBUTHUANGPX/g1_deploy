#include "xsens_mvn_cpp/udp_datagram_socket.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace xsens_mvn_cpp
{

namespace
{
std::runtime_error socketError(const std::string& context)
{
  return std::runtime_error(context + ": " + std::strerror(errno));
}
}  // namespace

UdpDatagramSocket::~UdpDatagramSocket()
{
  closeSocket();
}

void UdpDatagramSocket::bindToPort(std::uint16_t port)
{
  if (port == 0)
  {
    throw std::invalid_argument("UDP port must be greater than zero");
  }
  if (isOpen())
  {
    throw std::logic_error("UDP socket is already open");
  }

  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0)
  {
    throw socketError("failed to create UDP socket");
  }

  int reuse = 1;
  if (::setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
  {
    closeSocket();
    throw socketError("failed to set SO_REUSEADDR");
  }

  sockaddr_in local_address{};
  local_address.sin_family = AF_INET;
  local_address.sin_addr.s_addr = INADDR_ANY;
  local_address.sin_port = htons(port);

  if (::bind(socket_fd_, reinterpret_cast<sockaddr*>(&local_address), sizeof(local_address)) < 0)
  {
    closeSocket();
    throw socketError("failed to bind UDP socket");
  }
}

void UdpDatagramSocket::setReceiveTimeout(int timeout_ms)
{
  if (!isOpen())
  {
    throw std::logic_error("cannot set timeout before binding socket");
  }
  if (timeout_ms < 0)
  {
    throw std::invalid_argument("receive timeout must be non-negative");
  }

  timeval timeout{};
  timeout.tv_sec = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  if (::setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0)
  {
    throw socketError("failed to set SO_RCVTIMEO");
  }
}

int UdpDatagramSocket::receive(char* buffer, std::size_t buffer_size)
{
  if (!isOpen())
  {
    throw std::logic_error("cannot receive before binding socket");
  }
  const auto bytes_received = ::recvfrom(socket_fd_, buffer, buffer_size, 0, nullptr, nullptr);
  if (bytes_received < 0)
  {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
    {
      return 0;
    }
    throw socketError("failed to receive UDP datagram");
  }
  return static_cast<int>(bytes_received);
}

void UdpDatagramSocket::closeSocket()
{
  if (socket_fd_ >= 0)
  {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

bool UdpDatagramSocket::isOpen() const
{
  return socket_fd_ >= 0;
}

}  // namespace xsens_mvn_cpp
