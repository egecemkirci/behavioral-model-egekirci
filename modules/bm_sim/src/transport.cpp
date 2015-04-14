#include <iostream>
#include <memory>

#include <nanomsg/pubsub.h>

#include "bm_sim/transport.h"

int TransportSTDOUT::open(const std::string &name) {
  return 0;
}

int TransportSTDOUT::send(const std::string &msg) const {
  std::cout << msg << std::endl;
  return 0;
};

int TransportSTDOUT::send(const char *msg, int len) const {
  std::cout << msg << std::endl;
  return 0;
};

int TransportSTDOUT::send_msgs(
    const std::initializer_list<std::string> &msgs
) const
{
  for(const auto &msg : msgs) {
    send(msg);
  }
  return 0;
};

int TransportSTDOUT::send_msgs(
    const std::initializer_list<MsgBuf> &msgs
) const
{
  for(const auto &msg : msgs) {
    send(msg.buf, msg.len);
  }
  return 0;
};

TransportNanomsg::TransportNanomsg()
  : s(AF_SP, NN_PUB) {}

int TransportNanomsg::open(const std::string &name) {
  // TODO: catch exception
  s.bind(name.c_str());
  return 0;
}

int TransportNanomsg::send(const std::string &msg) const {
  s.send(msg.data(), msg.size(), 0);
  return 0;
};

int TransportNanomsg::send(const char *msg, int len) const {
  s.send(msg, len, 0);
  return 0;
};

// do not use this function, I think I should just remove it...
int TransportNanomsg::send_msgs(
    const std::initializer_list<std::string> &msgs
) const
{
  size_t num_msgs = msgs.size();
  if(num_msgs == 0) return 0;
  struct nn_msghdr msghdr;
  std::memset(&msghdr, 0, sizeof(msghdr));
  struct nn_iovec iov[num_msgs];
  int i = 0;
  for(const auto &msg : msgs) {
    // wish I could write this, but string.data() return const char *
    // iov[i].iov_base = msg.data();
    std::unique_ptr<char []> data(new char(msg.size()));
    msg.copy(data.get(), msg.size());
    iov[i].iov_base = data.get();
    iov[i].iov_len = msg.size();
    i++;
  }
  msghdr.msg_iov = iov;
  msghdr.msg_iovlen = num_msgs;
  s.sendmsg(&msghdr, 0);
  return 0;
};

int TransportNanomsg::send_msgs(
    const std::initializer_list<MsgBuf> &msgs
) const
{
  size_t num_msgs = msgs.size();
  if(num_msgs == 0) return 0;
  struct nn_msghdr msghdr;
  std::memset(&msghdr, 0, sizeof(msghdr));
  struct nn_iovec iov[num_msgs];
  int i = 0;
  for(const auto &msg : msgs) {
    iov[i].iov_base = msg.buf;
    iov[i].iov_len = msg.len;
    i++;
  }
  msghdr.msg_iov = iov;
  msghdr.msg_iovlen = num_msgs;
  s.sendmsg(&msghdr, 0);
  return 0;
};
