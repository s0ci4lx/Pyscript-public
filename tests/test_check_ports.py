import unittest
from unittest.mock import MagicMock, patch

from check_ports import port_is_available


class CheckPortsTests(unittest.TestCase):
    @patch("check_ports.socket.socket")
    def test_available_port(self, socket_factory):
        socket_factory.return_value.__enter__.return_value = MagicMock()
        self.assertTrue(port_is_available("127.0.0.1", 8081))

    @patch("check_ports.socket.socket")
    def test_occupied_port(self, socket_factory):
        sock = socket_factory.return_value.__enter__.return_value
        sock.bind.side_effect = OSError("in use")
        self.assertFalse(port_is_available("127.0.0.1", 8081))


if __name__ == "__main__":
    unittest.main()
