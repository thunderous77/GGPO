import importlib.util
import socket
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_network_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "areal" / "utils" / "network.py"
    spec = importlib.util.spec_from_file_location("areal_utils_network", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_ipv4_only_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv4_route"] and not ip_stack["ipv6_route"]):
        pytest.skip("Not an IPv4-only host (route-level)")
    network = _load_network_module()
    assert ":" not in network.gethostip()


def test_ipv6_only_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv6_route"] and not ip_stack["ipv4_route"]):
        pytest.skip("Not an IPv6-only host (route-level)")
    network = _load_network_module()
    assert ":" in network.gethostip()


def test_dual_stack_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv4_route"] and ip_stack["ipv6_route"]):
        pytest.skip("Not a dual-stack host (route-level)")
    network = _load_network_module()
    ip = network.gethostip()
    assert ip not in {"127.0.0.1", "::1"}


def test_gethostip_prefers_routed_ipv4_over_link_local_ipv6():
    network = _load_network_module()

    class FakeSocket:
        def __init__(self, family, _kind):
            self.family = family

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self, _address):
            assert self.family == socket.AF_INET

        def getsockname(self):
            return ("141.142.248.99", 43210)

    with patch.object(network.socket, "socket", FakeSocket):
        assert network.gethostip() == "141.142.248.99"


def test_gethostip_resolver_order_never_prefers_link_local_ipv6():
    network = _load_network_module()
    infos = [
        (socket.AF_INET6, socket.SOCK_DGRAM, 17, "", ("fe80::1", 0, 0, 7)),
        (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("172.28.80.99", 0)),
    ]

    with (
        patch.object(network.socket, "socket", side_effect=OSError("no route")),
        patch.object(network.socket, "gethostname", return_value="compute-node"),
        patch.object(network.socket, "getaddrinfo", return_value=infos),
    ):
        assert network.gethostip() == "172.28.80.99"


def test_split_hostport_accepts_unbracketed_ipv6():
    network = _load_network_module()
    host, port = network.split_hostport("2001:db8::1:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv4():
    network = _load_network_module()
    host, port = network.split_hostport("127.0.0.1:8000")
    assert host == "127.0.0.1"
    assert port == 8000


def test_split_hostport_accepts_bracketed_ipv6():
    network = _load_network_module()
    host, port = network.split_hostport("[2001:db8::1]:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_unbracketed():
    network = _load_network_module()
    host, port = network.split_hostport("fe80::1%eth0:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_bracketed():
    network = _load_network_module()
    host, port = network.split_hostport("[fe80::1%eth0]:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_format_hostport_brackets_ipv6():
    network = _load_network_module()
    assert network.format_hostport("2001:db8::1", 8000) == "[2001:db8::1]:8000"


def test_format_host_for_url_idempotent():
    network = _load_network_module()
    assert network.format_host_for_url("[2001:db8::1]") == "[2001:db8::1]"
    assert network.format_host_for_url("127.0.0.1") == "127.0.0.1"


def test_get_loopback_ip_returns_bindable_address():
    network = _load_network_module()
    ip = network.get_loopback_ip()
    assert ip in {"127.0.0.1", "::1"}
