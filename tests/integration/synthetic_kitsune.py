"""Used only by test_integration.py::test_s3_timing_path_on_exact_ties, with distinct_first_packets=False: random
traffic in which identical first packets are common, which makes stage 3's known tie behaviour appear. (The
integration test itself uses the fully specified archive of edge_case_data.py.)

A tiny synthetic stand-in for the Kitsune archive, for tests only: the same layout (one folder per capture holding a
gzipped capture file and a gzipped labels file), real packets written with dpkt, about 15,000 packets in all.

Nothing here comes from the real data. The traffic is made up so that every stage has what it needs:
  - every known capture has benign flows (long camera-like TCP streams, short UDP queries) and attack flows (short SYN
    probes and long streams that differ in size, window and TTL), started at random times through the capture;
  - a few benign two-packet flows copy the header sequence of an attack probe exactly, so some sequences carry both
    labels (Track B removes them, MaxAcc counts them);
  - ARP MitM holds ARP-only flows (the "ARP MitM (ARP)" group) and relayed IP flows, attack in the later half;
  - Mirai is written as PCAPNG (as the real file is) with time-based labels: every packet after the midpoint is malicious;
  - Mirai's labels file is a bare column; the others have a header row and an index column, as in the real archive.
"""
import gzip
import io
import socket
import zipfile

import dpkt
import numpy as np

from kitsune_lib import CAPTURES

BASE_TIME = 1.5e9
DURATION = 600.0  # seconds per capture


def mac(i):
    return bytes([0x02, 0, 0, 0, i // 256, i % 256])


def ip4(i):
    return socket.inet_aton(f"10.0.{i // 250}.{i % 250 + 1}")


def tcp_frame(a, b, sport, dport, flags, win, payload, ttl):
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=win, off=5, data=payload)
    ip = dpkt.ip.IP(src=ip4(a), dst=ip4(b), p=dpkt.ip.IP_PROTO_TCP, ttl=ttl, data=seg)
    ip.len = 20 + len(bytes(seg))
    return bytes(dpkt.ethernet.Ethernet(src=mac(a), dst=mac(b), type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def udp_frame(a, b, sport, dport, payload, ttl):
    seg = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    seg.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(src=ip4(a), dst=ip4(b), p=dpkt.ip.IP_PROTO_UDP, ttl=ttl, data=seg)
    ip.len = 20 + len(bytes(seg))
    return bytes(dpkt.ethernet.Ethernet(src=mac(a), dst=mac(b), type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def arp_frame(a, b, op):
    arp = dpkt.arp.ARP(op=op, sha=mac(a), spa=ip4(a), tha=mac(b) if op == 2 else b"\x00" * 6, tpa=ip4(b))
    return bytes(dpkt.ethernet.Ethernet(src=mac(a), dst=mac(b), type=dpkt.ethernet.ETH_TYPE_ARP, data=arp))


class Capture:
    """Packets of one capture as (time, frame, label); flows get distinct source ports so each is its own flow."""

    def __init__(self, rng, host_base, start_port=20000, distinct=False):
        self.rng, self.rows, self.port, self.host, self.distinct = rng, [], start_port, host_base, distinct

    def _start(self):
        return BASE_TIME + self.rng.uniform(0, DURATION * 0.9)

    def _next_port(self):
        self.port += 1
        return self.port

    def stream(self, label, packets, size, win, ttl, dport=554, gap=0.05):
        """A long TCP flow: data from the server side, acknowledgements back."""
        t, sport, a, b = self._start(), self._next_port(), self.host, self.host + 1
        for k in range(packets):
            t += self.rng.exponential(gap)
            if k % 3 == 2:
                self.rows.append((t, tcp_frame(a, b, sport, dport, dpkt.tcp.TH_ACK, win, b"", ttl), label))
            else:
                n = int(max(1, self.rng.normal(size, size / 10)))
                self.rows.append((t, tcp_frame(b, a, dport, sport, dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, win, b"x" * n, ttl), label))

    def probe(self, label, dport, ttl=64, win=1024):
        """A two-packet SYN probe and its reset, as in a port scan."""
        t, sport = self._start(), self._next_port()
        if self.distinct:
            dport = sport  # a port no other flow's first packet goes to
        a, b = self.host + 2, self.host + 3
        self.rows.append((t, tcp_frame(a, b, sport, dport, dpkt.tcp.TH_SYN, win, b"", ttl), label))
        t += self.rng.exponential(0.002)
        self.rows.append((t, tcp_frame(b, a, dport, sport, dpkt.tcp.TH_RST | dpkt.tcp.TH_ACK, 0, b"", ttl), label))

    def query(self, label, packets=4):
        t, sport, a, b = self._start(), self._next_port(), self.host, self.host + 4
        for k in range(packets):
            t += self.rng.exponential(0.5)
            src, dst, sp, dp = (a, b, sport, 53) if k % 2 == 0 else (b, a, 53, sport)
            size = int(self.rng.integers(20, 60))
            if self.distinct and k == 0:
                size = 20 + (sport - 20000) % 1400  # a size no other query's first packet has
            self.rows.append((t, udp_frame(src, dst, sp, dp, b"q" * size, 64), label))

    def arp(self, label, packets):
        t, a, b = self._start(), self.host + 5, self.host + 6
        for k in range(packets):
            t += self.rng.exponential(0.2)
            self.rows.append((t, arp_frame(a, b, 1) if k % 2 == 0 else arp_frame(b, a, 2), label))

    def sorted(self):
        return sorted(self.rows, key=lambda r: r[0])


def build_capture(name, index, rng, distinct=True):
    # distinct: every capture uses its own range of client ports (at most 150 flows each), so no two flows share one
    c = Capture(rng, host_base=10 * index, start_port=20000 + 150 * index if distinct else 20000, distinct=distinct)
    for _ in range(10):
        c.stream(0, int(rng.integers(60, 160)), size=900, win=29200, ttl=64)
    for _ in range(6):
        c.query(0)
    if name == "ARP MitM":
        for _ in range(3):
            c.arp(0, int(rng.integers(40, 70)))
        for _ in range(4):
            c.stream(0, int(rng.integers(60, 120)), size=700, win=14600, ttl=63)
        rows = c.sorted()
        mid = BASE_TIME + DURATION / 2  # relayed traffic counts as attack in the later half
        return [(t, f, int(lab or (t > mid and f[12:14] == b"\x08\x00"))) for t, f, lab in rows]
    for _ in range(40):
        c.probe(1, dport=int(rng.integers(1, 1024)))
    for _ in range(3):
        c.probe(0, dport=int(rng.integers(1, 1024)))  # benign copies of a probe's headers: conflicting sequences
    for _ in range(6):
        c.stream(1, int(rng.integers(60, 120)), size=300 + 40 * index, win=8192, ttl=128, dport=443)
    rows = c.sorted()
    if name == "Mirai":  # time-based labels: everything after the midpoint is malicious
        mid = BASE_TIME + DURATION / 2
        rows = [(t, f, int(t > mid)) for t, f, _ in rows]
    return rows


def write_archive(path, seed=0, distinct_first_packets=True):
    """Write the archive and return the number of packets per capture.

    distinct_first_packets: every flow's first packet differs from every other flow's first packet in a field the main
    model reads (its destination port, and the size of the first packet of each UDP query). Without it, identical
    first packets are common in data this small, and stage 3's OOD limit (a high quantile of a few hundred validation
    values) can then be exactly equal to a test segment's score. At an exact tie, stage 3's packet-by-packet timing path
    (float64 running sums) and its evaluation (float32) can decide differently, and its consistency check stops the
    run. That behaviour of s3_evaluate.py is recorded by test_integration.py::test_s3_timing_path_on_exact_ties
    (expected to fail); it never occurred on the real data (100% agreement). The random draws are the same either way.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    with zipfile.ZipFile(path, "w") as z:
        for index, (name, folder) in enumerate(CAPTURES.items(), start=1):
            rows = build_capture(name, index, rng, distinct_first_packets)
            counts[name] = len(rows)
            raw = io.BytesIO()
            writer = dpkt.pcapng.Writer(raw) if name == "Mirai" else dpkt.pcap.Writer(raw)
            for t, frame, _ in rows:
                writer.writepkt(frame, ts=t)
            stem = name.replace(" ", "_")
            z.writestr(f"{folder}/{stem}_pcap.pcap.gz", gzip.compress(raw.getvalue()))
            labels = [str(lab) for _, _, lab in rows]
            text = "\n".join(labels) if name == "Mirai" else "\n".join([",x"] + [f"{i},{v}" for i, v in enumerate(labels)])
            z.writestr(f"{folder}/{stem}_labels.csv.gz", gzip.compress((text + "\n").encode()))
    return counts
