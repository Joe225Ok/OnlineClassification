"""A small, fully specified Kitsune-like archive for the integration test: every flow is written out below, with no
random draws, so what the data stages must produce can be worked out by hand from this specification.

Layout and formats follow the real archive: one folder per capture with a gzipped capture file and a gzipped labels
file; captures in PCAPNG with microsecond timestamps (as the real files are), one in classic PCAP (the parser reads
both); labels with a header row and an index column, Mirai's as a bare column.

Edge cases, all in every known capture unless noted:
  - frames cut to 200 bytes (as in Kitsune): the IP length field, not the captured length, gives the packet size;
  - TCP, UDP, ICMP (one-packet flows), IPv6 (with a zero gap between two packets), a non-IP EtherType (LLDP), an 802.3
    length frame, and a frame too short to parse;
  - flows of 1, 2, 4, 30, 49, 50, 51, 60, 75, 80, 100, 110, 120 and 140 packets: segment boundaries at 50, 51, 100;
  - benign streams that share the attack streams' window, TTL and port (only their sizes differ), so some segments are
    hard to call and the models answer Medium;
  - attack flows whose first 10 packets are benign (their first segment is an attack segment that starts benign) and
    attack flows whose first 50 packets are benign (their first segment is benign);
  - identical sequences with both labels: the attack SYN probes and three benign copies of their headers; two copies
    differ in TTL (TTL separates them under "all inputs"), one is identical in every input;
  - held out: ARP MitM (ARP-only flows, the "ARP MitM (ARP)" group, and relayed IP streams, attack after the midpoint)
    and Mirai (another network; labels by time: every packet after the midpoint is malicious).
Every other sequence is distinct by construction (sizes, ports and the window of the client's ACKs are derived from
the capture and flow number). That matters beyond the counts: stage 3 checks its packet-by-packet timing path against
evaluation, and the two can decide differently when a segment's OOD score is exactly on a limit, which identical
segments make likely in data this small (a known behaviour of s3_evaluate.py, recorded by an expected-failure test).

write_archive() returns the specification: for each capture, its flows with their per-packet labels. Tests derive the
expected counts from it (packets, flows, segments and their labels); the order in which the parser numbers flows is not
assumed.
"""
import gzip
import io
import socket
import struct
import zipfile
from dataclasses import dataclass, field

import dpkt

from kitsune_lib import CAPTURES

L = 50
SNAP = 200  # bytes kept of each frame, as in the Kitsune captures
BASE_US = 1_500_000_000_000_000  # 1.5e9 s, in microseconds
CAPTURE_US = 600_000_000  # each capture spans 10 minutes
PCAP_CLASSIC = {"SYN DoS"}  # written as classic PCAP; the others as PCAPNG


@dataclass
class Flow:
    name: str
    labels: list = field(default_factory=list)  # one 0/1 per packet, in time order
    frames: list = field(default_factory=list)  # (microseconds since capture start, frame bytes)


def mac(c, f, side):
    return bytes([0x02, c, f // 256, f % 256, side, 0])


def ip4(c, f, side):
    return socket.inet_aton(f"10.{c}.{f % 250}.{side + 1}")


def ip6(c, f, side):
    return bytes(14) + bytes([c, f % 250]) if side == 0 else bytes(14) + bytes([c, 250 - f % 250])


def ether(c, f, a, b, etype, payload):
    return bytes(dpkt.ethernet.Ethernet(src=mac(c, f, a), dst=mac(c, f, b), type=etype, data=payload))


def ipv4(c, f, a, b, proto, l4, ttl, length=None):
    ip = dpkt.ip.IP(src=ip4(c, f, a), dst=ip4(c, f, b), p=proto, ttl=ttl, data=l4)
    ip.len = length or 20 + len(bytes(l4))
    return ether(c, f, a, b, dpkt.ethernet.ETH_TYPE_IP, ip)


def tcp(c, f, a, b, sport, dport, flags, win, ttl, ip_len=None):
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=win, off=5)
    if ip_len:  # a data packet: the payload is what the 200-byte snap length keeps of it
        seg.data = b"d" * (ip_len - 40)
    return ipv4(c, f, a, b, dpkt.ip.IP_PROTO_TCP, seg, ttl, ip_len)[:SNAP]


def udp(c, f, a, b, sport, dport, size, ttl=64):
    seg = dpkt.udp.UDP(sport=sport, dport=dport, data=b"q" * size)
    seg.ulen = 8 + size
    return ipv4(c, f, a, b, dpkt.ip.IP_PROTO_UDP, seg, ttl)


class Capture:
    def __init__(self, c):
        self.c, self.flows = c, []

    def flow(self, name):
        self.flows.append(Flow(name))
        return len(self.flows), self.flows[-1]  # flow number from 1: hosts, ports and sizes derive from it

    def stream(self, start, n, label, win, ttl, port, base_size, benign_first=0):
        """TCP stream: data packets from the server (cut to 200 bytes), every third packet an ACK from the client."""
        f, fl = self.flow("stream")
        sport, t = 20000 + 300 * self.c + f, start
        for k in range(n):
            t += 20_000 + 37 * (k % 11)  # whole microseconds
            if k % 3 == 2:  # the client's ACK advertises its own window: no two flows' ACKs are identical
                frame = tcp(self.c, f, 0, 1, sport, port, dpkt.tcp.TH_ACK, win - 4 * ((64 * self.c + f) % 640), ttl)
            else:
                frame = tcp(self.c, f, 1, 0, port, sport, dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, win, ttl,
                            ip_len=base_size + (f * 7 + k) % 50)
            fl.frames.append((t, frame))
            fl.labels.append(0 if k < benign_first else label)

    def probe(self, start, label, ttl=64, sport=None):
        """SYN to port 22 and its reset 100 microseconds later: the same headers for every probe. The client port is
        part of what the main model reads (the reply goes to it), so probes given the same port, in different
        captures (other addresses, so other flows), are identical in every input."""
        f, fl = self.flow("probe")
        sport = sport or 20000 + 300 * self.c + f
        fl.frames += [(start, tcp(self.c, f, 0, 1, sport, 22, dpkt.tcp.TH_SYN, 1024, ttl)),
                      (start + 100, tcp(self.c, f, 1, 0, 22, sport, dpkt.tcp.TH_RST | dpkt.tcp.TH_ACK, 0, ttl))]
        fl.labels += [label, label]

    def query(self, start, size_first):
        f, fl = self.flow("query")
        sport, t = 20000 + 300 * self.c + f, start
        for k in range(4):
            t += 300_000 + 1_000 * k
            a, b, sp, dp = (0, 1, sport, 53) if k % 2 == 0 else (1, 0, 53, sport)
            fl.frames.append((t, udp(self.c, f, a, b, sp, dp, size_first if k == 0 else 40 + k)))
            fl.labels.append(0)

    def icmp(self, start):
        f, fl = self.flow("icmp")
        echo = dpkt.icmp.ICMP(type=8, code=0, data=dpkt.icmp.ICMP.Echo(id=f, seq=1, data=b"ping"))
        fl.frames.append((start, ipv4(self.c, f, 0, 1, dpkt.ip.IP_PROTO_ICMP, echo, 64)))
        fl.labels.append(0)

    def ipv6(self, start, n=10):
        f, fl = self.flow("ipv6")
        t = start
        for k in range(n):
            t += 0 if k == 5 else 50_000  # packets 5 and 6 at the same microsecond: a zero gap
            seg = dpkt.udp.UDP(sport=5353, dport=5353, data=b"m" * (30 + k))
            seg.ulen = 8 + len(seg.data)
            pkt = dpkt.ip6.IP6(src=ip6(self.c, f, 0), dst=ip6(self.c, f, 1), nxt=dpkt.ip.IP_PROTO_UDP, hlim=255, data=seg)
            pkt.plen = len(bytes(seg))
            fl.frames.append((t, ether(self.c, f, 0, 1, dpkt.ethernet.ETH_TYPE_IP6, pkt)))
            fl.labels.append(0)

    def lldp(self, start):
        f, fl = self.flow("lldp")
        fl.frames.append((start, ether(self.c, f, 0, 1, 0x88CC, b"\x02\x07\x04" + bytes(6) + b"\x00\x00")))
        fl.labels.append(0)

    def ieee8023(self, start):
        f, fl = self.flow("802.3")
        llc = b"\x42\x42\x03" + bytes(35)  # spanning tree over LLC
        fl.frames.append((start, mac(self.c, f, 1) + mac(self.c, f, 0) + struct.pack("!H", len(llc)) + llc))
        fl.labels.append(0)

    def junk(self, start):
        f, fl = self.flow("unparseable")
        fl.frames.append((start, b"\x01\x02\x03\x04\x05\x06\x07\x08"))  # shorter than an Ethernet header
        fl.labels.append(0)

    def arp(self, start, n=60):
        f, fl = self.flow("arp")
        t = start
        for k in range(n):
            t += 200_000
            a, b, op = (0, 1, 1) if k % 2 == 0 else (1, 0, 2)
            pkt = dpkt.arp.ARP(op=op, sha=mac(self.c, f, a), spa=ip4(self.c, f, a),
                               tha=mac(self.c, f, b) if op == 2 else bytes(6), tpa=ip4(self.c, f, b))
            fl.frames.append((t, ether(self.c, f, a, b, dpkt.ethernet.ETH_TYPE_ARP, pkt)))
            fl.labels.append(0)


def slots(n):
    """Start times, in microseconds from the capture start, spread evenly over the first 70% of the capture."""
    return [int(CAPTURE_US * 0.7 * (j + 0.5) / n) for j in range(n)]


# Sized so that the test split holds about 200 segments. Stage 3 stops when its packet-by-packet timing path agrees
# with evaluation on fewer than 99% of the timed segments; with about 100 test segments, one borderline segment (an OOD
# score next to a limit, decided in float64 by one path and float32 by the other) was enough to stop it on some
# machines. With about 200, one such segment is tolerated, as on the real data. Every kind of flow appears twice as
# often; the totals of the plan below (110 flows) must not be a multiple of 37, the step that interleaves them in time.
BENIGN_STREAMS = [50, 51, 100, 120, 75, 60, 140, 30] * 2
ATTACK_STREAMS = [(60, 0), (110, 10), (50, 0), (80, 50)] * 2  # (packets, leading benign packets)
LOOKALIKE_STREAMS = [70] * 2  # benign streams that share the attack streams' window, TTL and port
ATTACK_PROBES, QUERIES = 60, 15
PROBE_PORT = 40000


def known_capture(c):
    cap = Capture(c)
    # every kind of flow is spread over the capture, so every class has early and late segments
    plan = ([("stream", n) for n in BENIGN_STREAMS] + [("attack stream", s) for s in ATTACK_STREAMS]
            + [("lookalike", n) for n in LOOKALIKE_STREAMS]
            + [("probe", 1)] * ATTACK_PROBES + [("probe copy", 0)] * 3 + [("query", q) for q in range(QUERIES)]
            + [("icmp",), ("icmp",), ("ipv6",), ("lldp",), ("802.3",), ("unparseable",)])
    order = sorted(range(len(plan)), key=lambda j: (j * 37) % len(plan))  # interleave the kinds in time
    for start, j in zip(slots(len(plan)), order):
        kind = plan[j]
        if kind[0] == "stream":
            cap.stream(start, kind[1], 0, win=29200, ttl=64, port=554, base_size=800 + 10 * c)
        elif kind[0] == "lookalike":  # benign, with the attack streams' window, TTL and port, other sizes
            cap.stream(start, kind[1], 0, win=8192, ttl=128, port=443, base_size=600 + 10 * c)
        elif kind[0] == "attack stream":
            n, benign_first = kind[1]
            cap.stream(start, n, 1, win=8192, ttl=128, port=443, base_size=300 + 10 * c, benign_first=benign_first)
        elif kind[0] == "probe":
            attacks = sum(1 for fl in cap.flows if fl.name == "probe" and fl.labels[0] == 1)
            cap.probe(start, 1, sport=PROBE_PORT + attacks)  # attack probe k of every capture uses port 40000 + k
        elif kind[0] == "probe copy":
            copies = sum(1 for fl in cap.flows if fl.name == "probe" and fl.labels[0] == 0)
            if copies == 0:  # identical in every input to attack probe 0 (of every capture)
                cap.probe(start, 0, ttl=64, sport=PROBE_PORT)
            else:  # the same headers, but TTL 128: "all inputs" tells them apart
                cap.probe(start, 0, ttl=128, sport=PROBE_PORT + ATTACK_PROBES + copies)
        elif kind[0] == "query":
            cap.query(start, size_first=100 + 20 * c + kind[1])
        else:
            getattr(cap, {"icmp": "icmp", "ipv6": "ipv6", "lldp": "lldp", "802.3": "ieee8023",
                          "unparseable": "junk"}[kind[0]])(start)
    return cap.flows


def arp_mitm(c):
    cap = Capture(c)
    # relayed streams before and after the midpoint (benign, then attack); slots 8 and 10 start after it
    plan = [("arp",), ("relay", 80), ("arp",), ("relay", 90), ("query", 0), ("query", 1), ("arp",), ("query", 2),
            ("relay", 100), ("query", 3), ("relay", 110)]
    plan = [kind for kind in plan for _ in range(2)]  # each twice, in place: the last two relays still start after it
    for start, kind in zip(slots(len(plan)), plan):
        if kind[0] == "arp":
            cap.arp(start)
        elif kind[0] == "relay":
            cap.stream(start, kind[1], 0, win=14600, ttl=63, port=8080, base_size=700)
        else:
            cap.query(start, size_first=400 + kind[1])
    mid = CAPTURE_US // 2  # relayed IP packets after the midpoint are the attack
    for fl in cap.flows:
        if fl.name == "stream":
            fl.labels = [int(t > mid) for t, _ in fl.frames]
    return cap.flows


def mirai(c):
    cap = Capture(c)
    plan = [("stream", n) for n in (70, 90, 120, 60)] + [("probe",)] * 20 + [("query", q) for q in range(4)]
    plan = [kind for kind in plan for _ in range(2)]  # each twice, in place: before and after the midpoint alike
    for start, kind in zip(slots(len(plan)), plan):
        if kind[0] == "stream":
            cap.stream(start, kind[1], 0, win=5840, ttl=64, port=80, base_size=500)
        elif kind[0] == "probe":
            cap.probe(start, 0, ttl=200)
        else:
            cap.query(start, size_first=500 + kind[1])
    mid = CAPTURE_US // 2  # time-based labels, as in the real Mirai capture
    for fl in cap.flows:
        fl.labels = [int(t > mid) for t, _ in fl.frames]
    return cap.flows


def build(name, c):
    return {"ARP MitM": arp_mitm, "Mirai": mirai}.get(name, known_capture)(c)


def write_archive(path):
    """Write the archive; return {capture: [flow, ...]} with each flow's per-packet labels (the specification)."""
    spec = {}
    with zipfile.ZipFile(path, "w") as z:
        for c, (name, folder) in enumerate(CAPTURES.items(), start=1):
            flows = build(name, c)
            spec[name] = flows
            packets = sorted(((t, frame, lab) for fl in flows for (t, frame), lab in zip(fl.frames, fl.labels)),
                             key=lambda p: p[0])
            raw = io.BytesIO()
            writer = dpkt.pcap.Writer(raw) if name in PCAP_CLASSIC else dpkt.pcapng.Writer(raw)
            for t, frame, _ in packets:
                us = BASE_US + c * 10 * CAPTURE_US + t
                writer.writepkt(frame, ts=us / 1e6)
            stem = name.replace(" ", "_")
            z.writestr(f"{folder}/{stem}_pcap.pcap.gz", gzip.compress(raw.getvalue()))
            labels = [str(lab) for _, _, lab in packets]
            text = "\n".join(labels) if name == "Mirai" else "\n".join([",x"] + [f"{i},{v}" for i, v in enumerate(labels)])
            z.writestr(f"{folder}/{stem}_labels.csv.gz", gzip.compress((text + "\n").encode()))
    return spec
