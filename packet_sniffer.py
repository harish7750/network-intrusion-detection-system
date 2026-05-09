from scapy.all import sniff
from scapy.layers.inet import IP

def process_packet(packet):

    if packet.haslayer(IP):

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        packet_length = len(packet)

        print("\nPacket Detected")
        print("Source IP:", src_ip)
        print("Destination IP:", dst_ip)
        print("Protocol:", protocol)
        print("Packet Length:", packet_length)

sniff(
    prn=process_packet,
    count=20
)