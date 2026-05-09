from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("model/nids_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Original dataset feature names
feature_names = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

# Extract packet features
def extract_features(packet):

    packet_length = len(packet)

    protocol = 0

    if packet.haslayer(TCP):
        protocol = 1

    elif packet.haslayer(UDP):
        protocol = 2

    # Create 41 features
    feature_vector = [0] * 41

    # Fill important values
    feature_vector[0] = packet_length
    feature_vector[1] = protocol

    return feature_vector

# Process each packet
def process_packet(packet):

    try:

        # Extract features
        features = extract_features(packet)

        # Convert to dataframe
        df = pd.DataFrame(
            [features],
            columns=feature_names
        )

        # Scale data
        scaled_data = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_data)

        print("\n==============================")

        if packet.haslayer(IP):

            print("Source IP:", packet[IP].src)
            print("Destination IP:", packet[IP].dst)

        print("Packet Length:", len(packet))

        if prediction[0] == 0:
            print("Prediction: ATTACK DETECTED")
        else:
            print("Prediction: NORMAL TRAFFIC")

    except Exception as e:
        print("Error:", e)

# Start live packet sniffing
print("Starting Intrusion Detection System...")
print("Capturing 50 packets...\n")

sniff(
    prn=process_packet,
    store=False,
    count=50
)

print("\nPacket capture completed.")