# Feature configuration for WIDS ML pipeline.
#
# These features were selected after profiling the complete
# training portion of the WPA3 dataset.

TARGET_COLUMN = "Label"

FEATURES = [
    # Frame characteristics
    "frame.len",
    "frame.time_delta",
    "frame.time_delta_displayed",

    # Radio / physical-layer characteristics
    "radiotap.channel.flags.cck",
    "radiotap.channel.flags.ofdm",
    "radiotap.channel.freq",
    "radiotap.length",

    # 802.11 frame-control characteristics
    "wlan.fc.frag",
    "wlan.fc.moredata",
    "wlan.fc.protected",
    "wlan.fc.pwrmgt",
    "wlan.fc.retry",
    "wlan.fc.subtype",
    "wlan.fc.type",

    # 802.11 / radio characteristics
    "wlan.duration",
    "wlan_radio.channel",
    "wlan_radio.frequency",
    "wlan_radio.phy",
    "wlan_radio.data_rate",
    "wlan_radio.duration",

    # Signal / transmission characteristics
    "radiotap.dbm_antsignal",
    "wlan_radio.signal_dbm",
    "radiotap.datarate",
]

# Features deliberately excluded from the first model:
#
# - Label: target variable
# - frame.number: packet/capture position
# - frame.time: absolute timestamp
# - frame.time_epoch: absolute timestamp
# - frame.time_relative: capture-relative timestamp
# - wlan.ra / wlan.ta / wlan.da / wlan.bssid: device identifiers
# - wlan.seq: sequence number
# - constant features
# - protocol-specific features that are completely missing
#
# Window-level traffic-rate features will be added later for
# the live WIDS/deauthentication-flood detection stage.
