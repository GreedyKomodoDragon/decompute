use protocol::{Acceleration, HardwareInfo};

pub fn detect(acceleration: Acceleration) -> HardwareInfo {
    decompute_llama::hardware_info(acceleration)
}
