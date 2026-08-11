use protocol::{Acceleration, HardwareInfo};

pub fn detect(acceleration: Acceleration) -> HardwareInfo {
    inference::hardware_info(acceleration)
}
