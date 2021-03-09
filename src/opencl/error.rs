use opencl3::{device::DeviceInfo, program::ProgramInfo, types::cl_int};

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("Opencl3 Error: {0}")]
    Opencl3(cl_int),
    #[error("Device not found!")]
    DeviceNotFound,
    #[error("Device info not available!")]
    DeviceInfoNotAvailable(DeviceInfo),
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ProgramInfo),
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Cannot get bus ID for device with vendor {0}")]
    DeviceBusId(String),
}

#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<cl_int> for GPUError {
    fn from(error: cl_int) -> Self {
        GPUError::Opencl3(error)
    }
}
