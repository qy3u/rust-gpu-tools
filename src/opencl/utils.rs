use std::collections::HashMap;

use cl3::{api_info_value, api_info_vector};
use lazy_static::lazy_static;
use log::{debug, warn};
use opencl3::error_codes::CL_SUCCESS;
use opencl3::types::{cl_int, cl_uint};
// TODO vmx 2021-02-26: Don't use cl_sys directly, but implement it in opecl3/cl3
use cl_sys::clGetDeviceInfo;
// TODO vmx 2021-02-26: This is needed for the `api_info_value` macro. Change the macro itself
// to use `std::mem` instead of `mem`. This won't be needed in case that `api_info_value`
// invocation gets directly implemented in `cl3`
use std::mem;

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

const AMD_DEVICE_VENDOR_STRING: &'static str = "AMD";
const NVIDIA_DEVICE_VENDOR_STRING: &'static str = "NVIDIA Corporation";

// Creates a new function called `get_device_info_bus_id` which takes two arguments, the device
// and the parameter ID.
api_info_value!(get_device_info_uint, cl_uint, clGetDeviceInfo);

// NOTE vmx 2021-02-26: This is the same as the `get_string` in `cl3/src/devices.rs`
api_info_vector!(get_device_info_bytes, u8, clGetDeviceInfo);

pub fn get_bus_id(d: &opencl3::device::Device) -> Result<u32, GPUError> {
    let vendor = d.vendor()?;
    match vendor.to_str() {
        Ok(AMD_DEVICE_VENDOR_STRING) => get_amd_bus_id(d).map_err(Into::into),
        Ok(NVIDIA_DEVICE_VENDOR_STRING) => get_nvidia_bus_id(d).map_err(Into::into),
        _ => Err(GPUError::DeviceBusId(
            vendor
                .to_str()
                .expect("Vendor is a valid UTF-8")
                .to_string(),
        )),
    }
}

pub fn get_nvidia_bus_id(d: &opencl3::device::Device) -> Result<u32, cl_int> {
    const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
    let result = get_device_info_uint(d.id(), CL_DEVICE_PCI_BUS_ID_NV.into())?;
    Ok(result)
}

pub fn get_amd_bus_id(d: &opencl3::device::Device) -> Result<u32, cl_int> {
    const CL_DEVICE_TOPOLOGY_AMD: u32 = 0x4037;
    let size = std::mem::size_of::<cl_amd_device_topology>();
    let result = get_device_info_bytes(d.id(), CL_DEVICE_TOPOLOGY_AMD, size)?;
    assert_eq!(result.len(), size);
    let mut topo = cl_amd_device_topology::default();
    unsafe {
        std::slice::from_raw_parts_mut(&mut topo as *mut cl_amd_device_topology as *mut u8, size)
            .copy_from_slice(&result);
    }
    Ok(topo.bus as u32)
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    // If there are multiple devices with the same name and neither has a Bus-Id,
    // then there will be a collision. Bus-Id can be missing in the case of an Apple
    // GPU. For now, we assume that in the unlikely event of a collision, the same
    // cache can be used.
    // TODO: We might be able to get around this issue by using cl_vendor_id instead of Bus-Id.
    hasher.input(device.name.as_bytes());
    if let Some(bus_id) = device.bus_id {
        hasher.input(bus_id.to_be_bytes());
    }
    hasher.input(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.result()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
}

lazy_static! {
    pub static ref PLATFORMS: Vec<opencl3::platform::Platform> =
        opencl3::platform::get_platforms().unwrap_or_default();
    pub static ref DEVICES: HashMap<Brand, Vec<Device>> = build_device_list();
}

pub fn find_platform(platform_name: &str) -> Result<Option<&opencl3::platform::Platform>, cl_int> {
    let platform = PLATFORMS.iter().find(|&p| match p.clone().name() {
        Ok(p) => {
            p == CString::new(platform_name.to_string())
                .expect("Platform string contains null byte")
        }
        Err(_) => false,
    });
    Ok(platform)
}

fn build_device_list() -> HashMap<Brand, Vec<Device>> {
    let brands = Brand::all();
    let mut map = HashMap::with_capacity(brands.len());

    for brand in brands.into_iter() {
        match find_platform(brand.platform_name()) {
            Ok(Some(platform)) => {
                let devices = platform
                    .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
                    .map_err(Into::into)
                    .and_then(|devices| {
                        devices
                            .into_iter()
                            .map(opencl3::device::Device::new)
                            .filter(|d| {
                                if let Ok(vendor) = d.vendor() {
                                    match vendor
                                        .to_str()
                                        .expect("Vendor string contains invalid UTF-8")
                                    {
                                        // Only use devices from the accepted vendors ...
                                        AMD_DEVICE_VENDOR_STRING | NVIDIA_DEVICE_VENDOR_STRING => {
                                            // ... which are available.
                                            return d.available().unwrap_or(0) != 0;
                                        }
                                        _ => (),
                                    }
                                }
                                false
                            })
                            .map(|d| -> GPUResult<_> {
                                Ok(Device {
                                    brand,
                                    name: d
                                        .name()?
                                        .into_string()
                                        .expect("Device name contains invalud UTF-8"),
                                    memory: get_memory(&d)?,
                                    bus_id: utils::get_bus_id(&d).ok(),
                                    device: d,
                                })
                            })
                            .collect::<GPUResult<Vec<_>>>()
                    });
                match devices {
                    Ok(devices) => {
                        map.insert(brand, devices);
                    }
                    Err(err) => {
                        warn!("Unable to retrieve devices for {:?}: {:?}", brand, err);
                    }
                }
            }
            Ok(None) => {}
            Err(err) => {
                warn!("Platform issue for brand {:?}: {:?}", brand, err);
            }
        }
    }

    debug!("loaded devices: {:?}", map);
    map
}
