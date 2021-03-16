mod error;
mod utils;

pub use error::*;
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::ptr;

// TODO vmx 2021-02-25: Don't leak `size_t` outside of `opencl3`. This is needed for the
// `cl_buffer_region` struct. There surely should be an abstraction.
use cl3::memory::CL_MEM_READ_WRITE;
use libc::{c_void, size_t};
// TODO vmx 2021-02-25: Should there be a higher level abstraction in `opencl` like there is in
// `ocl-core` (called `BufferRegion<T>`)
use cl3::types::cl_buffer_region;
// TODO vmx 2021-02-25: Don't use cl_sys directly, make sure `cl3` imports it. It seems that the
// only possible value for the `buffer_create_type` is `CL_BUFFER_CREATE_REGION`, so perhaps the
// `opencl3` `Buffer.create_sub_buffer()` might even just ignore that parameter and make `cl3`
// to hard code `CL_BUFFER_CREATE_REGION`
use cl_sys::CL_BUFFER_CREATE_TYPE_REGION;

pub type BusId = u32;

#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Brand {
    Amd,
    Apple,
    Nvidia,
}

impl Brand {
    pub fn platform_name(&self) -> &'static str {
        match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
            Brand::Apple => "Apple",
        }
    }

    fn all() -> Vec<Brand> {
        vec![Brand::Nvidia, Brand::Amd, Brand::Apple]
    }
}

pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<u8>,
    length: usize,
    queue: opencl3::command_queue::CommandQueue,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    /// The number of bytes / size_of(T)
    pub fn length(&self) -> usize {
        self.length
    }

    pub fn write_from(&mut self, offset: usize, data: &[T]) -> GPUResult<()> {
        assert!(offset + data.len() <= self.length());

        let buffer_create_info = cl_buffer_region {
            origin: (offset * std::mem::size_of::<T>()) as size_t,
            size: (data.len() * std::mem::size_of::<T>()) as size_t,
        };
        let buff = self.buffer.create_sub_buffer(
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_create_info as *const _ as *const c_void,
        )?;

        let data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        self.queue
            .enqueue_write_buffer(&buff, opencl3::types::CL_BLOCKING, 0, &data, &[])?;

        Ok(())
    }

    pub fn read_into(&self, offset: usize, data: &mut [T]) -> GPUResult<()> {
        assert!(offset + data.len() <= self.length());
        let buffer_create_info = cl_buffer_region {
            origin: (offset * std::mem::size_of::<T>()) as size_t,
            size: (data.len() * std::mem::size_of::<T>()) as size_t,
        };
        let buff = self.buffer.create_sub_buffer(
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_create_info as *const _ as *const c_void,
        )?;

        let mut data = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        self.queue
            .enqueue_read_buffer(&buff, opencl3::types::CL_BLOCKING, 0, &mut data, &[])?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    bus_id: Option<BusId>,
    pub device: opencl3::device::Device,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bus_id.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.bus_id == other.bus_id
    }
}

impl Eq for Device {}

impl Device {
    pub fn brand(&self) -> Brand {
        self.brand
    }
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn memory(&self) -> u64 {
        self.memory
    }
    pub fn is_little_endian(&self) -> GPUResult<bool> {
        match self.device.endian_little() {
            Ok(0) => Ok(false),
            Ok(_) => Ok(true),
            Err(_) => Err(GPUError::DeviceInfoNotAvailable(
                opencl3::device::DeviceInfo::CL_DEVICE_ENDIAN_LITTLE,
            )),
        }
    }
    pub fn bus_id(&self) -> Option<BusId> {
        self.bus_id
    }

    /// Return all available GPU devices of supported brands, ordered by brand as
    /// defined by `Brand::all()`.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn all_iter() -> impl Iterator<Item = &'static Device> {
        Brand::all()
            .into_iter()
            .filter_map(|brand| utils::DEVICES.get(&brand))
            .flatten()
    }

    pub fn by_bus_id(bus_id: BusId) -> GPUResult<&'static Device> {
        Device::all_iter()
            .find(|d| match d.bus_id {
                Some(id) => bus_id == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_brand(brand: Brand) -> Option<&'static Vec<Device>> {
        utils::DEVICES.get(&brand)
    }

    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GPUSelector {
    BusId(u32),
    Index(usize),
}

impl GPUSelector {
    pub fn get_bus_id(&self) -> Option<u32> {
        match self {
            GPUSelector::BusId(bus_id) => Some(*bus_id),
            GPUSelector::Index(index) => get_device_bus_id_by_index(*index),
        }
    }

    pub fn get_device(&self) -> Option<&'static Device> {
        match self {
            GPUSelector::BusId(bus_id) => Device::all_iter().find(|d| d.bus_id == Some(*bus_id)),
            GPUSelector::Index(index) => get_device_by_index(*index),
        }
    }

    pub fn get_key(&self) -> String {
        match self {
            GPUSelector::BusId(id) => format!("BusID: {}", id),
            GPUSelector::Index(idx) => {
                if let Some(id) = self.get_bus_id() {
                    format!("BusID: {}", id)
                } else {
                    format!("Index: {}", idx)
                }
            }
        }
    }
}

fn get_device_bus_id_by_index(index: usize) -> Option<BusId> {
    if let Some(device) = get_device_by_index(index) {
        device.bus_id
    } else {
        None
    }
}

fn get_device_by_index(index: usize) -> Option<&'static Device> {
    Device::all_iter().nth(index)
}

// TODO vmx 2021-02-26: Move to utils and re-export as here as public export
pub fn get_memory(d: &opencl3::device::Device) -> GPUResult<u64> {
    d.global_mem_size().map_err(|_| {
        GPUError::DeviceInfoNotAvailable(opencl3::device::DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE)
    })
}

pub struct Program {
    device: Device,
    context: opencl3::context::Context,
}

impl Program {
    pub fn device(&self) -> Device {
        self.device.clone()
    }
    pub fn from_opencl(device: Device, src: &str) -> GPUResult<Program> {
        let cached = utils::cache_path(&device, src)?;
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached)?;
            Program::from_binary(device, bin)
        } else {
            let mut context = opencl3::context::Context::from_device(device.device)?;
            let options = CString::default();
            let src_cstring = CString::new(src).expect("Program source contains a null byte");
            context.build_program_from_source(&src_cstring, &options)?;
            context.create_command_queues(0)?;
            let prog = Program { device, context };
            std::fs::write(cached, prog.to_binary()?)?;
            Ok(prog)
        }
    }
    pub fn from_binary(device: Device, bin: Vec<u8>) -> GPUResult<Program> {
        let mut context = opencl3::context::Context::from_device(device.device)?;
        let bins = vec![&bin[..]];
        let options = CString::default();
        context.build_program_from_binary(&bins, &options)?;
        context.create_command_queues(0)?;
        Ok(Program { device, context })
    }
    pub fn to_binary(&self) -> GPUResult<Vec<u8>> {
        match self.context.programs()[0].get_binaries() {
            Ok(bins) => Ok(bins[0].clone()),
            Err(_) => Err(GPUError::ProgramInfoNotAvailable(
                opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES,
            )),
        }
    }
    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buff = opencl3::memory::Buffer::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_WRITE,
            // TODO vmx 2021-03-15: multiplying with the memsize of the type seems to be wrong.
            // The `opencl3::memory::Buffer::create()` takes the number of elements and does
            // internally multiply with the memsize of the given test. Though `ocl` seems to do
            // the same thing, doing this multiplication makes the tests pass.
            length * std::mem::size_of::<T>(),
            ptr::null_mut(),
        )?;
        let queue = self.context.default_queue();
        queue.enqueue_write_buffer(&buff, opencl3::types::CL_BLOCKING, 0, &vec![0u8], &[])?;

        Ok(Buffer::<T> {
            buffer: buff,
            length,
            queue: queue.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn create_buffer_flexible<T>(&self, max_length: usize) -> GPUResult<Buffer<T>> {
        let mut curr = max_length;
        let mut step = max_length / 2;
        let mut n = 1;
        while step > 0 && n < max_length {
            if self.create_buffer::<T>(curr).is_ok() {
                n = curr;
                curr = std::cmp::min(curr + step, max_length);
            } else {
                curr -= step;
            }
            step = step / 2;
        }
        self.create_buffer::<T>(n)
    }
    pub fn create_kernel(&self, name: &str, gws: usize, lws: Option<usize>) -> Kernel {
        // TODO vmx 2021-03-01: Replace `unwrap()` with proper error handling
        let kernel = self
            .context
            .get_kernel(&CString::new(name).expect("Kernel name contains a null byte"))
            .unwrap();
        let mut builder = opencl3::kernel::ExecuteKernel::new(kernel);
        builder.set_global_work_size(gws);
        if let Some(lws) = lws {
            builder.set_local_work_size(lws);
        }
        Kernel {
            builder,
            queue: self.context.default_queue(),
        }
    }
}

pub trait KernelArgument<'a> {
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for &'a Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel.builder.set_arg(&self.buffer);
    }
}

impl KernelArgument<'_> for u32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

pub struct LocalBuffer<T> {
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelArgument<'_> for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .builder
            .set_arg_local_buffer(self.length * std::mem::size_of::<T>())
            .unwrap();
    }
}

#[derive(Debug)]
pub struct Kernel<'a> {
    builder: opencl3::kernel::ExecuteKernel<'a>,
    queue: &'a opencl3::command_queue::CommandQueue,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: T) -> Self {
        t.push(&mut self);
        self
    }
    pub fn run(mut self) -> GPUResult<()> {
        self.builder.enqueue_nd_range(&self.queue)?;
        Ok(())
    }
}

#[macro_export]
macro_rules! call_kernel {
    ($kernel:expr, $($arg:expr),*) => {{
        $kernel
        $(.arg($arg))*
        .run()
    }};
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_all() {
        for _ in 0..10 {
            let devices = Device::all();
            dbg!(&devices.len());
        }
    }
}
