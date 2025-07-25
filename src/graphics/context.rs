use vulkano::image::{Image, ImageUsage};
use vulkano::VulkanLibrary;
use vulkano::device::DeviceExtensions;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};
use vulkano::instance::Instance;
use vulkano::device::Device;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{DeviceCreateInfo, QueueCreateInfo, Queue, QueueFlags};
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::instance::{InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::{SwapchainCreateInfo};

/// This creates the swapchain and images based on the physical device, device, surface, and window.
fn get_swapchain(physical_device: Arc<PhysicalDevice>, device: Arc<Device>, surface: Arc<Surface>, window: Arc<Window>) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("failed to get surface capabilities");

    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format =  physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;


    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1, // How many buffers to use in the swapchain
            // min_image_count: caps.min_image_count, // How many buffers to use in the swapchain
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    (swapchain, images)
}
fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,

            // Note that there exists `PhysicalDeviceType::Other`, however,
            // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
            // match wildcard `_` to catch all unknown device types.
            _ => 4,
        })
        .expect("no device available")
}



/// This selects the physical device and queue family index based on the surface and instance.
fn get_physical_device(surface: Arc<Surface>, instance: Arc<Instance>) -> (Arc<PhysicalDevice>, u32, DeviceExtensions) {


    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };


    let (physical_device, queue_family_index) = select_physical_device(&instance, &surface, &device_extensions);
    (physical_device, queue_family_index, device_extensions)
}

///Window is the window that the application is running in.
fn get_window<T>(event_loop: &EventLoop<T>) -> Arc<Window> {
    Arc::new(WindowBuilder::new().build(event_loop).unwrap())
}

/// This creates the device and queue interface based on the physical device properties.
fn get_device_and_queue(physical_device: Arc<PhysicalDevice>, queue_family_index: u32, device_extensions: DeviceExtensions) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
        .expect("failed to create device");

    let queue = queues.next().expect("device had no queues!");


    (device, queue)

}

///Instance is the connection between the application and the Vulkan library.
fn get_instance<T>(event_loop: &EventLoop<T>) -> Arc<Instance> {

    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
        .expect("failed to create instance");

    instance
}

/// Context is meant to represent the environtment the application is running in.
pub struct Context {
    pub window: Arc<Window>,
    pub surface: Arc<Surface>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub event_loop: EventLoop<()>,
}

impl Context {
    pub fn get_swapchain(&self) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        get_swapchain(self.physical_device.clone(), self.device.clone(), self.surface.clone(), self.window.clone())
    }
    pub fn new() -> Self {
        let event_loop = EventLoop::new();
        let window = get_window(&event_loop);
        let instance = get_instance(&event_loop);
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let (physical_device, queue_family_index, device_extensions) = get_physical_device(surface.clone(), instance.clone());
        let (device, queue) = get_device_and_queue(physical_device.clone(), queue_family_index, device_extensions);

        Self {
            window,
            surface,
            physical_device,
            device,
            queue,
            event_loop,
        }
    }
}