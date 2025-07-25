mod context;
mod visuals;
mod common;
mod buffers;

use std::ops::IndexMut;
use std::sync::Arc;

use visuals::Visuals;
use context::Context;
use buffers::Buffers;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::image::Image;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Framebuffer;
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;


struct AppState {
    context: Context,
    visuals: Visuals,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl AppState {
    pub fn new() -> Self {
        let context = Context::new();
        let (swapchain, images) = context.get_swapchain();
        let visuals = Visuals::new(context.device.clone(), swapchain.clone());

        let framebuffers = visuals.recreate_framebuffers(&images);
        let graphics_pipeline = visuals.recreate_graphics_pipeline(
            context.device.clone(),
            context.window.clone(),
        );

        Self {
            context,
            visuals,
            swapchain,
            images,
            graphics_pipeline,
            framebuffers,
        }
    }
}

pub struct App;

impl App {
    pub fn run() {
        let mut state = AppState::new();

        let buffers = Buffers::new(state.context.device.clone());

        let mut window_resized = false;
        let mut recreate_swapchain = false;


        let frames_in_flight = state.images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            state.context.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        let mut draw_command_buffers = buffers.get_draw_command_buffers(
            state.context.queue.clone(), 
            state.graphics_pipeline.clone(), 
            &state.framebuffers, &command_buffer_allocator, 
            buffers.vertex_buffer.clone()
        );


        let mut now = std::time::Instant::now();
        
        state.context.event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            }
            Event::MainEventsCleared => {

                let delta = now.elapsed();
                if delta.as_secs_f64() > 0.05 {
                    now = std::time::Instant::now();

                    match buffers.staging_buffer.write() {
                        Ok(mut buf) => {
                            buf.index_mut(0).position[0] += 0.01;
                        }
                        Err(e) => {
                            println!("failed to write buffer: {e}");
                        }
                    }

                    buffers.get_copy_command_buffer(&command_buffer_allocator, state.context.queue.clone())
                        .clone()
                        .execute(state.context.queue.clone())
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap()
                        .wait(None)
                        .unwrap();

                    
                }


                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;
                
                    let new_dimensions = state.context.window.inner_size();
                    (state.swapchain, state.images) = state.swapchain.recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(),
                        ..state.swapchain.create_info()
                    }).expect("failed to recreate swapchain: {e}");
                    state.framebuffers = state.visuals.recreate_framebuffers(&state.images);
                
                    if window_resized {
                        window_resized = false;
                        state.graphics_pipeline = state.visuals.recreate_graphics_pipeline(
                            state.context.device.clone(),
                            state.context.window.clone(),
                        );
                    }

                    draw_command_buffers = buffers.get_draw_command_buffers(
                        state.context.queue.clone(), 
                        state.graphics_pipeline.clone(), 
                        &state.framebuffers, &command_buffer_allocator, 
                        buffers.vertex_buffer.clone()
                    );
                }

                let (image_i, suboptimal, acquire_future) =
                    match vulkano::swapchain::acquire_next_image(state.swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };


                if suboptimal {
                    recreate_swapchain = true;
                }


                if let Some(image_fence) = &fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }

                let previous_future = match fences[previous_fence_i as usize].clone() {
                    // Create a `NowFuture`.
                    None => {
                        let mut now = sync::now(state.context.device.clone());
                        now.cleanup_finished();
                
                        now.boxed()
                    }
                    // Use the existing `FenceSignalFuture`.
                    Some(fence) => fence.boxed(),
                };

                let queue = state.context.queue.clone();

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(
                        state.context.queue.clone(), 
                        draw_command_buffers[image_i as usize].clone()
                    )
                    .unwrap()
                    .then_swapchain_present(
                        queue,
                        SwapchainPresentInfo::swapchain_image_index(state.swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => Some(Arc::new(value)),
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        None
                    }
                };

                previous_fence_i = image_i;


            }
            _ => (),
        });
    }
}



