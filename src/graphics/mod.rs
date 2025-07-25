mod context;
mod visuals;
mod common;
mod buffers;

extern crate nalgebra as na;

use std::ops::IndexMut;
use std::sync::Arc;
use visuals::Visuals;
use context::Context;
use buffers::Buffers;
use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        PrimaryCommandBufferAbstract,
    },
    image::Image,
    pipeline::GraphicsPipeline,
    render_pass::Framebuffer,
    swapchain::{Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture, future::FenceSignalFuture},
    Validated, VulkanError,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;


/// This represents the state of the application.
/// It stores the necessary components to render graphics
struct AppState {
    context: Context,
    visuals: Visuals,
    buffers: Buffers,
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

        let buffers = Buffers::new(context.device.clone());

        Self {
            context,
            visuals,
            swapchain,
            images,
            graphics_pipeline,
            framebuffers,
            buffers,
        }
    }
}

pub struct App;

impl App {
    pub fn run() {

        // start by setting up the stuff needed to run the application
        let mut state = AppState::new();


        // booleans we will use to track when we need to recreate the swapchain or the graphics pipeline
        let mut window_resized = false;
        let mut recreate_swapchain = false;

        // size of the swapchain
        let frames_in_flight = state.images.len();

        // fences to notify when the image is ready to be used
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;

        // we will use this to allow us to allocate command buffers
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            state.context.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );

        // the command buffers we will use to draw the graphics, we have multiple of these because we have multiple frames in the framebuffer
        let mut draw_command_buffers = state.buffers.get_draw_command_buffers(
            state.context.queue.clone(), 
            state.graphics_pipeline.clone(), 
            &state.framebuffers, &command_buffer_allocator, 
            state.buffers.vertex_buffer.clone()
        );


        // we will use this to track the time
        let mut now = std::time::Instant::now();
        // we will use this to track the position of the vertex
        let mut position = [-0.5, -0.5];

        // we will use this to 'remember' the last command buffer we executed for copying to the vertex buffer
        let mut copy_command_future: Option<FenceSignalFuture<_>> = None;
        
        // run the event loop
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

            // this is the main event loop, it will be called every time the main events are cleared
            Event::MainEventsCleared => {
                // time delta since the last moment we updated the position of the vertex
                let delta = now.elapsed();

                // if the time has passed 0.05 seconds, we will update the position of the vertex
                if delta.as_secs_f64() > 0.05 {
                    now = std::time::Instant::now();


                    // rotate the position of the vertex by 0.05 radians around the origin
                    let v = na::Vector2::new(position[0], position[1]);
                    let v = na::Rotation2::new(0.05).transform_vector(&v);
                    position = [v.x, v.y];

                    // if the previous command buffer has finished executing, we will write the new position to the vertex buffer
                    if copy_command_future.is_none() || copy_command_future.as_ref().unwrap().is_signaled().unwrap() {
                        match state.buffers.staging_buffer.write() {
                            Ok(mut buf) => {
                                buf.index_mut(0).position = position;
                            }
                            Err(e) => {
                                println!("failed to write buffer: {e}");
                            }
                        }
                        copy_command_future = Some(
                            state.buffers.get_copy_command_buffer(&command_buffer_allocator, state.context.queue.clone())
                            .clone()
                            .execute(state.context.queue.clone())
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap()
                        );
                    }
                }


                // if the window has been resized or the swapchain needs to be recreated, we will recreate the swapchain and the framebuffers
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

                    draw_command_buffers = state.buffers.get_draw_command_buffers(
                        state.context.queue.clone(), 
                        state.graphics_pipeline.clone(), 
                        &state.framebuffers, &command_buffer_allocator, 
                        state.buffers.vertex_buffer.clone()
                    );
                }


                // trying to get the next image from the swapchain
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


                // wait until the previous image at this index is ready to be overriden
                if let Some(image_fence) = &fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }


                // if the previous image exists, we will wait for it to be finished
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

                // execute the command buffer for this image, and present the result
                let future = previous_future
                    .join(acquire_future)
                    .then_execute(
                        state.context.queue.clone(), 
                        draw_command_buffers[image_i as usize].clone()
                    )
                    .unwrap()
                    .then_swapchain_present(
                        state.context.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(state.swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                // store the new fence for this image
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

                // update the index of the previous fence
                previous_fence_i = image_i;
            }
            _ => (),
        });
    }
}



