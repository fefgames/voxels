use std::sync::Arc;

use vulkano::{command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer}, device::{Device, Queue}};

use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Framebuffer;
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};

use super::common::MyVertex;




fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[MyVertex]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                // Don't forget to write the correct buffer usage.
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                // .copy_buffer(CopyBufferInfo::buffers(
                //     staging_buf.clone(),
                //     vertex_buffer.clone(),
                // ))
                // .unwrap()
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass(SubpassEndInfo::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

const VERTICES: [MyVertex; 3] = [MyVertex { position: [-0.5, -0.5] }, MyVertex { position: [ 0.0,  0.5] }, MyVertex { position: [ 0.5, -0.25] }];

fn get_staging_buffer(memory_allocator: Arc<StandardMemoryAllocator>) -> Subbuffer<[MyVertex]> {

    Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        VERTICES,
    ).unwrap()
}

fn get_buffer(memory_allocator: Arc<StandardMemoryAllocator>) -> Subbuffer<[MyVertex]> {

    Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        VERTICES,
    )
        .unwrap()

}

/// This represents the buffers that are used to store the data for the graphics pipeline.
/// The staging buffer is used to store the data that is used to create the vertex buffer.
/// The vertex buffer is used to store the data that is used to draw the graphics.
pub struct Buffers {
    pub staging_buffer: Subbuffer<[MyVertex]>,
    pub vertex_buffer: Subbuffer<[MyVertex]>,
}

impl Buffers {
    pub fn new(device: Arc<Device>) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let staging_buffer = get_staging_buffer(memory_allocator.clone());
        let vertex_buffer = get_buffer(memory_allocator.clone());

        Self {
            staging_buffer,
            vertex_buffer,
        }
    }

    pub fn get_draw_command_buffers(
        &self,
        queue: Arc<Queue>,
        pipeline: Arc<GraphicsPipeline>,
        framebuffers: &Vec<Arc<Framebuffer>>,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        vertex_buffer: Subbuffer<[MyVertex]>,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {

        let draw_command_buffers = get_command_buffers(&command_buffer_allocator, queue.clone(), pipeline.clone(), framebuffers, vertex_buffer.clone());

        draw_command_buffers
    }

    pub fn get_copy_command_buffer(
        &self,
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: Arc<Queue>,
    ) -> Arc<PrimaryAutoCommandBuffer> {

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index() as u32,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.copy_buffer(CopyBufferInfo::buffers(
            self.staging_buffer.clone(),
            self.vertex_buffer.clone(),
        ))
        .unwrap();

        builder.build().unwrap().into()
    }
   
}