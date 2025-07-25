use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};


#[derive(BufferContents, Vertex, Copy, Clone)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}