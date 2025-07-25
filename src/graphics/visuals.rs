use std::sync::Arc;
use super::common::MyVertex;

use vulkano::{
    device::Device, 
    image::{
        view::ImageView, Image
    }, 
    pipeline::{
        graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::VertexDefinition, GraphicsPipeline, GraphicsPipelineCreateInfo}, 
        layout::PipelineDescriptorSetLayoutCreateInfo, PipelineLayout, PipelineShaderStageCreateInfo
    }, 
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass}, 
    shader::ShaderModule, 
    swapchain::Swapchain,
};

use winit::window::Window;

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}


fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
        .unwrap()
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}




fn get_graphics_pipeline(
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    window: Arc<Window>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
) -> Arc<GraphicsPipeline> {

    use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
    use vulkano::pipeline::graphics::vertex_input::Vertex;
    use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
    use vulkano::pipeline::GraphicsPipeline;
    use vulkano::render_pass::Subpass;

    // More on this latter.
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
    // let vs = vs::load(device.clone()).expect("failed to create shader module");
    // let fs = fs::load(device.clone()).expect("failed to create shader module");

    let pipeline = {
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify
        // which one.
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
            .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                // The stages of our pipeline, we have vertex and fragment stages.
                stages: stages.into_iter().collect(),
                // Describes the layout of the vertex input and how should it behave.
                vertex_input_state: Some(vertex_input_state),
                // Indicate the type of the primitives (the default is a list of triangles).
                input_assembly_state: Some(InputAssemblyState::default()),
                // Set the fixed viewport.
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                // Ignore these for now.
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                // This graphics pipeline object concerns the first pass of the render pass.
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
            .unwrap()
    };

    pipeline
}


/// This represents the visual components of the application.
pub struct Visuals {
    pub render_pass: Arc<RenderPass>,
    pub vs: Arc<ShaderModule>,
    pub fs: Arc<ShaderModule>,
}

impl Visuals {
    pub fn new(
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
    ) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module"); 
        let render_pass = get_render_pass(device.clone(), swapchain.clone());
        Self {
            render_pass,
            vs,
            fs,
        }
    }

    pub fn recreate_framebuffers(
        &self,
        images: &[Arc<Image>],
    ) -> Vec<Arc<Framebuffer>> {
        let framebuffers = get_framebuffers(&images, self.render_pass.clone());
        framebuffers
    }

    pub fn recreate_graphics_pipeline(
        &self,
        device: Arc<Device>,
        window: Arc<Window>,
    ) -> Arc<GraphicsPipeline> {
        get_graphics_pipeline(device.clone(), self.render_pass.clone(), window.clone(), self.vs.clone(), self.fs.clone())
    }
}
