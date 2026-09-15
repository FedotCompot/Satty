use std::cell::RefCell;

use anyhow::Result;
use femtovg::{Color, ImageFilter, ImageFlags, ImageId, Paint, Path, imgref::Img};

use relm4::Sender;

use crate::{
    math::{self, Vec2D},
    sketch_board::{MouseButton, MouseEventMsg, MouseEventType, SketchBoardInput},
    style::Style,
    tools::{RenderingMode, hit_test_rectangle},
};

use super::{
    Drawable, DrawableClone, Tool, ToolUpdateResult, Tools,
    drag_box::{DragBox, draw_center_marker},
};

#[derive(Clone, Debug)]
pub struct Blur {
    origin: Vec2D,
    top_left: Vec2D,
    size: Vec2D,
    style: Style,
    centered: bool,
    editing: bool,
    // (blurred image, and the image-space rect it was actually sampled from)
    cached_image: RefCell<Option<(ImageId, Vec2D, Vec2D)>>,
}

impl Blur {
    fn calculate_shape(&mut self, sender: &Sender<SketchBoardInput>, event: &MouseEventMsg) {
        let drag_box = DragBox::from_origin_delta(self.origin, self.size, event, sender);
        self.centered = drag_box.centered;
        self.top_left = drag_box.top_left;
        self.size = drag_box.size;
    }

    /// Returns `Ok(None)` when the rect does not overlap what this render target shows, which is
    /// the case for a blur belonging to another monitor in fullscreen="all". Otherwise it returns
    /// the blurred image and the image-space rect it was sampled from, which is clipped to the
    /// target and so can be smaller than the requested rect.
    fn blur(
        canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
        pos: Vec2D,
        size: Vec2D,
        sigma: f32,
    ) -> Result<Option<(ImageId, Vec2D, Vec2D)>> {
        let transform = canvas.transform();
        let scale = transform.average_scale().max(f32::EPSILON);
        let transformed_pos = transform.transform_point(pos.x, pos.y);
        let transformed_size = size * scale;

        // the queued render target only takes effect on flush, and screenshot() reads the bound
        // one: without this an export wider than the screen drops every blur past the screen edge
        canvas.flush();

        // clamp each edge independently; a forced minimum would break the `left + width <=
        // width()` bound that sub_image asserts on for a rect past the right or bottom edge
        let clamp = |v: f32, max: usize| (v.max(0.0) as usize).min(max);
        let target_w = canvas.width() as usize;
        let target_h = canvas.height() as usize;
        let left = clamp(transformed_pos.0, target_w);
        let top = clamp(transformed_pos.1, target_h);
        let right = clamp(transformed_pos.0 + transformed_size.x, target_w);
        let bottom = clamp(transformed_pos.1 + transformed_size.y, target_h);

        // bail before the expensive read-back when there is nothing to sample
        if right <= left || bottom <= top {
            return Ok(None);
        }

        let img = canvas.screenshot()?;

        // re-clamp: sub_image asserts against the screenshot, which need not match the canvas view
        let left = left.min(img.width());
        let top = top.min(img.height());
        let width = right.min(img.width()).saturating_sub(left);
        let height = bottom.min(img.height()).saturating_sub(top);
        if width == 0 || height == 0 {
            return Ok(None);
        }

        let (buf, width, height) = img.sub_image(left, top, width, height).to_contiguous_buf();
        let sub = Img::new(buf.into_owned(), width, height);

        let src_image_id = canvas.create_image(sub.as_ref(), ImageFlags::empty())?;
        let dst_image_id = canvas.create_image_empty(
            sub.width(),
            sub.height(),
            femtovg::PixelFormat::Rgba8,
            ImageFlags::empty(),
        )?;

        canvas.filter_image(
            dst_image_id,
            ImageFilter::GaussianBlur { sigma },
            src_image_id,
        );
        //canvas.delete_image(src_image_id);

        // map the sampled device rect back to image space so draw paints it 1:1 instead of
        // stretching a clipped sample across the whole rect
        let origin = transform.transform_point(0.0, 0.0);
        let to_image = |x: f32, y: f32| Vec2D::new((x - origin.0) / scale, (y - origin.1) / scale);
        let paint_pos = to_image(left as f32, top as f32);
        let paint_end = to_image((left + width) as f32, (top + height) as f32);

        Ok(Some((dst_image_id, paint_pos, paint_end - paint_pos)))
    }
}

impl Drawable for Blur {
    fn get_rendering_mode(&self) -> RenderingMode {
        RenderingMode::Blur
    }

    fn bounds(&self) -> Option<(Vec2D, Vec2D)> {
        Some(math::ensure_bounding_box(
            self.top_left,
            self.top_left + self.size,
        ))
    }

    fn hit_test(&self, pos: Vec2D, tolerance: f32) -> bool {
        hit_test_rectangle(pos, self.top_left, self.size, tolerance, true)
    }

    fn invalidate_gl_cache(&mut self) {
        *self.cached_image.borrow_mut() = None;
    }

    fn translate(&mut self, delta: Vec2D) {
        self.top_left += delta;
        // invalidate cached blur image since position changed
        *self.cached_image.borrow_mut() = None;
    }

    fn resize_bounds(&mut self, tl: Vec2D, br: Vec2D, _delta: Vec2D, _keep_aspect: bool) {
        let (tl, br) = math::ensure_bounding_box(tl, br);
        self.top_left = tl;
        self.size = br - tl;
        *self.cached_image.borrow_mut() = None;
    }

    fn get_style(&self) -> Option<&Style> {
        Some(&self.style)
    }

    fn get_style_mut(&mut self) -> Option<&mut Style> {
        *self.cached_image.borrow_mut() = None;
        Some(&mut self.style)
    }

    fn draw(
        &self,
        canvas: &mut femtovg::Canvas<femtovg::renderer::OpenGl>,
        _font: femtovg::FontId,
        bounds: (Vec2D, Vec2D),
    ) -> Result<()> {
        let size = self.size;
        let (pos, size) = math::rect_ensure_in_bounds(
            math::rect_ensure_positive_size(self.top_left, size),
            bounds,
        );
        if self.editing {
            // set style
            let mut color = Color::black();
            color.set_alphaf(0.6);
            let paint = Paint::color(color);

            // make rect
            let mut path = Path::new();
            path.rounded_rect(pos.x, pos.y, size.x, size.y, self.style.corner_radius());

            // draw
            canvas.fill_path(&path, &paint);
        } else {
            if size.x <= 0.0 || size.y <= 0.0 {
                return Ok(());
            }

            // left uncached when invisible here, so the full-image export recomputes it
            if self.cached_image.borrow().is_none() {
                let blurred = Self::blur(canvas, pos, size, self.style.blur_factor())?;
                if let Some(entry) = blurred {
                    self.cached_image.borrow_mut().replace(entry);
                }
            }

            if let Some((image_id, paint_pos, paint_size)) = *self.cached_image.borrow() {
                let mut path = Path::new();
                path.rounded_rect(pos.x, pos.y, size.x, size.y, self.style.corner_radius());

                canvas.fill_path(
                    &path,
                    &Paint::image(
                        image_id,
                        paint_pos.x,
                        paint_pos.y,
                        paint_size.x,
                        paint_size.y,
                        0f32,
                        1f32,
                    ),
                );
            }
        }

        if self.editing && self.centered {
            draw_center_marker(canvas, self.origin);
        }

        Ok(())
    }

    fn set_centered(&mut self, centered: bool) {
        self.centered = centered;
        self.origin = self.top_left + self.size / 2.0;
    }
    fn set_editing(&mut self, editing: bool) {
        self.editing = editing;
    }
}

#[derive(Default)]
pub struct BlurTool {
    blur: Option<Blur>,
    style: Style,
    input_enabled: bool,
    sender: Option<Sender<SketchBoardInput>>,
}

impl Tool for BlurTool {
    fn input_enabled(&self) -> bool {
        self.input_enabled
    }

    fn set_input_enabled(&mut self, value: bool) {
        self.input_enabled = value;
    }

    fn active(&self) -> bool {
        self.blur.is_some()
    }

    fn get_tool_type(&self) -> super::Tools {
        Tools::Blur
    }

    fn handle_mouse_event(&mut self, event: MouseEventMsg) -> ToolUpdateResult {
        match event.type_ {
            MouseEventType::BeginDrag => {
                if event.button == MouseButton::Middle {
                    return ToolUpdateResult::Unmodified;
                }

                // start new
                self.blur = Some(Blur {
                    origin: event.pos,
                    top_left: event.pos,
                    size: Vec2D::zero(),
                    style: self.style,
                    centered: false,
                    editing: true,
                    cached_image: RefCell::new(None),
                });

                ToolUpdateResult::Redraw
            }
            MouseEventType::EndDrag => {
                if event.button == MouseButton::Middle {
                    return ToolUpdateResult::Unmodified;
                }

                if let Some(a) = &mut self.blur {
                    if event.pos == Vec2D::zero() {
                        self.blur = None;

                        ToolUpdateResult::Redraw
                    } else {
                        a.calculate_shape(self.sender.as_ref().unwrap(), &event);
                        a.editing = false;

                        let result = a.clone_box();
                        self.blur = None;

                        ToolUpdateResult::Commit(result)
                    }
                } else {
                    ToolUpdateResult::Unmodified
                }
            }
            MouseEventType::UpdateDrag => {
                if event.button == MouseButton::Middle {
                    return ToolUpdateResult::Unmodified;
                }

                if let Some(a) = &mut self.blur {
                    if event.pos == Vec2D::zero() {
                        return ToolUpdateResult::Unmodified;
                    }
                    a.calculate_shape(self.sender.as_ref().unwrap(), &event);

                    ToolUpdateResult::Redraw
                } else {
                    ToolUpdateResult::Unmodified
                }
            }
            _ => ToolUpdateResult::Unmodified,
        }
    }

    fn handle_style_event(&mut self, style: Style) -> ToolUpdateResult {
        self.style = style;
        ToolUpdateResult::Unmodified
    }

    fn get_drawable(&self) -> Option<&dyn Drawable> {
        match &self.blur {
            Some(d) => Some(d),
            None => None,
        }
    }

    fn set_sender(&mut self, sender: Sender<SketchBoardInput>) {
        self.sender = Some(sender);
    }
}
