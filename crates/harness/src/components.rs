use crate::theme;
use eframe::egui::{
    self, Button, Color32, Frame, Margin, RichText, Stroke, WidgetInfo, WidgetType,
};

#[derive(Clone, Copy)]
pub enum MessageSurface {
    User,
    Assistant,
    Error,
}

pub fn composer_card(focused: bool) -> Frame {
    Frame::new()
        .fill(theme::SURFACE)
        .stroke(Stroke::new(
            if focused { 1.5 } else { 1.0 },
            if focused {
                theme::BANANA
            } else {
                theme::BORDER
            },
        ))
        .corner_radius(theme::RADIUS_LARGE)
        .inner_margin(Margin::same(theme::SPACE_12 as i8))
}

fn message_text_color(surface: MessageSurface) -> Color32 {
    match surface {
        MessageSurface::User => theme::SIDEBAR,
        MessageSurface::Assistant => theme::TEXT,
        MessageSurface::Error => theme::ERROR,
    }
}

fn message_text(surface: MessageSurface, text: &str) -> RichText {
    RichText::new(text).color(message_text_color(surface))
}

pub fn chat_bubble(
    ui: &mut egui::Ui,
    surface: MessageSurface,
    text: &str,
    align_right: bool,
) -> egui::Response {
    const MIN_MESSAGE_WIDTH: f32 = 144.0;
    const MAX_MESSAGE_WIDTH: f32 = 620.0;
    let (fill, stroke) = match surface {
        MessageSurface::User => (theme::BANANA, Stroke::NONE),
        MessageSurface::Assistant => (theme::SURFACE, Stroke::new(1.0, theme::BORDER)),
        MessageSurface::Error => (
            Color32::from_rgb(66, 38, 32),
            Stroke::new(1.0, theme::ERROR),
        ),
    };
    let frame = Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(theme::RADIUS_LARGE)
        .inner_margin(Margin::symmetric(14, 10));
    let content = |ui: &mut egui::Ui| {
        ui.add(
            egui::Label::new(message_text(surface, text))
                .wrap()
                .selectable(true),
        );
    };
    if align_right {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            ui.scope(|ui| {
                ui.set_min_width(MIN_MESSAGE_WIDTH);
                ui.set_max_width(MAX_MESSAGE_WIDTH);
                frame.show(ui, content).response
            })
            .inner
        })
        .inner
    } else {
        ui.scope(|ui| {
            ui.set_min_width(MIN_MESSAGE_WIDTH);
            ui.set_max_width(MAX_MESSAGE_WIDTH);
            frame.show(ui, content).response
        })
        .inner
    }
}

pub fn primary(label: impl Into<String>) -> Button<'static> {
    Button::new(RichText::new(label).color(theme::SIDEBAR).strong())
        .fill(theme::BANANA)
        .stroke(Stroke::NONE)
        .corner_radius(theme::RADIUS_SMALL)
}

pub fn secondary(label: impl Into<String>) -> Button<'static> {
    Button::new(RichText::new(label).color(theme::TEXT))
        .fill(theme::SURFACE_RAISED)
        .stroke(Stroke::new(1.0, theme::BORDER))
        .corner_radius(theme::RADIUS_SMALL)
}

pub struct NavigationItemResponse {
    pub select: egui::Response,
    pub delete: egui::Response,
}

pub fn navigation_item(
    ui: &mut egui::Ui,
    conversation_id: uuid::Uuid,
    label: &str,
    selected: bool,
) -> NavigationItemResponse {
    const DELETE_WIDTH: f32 = 34.0;
    let desired_size = egui::vec2(ui.available_width(), 38.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let select_rect =
        egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x - DELETE_WIDTH, rect.max.y));
    let id = ui.make_persistent_id(("conversation", conversation_id));
    let select = ui.interact(select_rect, id.with("select"), egui::Sense::click());
    select.widget_info(|| WidgetInfo::labeled(WidgetType::Button, ui.is_enabled(), label));
    if select.clicked() {
        select.request_focus();
    }
    let delete_rect =
        egui::Rect::from_min_max(egui::pos2(rect.max.x - DELETE_WIDTH, rect.min.y), rect.max);
    let delete = ui.interact(delete_rect, id.with("delete"), egui::Sense::click());
    delete.widget_info(|| {
        WidgetInfo::labeled(
            WidgetType::Button,
            ui.is_enabled(),
            format!("Delete chat: {label}"),
        )
    });
    let fill = if selected {
        theme::BANANA.gamma_multiply(0.32)
    } else if select.hovered() || delete.hovered() {
        theme::SURFACE_RAISED
    } else {
        Color32::TRANSPARENT
    };
    let stroke = if select.has_focus() || delete.has_focus() {
        Stroke::new(1.0, theme::BANANA)
    } else {
        Stroke::NONE
    };
    ui.painter().rect(
        rect,
        theme::RADIUS_SMALL,
        fill,
        stroke,
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        rect.left_center() + egui::vec2(theme::SPACE_12, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(15.0),
        theme::TEXT,
    );
    if selected || select.hovered() || delete.hovered() {
        ui.painter().text(
            delete_rect.center(),
            egui::Align2::CENTER_CENTER,
            "×",
            egui::FontId::proportional(18.0),
            theme::MUTED,
        );
    }
    NavigationItemResponse {
        select: select.on_hover_text(label),
        delete: delete.on_hover_text(format!("Delete chat: {label}")),
    }
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).color(theme::MUTED)
}

pub fn status(ui: &mut egui::Ui, online: bool, text: &str) {
    ui.horizontal(|ui| {
        let (dot_rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(
            dot_rect.center(),
            4.0,
            if online { theme::SUCCESS } else { theme::MUTED },
        );
        ui.label(muted(text));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bubbles_anchor_to_opposite_edges_of_a_full_width_transcript() {
        let context = egui::Context::default();
        let mut rects = None;
        let _ = context.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1_000.0, 600.0),
                )),
                ..Default::default()
            },
            |context| {
                egui::CentralPanel::default().show(context, |ui| {
                    let available = ui.max_rect();
                    let assistant =
                        chat_bubble(ui, MessageSurface::Assistant, "Incoming message", false);
                    ui.add_space(10.0);
                    let user = chat_bubble(ui, MessageSurface::User, "Outgoing message", true);
                    rects = Some((available, assistant.rect, user.rect));
                });
            },
        );

        let (available, assistant, user) = rects.expect("the test UI renders both bubbles");
        assert!(
            assistant.left() <= available.left() + 1.0,
            "incoming messages should start at the transcript's left edge: {assistant:?} in {available:?}"
        );
        assert!(
            user.right() >= available.right() - 1.0,
            "outgoing messages should end at the transcript's right edge: {user:?} in {available:?}"
        );
        assert!(
            user.width() < 620.0,
            "short outgoing messages should not expand to their maximum width"
        );
    }
}
